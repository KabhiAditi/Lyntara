"""
Pipeline service: schema extraction → context generation → Qdrant upsert.

Ported from View_Selection notebooks:
  - schema_extractor.ipynb
  - context_gen.ipynb
  - upsert.ipynb
"""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncGenerator
import time

import psycopg2
import httpx
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VIEW_DIR = _REPO_ROOT / "View_Selection"

# ──────────────────────────────────────────────
# 1. SCHEMA EXTRACTION  (from schema_extractor.ipynb)
# ──────────────────────────────────────────────

def _pg_connect(host: str, port: int, database: str, user: str, password: str):
    return psycopg2.connect(
        dbname=database, user=user, password=password,
        host=host, port=str(port),
    )


def _get_tables_and_columns(conn) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
          AND c.table_name NOT IN (
              SELECT child.relname
              FROM pg_inherits i
              JOIN pg_class child ON child.oid = i.inhrelid
              JOIN pg_namespace n ON n.oid = child.relnamespace
              WHERE n.nspname = 'public'
          )
        ORDER BY table_name, ordinal_position;
    """)
    schema: dict = {}
    for table, column, dtype in cur.fetchall():
        if table not in schema:
            schema[table] = {"columns": [], "primary_keys": [], "foreign_keys": []}
        schema[table]["columns"].append({"column_name": column, "data_type": dtype})
    return schema


def _get_primary_keys(conn, schema: dict) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = 'public';
    """)
    for table, column in cur.fetchall():
        if table in schema and column not in schema[table]["primary_keys"]:
            schema[table]["primary_keys"].append(column)
    return schema


def _get_foreign_keys(conn, schema: dict) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT tc.table_name, kcu.column_name,
               ccu.table_name AS foreign_table,
               ccu.column_name AS foreign_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
    """)
    for table, column, f_table, f_column in cur.fetchall():
        if table not in schema:
            continue
        fk = {"column": column, "references_table": f_table, "references_column": f_column}
        existing = schema[table]["foreign_keys"]
        if not any(
            e["column"] == column and e["references_table"] == f_table and e["references_column"] == f_column
            for e in existing
        ):
            existing.append(fk)
    return schema


def _mark_indexed_columns(conn, schema: dict) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT t.relname AS table_name, a.attname AS column_name
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE t.relkind = 'r';
    """)
    indexed_set = set(cur.fetchall())
    for table in schema:
        for col in schema[table]["columns"]:
            col["is_indexed"] = (table, col["column_name"]) in indexed_set
    return schema


def _attach_sample_rows(conn, schema: dict, limit: int = 2) -> dict:
    cur = conn.cursor()
    for table, info in schema.items():
        columns = [c["column_name"] for c in info["columns"]]
        if not columns:
            info["sample_rows"] = []
            continue
        col_list_sql = ", ".join(f'"{c}"' for c in columns)
        query = f'SELECT {col_list_sql} FROM "{table}" LIMIT %s'
        try:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
        except Exception:
            conn.rollback()
            info["sample_rows"] = []
            continue
        samples = []
        for row in rows:
            sample = {}
            for i, col in enumerate(columns):
                val = row[i]
                if isinstance(val, Decimal):
                    val = float(val)
                elif isinstance(val, (datetime, date)):
                    val = val.isoformat()
                sample[col] = val
            samples.append(sample)
        info["sample_rows"] = samples
    return schema


def extract_schema(host: str, port: int, database: str, user: str, password: str) -> dict:
    """Full schema extraction pipeline. Returns schema dict and saves to disk."""
    conn = _pg_connect(host, port, database, user, password)
    try:
        schema = _get_tables_and_columns(conn)
        schema = _get_primary_keys(conn, schema)
        schema = _get_foreign_keys(conn, schema)
        schema = _mark_indexed_columns(conn, schema)
        schema = _attach_sample_rows(conn, schema)
    finally:
        conn.close()

    # Save per-database schema JSON
    out_path = _VIEW_DIR / f"{database}_schema.json"
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)

    return schema


# ──────────────────────────────────────────────
# 2. CONTEXT GENERATION  (from context_gen.ipynb)
# ──────────────────────────────────────────────

def generate_context(schema: dict, database: str, on_table_done=None) -> dict:
    """
    Enrich each table with Gemini-generated descriptions.
    Saves to {database}_context.json AND copies to context.json.
    on_table_done(table_name, index, total) callback for progress.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Cannot generate context.")

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    content: dict = {}
    tables = list(schema.keys())
    total = len(tables)

    for idx, key in enumerate(tables):
        value = schema[key]
        
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"""Act as a database documentation assistant. I will provide a table name (`key`) and its JSON metadata (`value`) which includes columns, primary_keys, foreign_keys, and sample_rows.

        Your tasks:
        1. Generate a concise, one-sentence description of the table's purpose, inside it MAKE SURE TO mention all of the column names at the end. Add this as a new key called "table_description" at the root level of the JSON object.
        2. Generate a brief description for each column. Add this as a new key called "description" inside each respective column's object within the "columns" section.
        3. Retain all original data (sample rows, keys, etc.).

        Output Requirements:
        Return ONLY the updated JSON object (the value). Do not wrap it in the original table name key, and do not include any markdown formatting, conversational text, or explanations.

        Table Name: {key}
        Original JSON Value: {value}""",
                    config=types.GenerateContentConfig(
                        # thinking_config=types.ThinkingConfig(thinking_level="low"),
                        response_mime_type="application/json"
                    ),
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                else:
                    print(f"[context_gen] API call failed for '{key}' after {max_retries} retries: {e}")
                    raise e

        try:
            enriched = json.loads(response.text) if response else value
            content[key] = enriched
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[context_gen] Failed to parse JSON for table '{key}': {e}")
            # Fallback: keep raw schema without enrichment
            content[key] = value

        if on_table_done:
            on_table_done(key, idx + 1, total)
        
        # Add a short delay to avoid overwhelming the API rate limits
        time.sleep(2)

    # Save per-database context
    db_context_path = _VIEW_DIR / f"{database}_context.json"
    with open(db_context_path, "w") as f:
        json.dump(content, f, indent=4)

    # Also write to the canonical context.json used by SchemaRetriever
    canonical_path = _VIEW_DIR / "context.json"
    with open(canonical_path, "w") as f:
        json.dump(content, f, indent=4)

    return content


# ──────────────────────────────────────────────
# 3. QDRANT UPSERT  (from upsert.ipynb)
# ──────────────────────────────────────────────

def _qdrant_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _collection_exists(base_url: str, name: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/collections/{name}", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _delete_collection(base_url: str, name: str):
    try:
        httpx.delete(f"{base_url}/collections/{name}", timeout=5.0)
    except Exception:
        pass


def _create_collection(base_url: str, name: str, vector_size: int = 384):
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }
    r = httpx.put(f"{base_url}/collections/{name}", json=payload, timeout=10.0)
    r.raise_for_status()


def upsert_to_qdrant(
    context: dict,
    database: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> None:
    """Create collections and upsert table + column embeddings."""
    base_url = _qdrant_base_url(qdrant_host, qdrant_port)
    table_col_name = f"{database}-table"
    column_col_name = f"{database}-table-column"

    # Load embedding model (read from env so it stays in sync with SchemaRetriever)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"[upsert] Using embedding model '{embedding_model_name}' — vector size: {vector_size}")

    # Delete old collections if they exist (for demo replayability)
    _delete_collection(base_url, table_col_name)
    _delete_collection(base_url, column_col_name)

    # Create fresh collections with dynamic vector size
    _create_collection(base_url, table_col_name, vector_size=vector_size)
    _create_collection(base_url, column_col_name, vector_size=vector_size)

    points_table = []
    points_column = []
    col_id = 0

    for idx, (key, value) in enumerate(context.items()):
        # Table-level embedding
        table_text = f"Table: {key}. Description: {value.get('table_description', '')}."
        table_vector = model.encode(table_text).tolist()
        points_table.append({
            "id": idx,
            "vector": table_vector,
            "payload": {
                "table": key,
                "description": value.get("table_description", ""),
                "primary_keys": value.get("primary_keys", []),
                "foreign_keys": value.get("foreign_keys", []),
            }
        })

        # Column-level embeddings
        for col in value.get("columns", []):
            col_text = (
                f"Column: {col.get('column_name', '')}. "
                f"Table: {key}. "
                f"Description: {col.get('description', '')}. "
                f"Data type: {col.get('data_type', '')}."
            )
            col_vector = model.encode(col_text).tolist()
            points_column.append({
                "id": col_id,
                "vector": col_vector,
                "payload": {
                    "table": key,
                    "column": col.get("column_name", ""),
                    "datatype": col.get("data_type", ""),
                    "indexed": col.get("is_indexed", False),
                    "description": col.get("description", ""),
                }
            })
            col_id += 1

    # Upsert in batches via REST API
    if points_table:
        r = httpx.put(
            f"{base_url}/collections/{table_col_name}/points",
            json={"points": points_table},
            timeout=30.0,
        )
        r.raise_for_status()

    if points_column:
        r = httpx.put(
            f"{base_url}/collections/{column_col_name}/points",
            json={"points": points_column},
            timeout=30.0,
        )
        r.raise_for_status()

    # Create payload index on "table" field in column collection
    httpx.put(
        f"{base_url}/collections/{column_col_name}/index",
        json={"field_name": "table", "field_schema": "keyword"},
        timeout=10.0,
    )


# ──────────────────────────────────────────────
# 4. REDUNDANCY CHECK
# ──────────────────────────────────────────────

def check_pipeline_done(
    database: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> bool:
    """Return True if collection + context JSON already exist for this database."""
    base_url = _qdrant_base_url(qdrant_host, qdrant_port)
    collection_exists = _collection_exists(base_url, f"{database}-table")
    context_exists = (_VIEW_DIR / f"{database}_context.json").exists()
    return collection_exists and context_exists


# ──────────────────────────────────────────────
# 5. DELETE COLLECTIONS (demo reset)
# ──────────────────────────────────────────────

def delete_collections(
    database: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> None:
    """Remove Qdrant collections and cached JSON for given database."""
    base_url = _qdrant_base_url(qdrant_host, qdrant_port)
    _delete_collection(base_url, f"{database}-table")
    _delete_collection(base_url, f"{database}-table-column")

    for suffix in ("_schema.json", "_context.json"):
        p = _VIEW_DIR / f"{database}{suffix}"
        if p.exists():
            p.unlink()


# ──────────────────────────────────────────────
# 6. FULL PIPELINE GENERATOR (yields progress dicts)
# ──────────────────────────────────────────────

def run_full_pipeline(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
):
    """
    Synchronous generator that yields progress dicts:
      {"step": str, "progress": int, "message": str}
    """
    yield {"step": "extracting_schema", "progress": 5, "message": "Connecting to database..."}

    try:
        schema = extract_schema(host, port, database, user, password)
    except Exception as e:
        yield {"step": "error", "progress": 0, "message": f"Schema extraction failed: {e}"}
        return

    table_count = len(schema)
    yield {"step": "extracting_schema", "progress": 20, "message": f"Extracted {table_count} tables."}

    # Context generation — report per-table progress
    yield {"step": "generating_context", "progress": 22, "message": "Enriching tables with Gemini..."}

    def on_table(name, done, total):
        pass  # progress yielded below

    try:
        tables = list(schema.keys())
        total = len(tables)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        content: dict = {}

        for idx, key in enumerate(tables):
            value = schema[key]
            
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=f"""Act as a database documentation assistant. I will provide a table name (`key`) and its JSON metadata (`value`) which includes columns, primary_keys, foreign_keys, and sample_rows.

        Your tasks:
        1. Generate a concise, one-sentence description of the table's purpose. Add this as a new key called "table_description" at the root level of the JSON object.
        2. Generate a brief description for each column. Add this as a new key called "description" inside each respective column's object within the "columns" section.
        3. Retain all original data (sample rows, keys, etc.).

        Output Requirements:
        Return ONLY the updated JSON object (the value). Do not wrap it in the original table name key, and do not include any markdown formatting, conversational text, or explanations.

        Table Name: {key}
        Original JSON Value: {value}""",
                        config=types.GenerateContentConfig(
                            # thinking_config=types.ThinkingConfig(thinking_level="low"),
                            response_mime_type="application/json"
                        ),
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        yield {"step": "generating_context", "progress": 22 + int(idx / total * 55), "message": f"API overload. Retrying table {key} in {(attempt+1)*5}s..."}
                        time.sleep((attempt + 1) * 5)
                    else:
                        raise e

            try:
                enriched = json.loads(response.text) if response else value
                content[key] = enriched
            except (json.JSONDecodeError, AttributeError):
                content[key] = value

            pct = 22 + int((idx + 1) / total * 55)  # 22% → 77%
            yield {"step": "generating_context", "progress": pct, "message": f"Enriched table: {key} ({idx+1}/{total})"}
            
            # Avoid exceeding rate limits with rapid sequential calls
            time.sleep(2)

        # Save context files
        db_ctx = _VIEW_DIR / f"{database}_context.json"
        with open(db_ctx, "w") as f:
            json.dump(content, f, indent=4)
        canonical = _VIEW_DIR / "context.json"
        with open(canonical, "w") as f:
            json.dump(content, f, indent=4)

    except Exception as e:
        yield {"step": "error", "progress": 0, "message": f"Context generation failed: {e}"}
        return

    yield {"step": "upserting_embeddings", "progress": 80, "message": "Encoding & upserting to Qdrant..."}

    try:
        upsert_to_qdrant(content, database, qdrant_host, qdrant_port)
    except Exception as e:
        yield {"step": "error", "progress": 0, "message": f"Qdrant upsert failed: {e}"}
        return

    yield {"step": "creating_indexes", "progress": 95, "message": "Creating payload indexes..."}
    yield {"step": "done", "progress": 100, "message": f"Pipeline complete. {table_count} tables indexed for '{database}'."}
