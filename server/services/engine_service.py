from __future__ import annotations

import os
import re
import uuid
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from server.config import get_settings

# Ensure Engine module can be imported
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Engine.classifier import predict_query_with_confidence, combine_mrd_query, add_to_history
from Engine.sql import SchemaRetriever, NL2SQL

@lru_cache(maxsize=1)
def _get_nl2sql(model_name: str):
    return NL2SQL(model_name=model_name)

@lru_cache(maxsize=8)
def _get_retriever(collection_name: str = "table", context_file: str = "View_Selection/context.json", model_name: str = "all-MiniLM-L6-v2"):
    return SchemaRetriever(collection_name=collection_name, context_file=context_file, model_name=model_name)

def _extract_matched_tables(schema: dict[str, Any] | None) -> list[str]:
    if not schema:
        return []
    return [t for t in schema.keys() if t]

def _sql_is_select_only(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")

def _sql_is_single_statement(sql: str) -> bool:
    s = sql.strip()
    if not s.endswith(";"):
        return False
    return s[:-1].find(";") == -1

def _sql_contains_write_ops(sql: str) -> bool:
    return bool(
        re.search(
            r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b",
            sql,
            flags=re.IGNORECASE,
        )
    )

def _looks_like_null(sql: str) -> bool:
    return bool(re.fullmatch(r"null;?", sql.strip(), flags=re.IGNORECASE))


def run_readonly_pipeline(query: str, top_k: int = 5, session_id: str | None = None, database_name: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    session_id = session_id or str(uuid.uuid4())
    warnings: list[str] = []

    # Resolve collection name and context file based on database
    if database_name:
        collection = f"{database_name}-table"
        context_file = f"View_Selection/{database_name}_context.json"
    else:
        collection = settings.qdrant_collection
        context_file = str(settings.context_json_path)

    # 1. Classification & Resolution (via Engine)
    confidence = None
    query_type = "UNKNOWN"
    resolved_query = query
    try:
        if os.getenv("ENABLE_CLASSIFIER", "0") != "1":
            warnings.append("Classifier disabled (set ENABLE_CLASSIFIER=1 to enable).")
        else:
            pred, confidence = predict_query_with_confidence(query)
            is_mrd = str(pred).upper() in ("MRD", "1")
            query_type = "MRD" if is_mrd else "SRD"
            
            if is_mrd:
                resolved_query = combine_mrd_query(query, confidence)
                # If combiner returned query unchanged (low confidence / no history),
                # downgrade to SRD so frontend displays correctly.
                if resolved_query == query:
                    query_type = "SRD"
            
            add_to_history(query, resolved_query)
    except Exception as e:
        warnings.append(f"Classifier/Combiner error: {type(e).__name__}")

    # 2. Schema Retrieval (via Engine)
    schema: dict[str, Any] | None = None
    try:
        retriever = _get_retriever(collection, context_file, settings.embedding_model_name)
        schema = retriever.retrieve(resolved_query, top_k=top_k)
        if not schema:
            warnings.append("No matching schema context found for query.")
    except Exception as e:
        warnings.append(f"Schema retrieval failed ({type(e).__name__}).")

    matched_tables = _extract_matched_tables(schema)

    # 3. SQL Generation (via Engine)
    sql: str | None = None
    sql_valid = False
    try:
        if schema:
            if os.getenv("ENABLE_SQL_GENERATION", "0") != "1":
                warnings.append("SQL generation is disabled (set ENABLE_SQL_GENERATION=1).")
            else:
                nl2sql = _get_nl2sql(settings.gemini_model_name)
                sql = nl2sql.generate(query=resolved_query, schema=schema, retries=2)

            if sql and _looks_like_null(sql):
                sql = None
                sql_valid = False
            elif sql:
                if _sql_contains_write_ops(sql):
                    warnings.append("Generated SQL contained write operations; suppressed for safety.")
                    sql = None
                    sql_valid = False
                elif not _sql_is_select_only(sql):
                    warnings.append("Generated SQL was not SELECT/WITH; suppressed for safety.")
                    sql = None
                    sql_valid = False
                elif not _sql_is_single_statement(sql):
                    warnings.append("Generated SQL was not a single statement; suppressed for safety.")
                    sql = None
                    sql_valid = False
                else:
                    sql_valid = True
    except Exception as e:
        warnings.append(f"SQL generation failed ({type(e).__name__}).")

    return {
        "ok": True,
        "query": query,
        "session_id": session_id,
        "query_type": query_type,
        "confidence": float(confidence) if confidence is not None else None,
        "resolved_query": resolved_query,
        "schema": schema,
        "matched_tables": matched_tables,
        "sql": sql,
        "sql_valid": bool(sql_valid and sql is not None),
        "execution_status": "disabled",
        "can_execute": False,
        "warnings": warnings,
    }


def fix_sql_with_gemini(database_name: str, nl_query: str, failed_sql: str, error_msg: str) -> str | None:
    settings = get_settings()
    collection = f"{database_name}-table"
    context_file = f"View_Selection/{database_name}_context.json"
    
    try:
        retriever = _get_retriever(collection, context_file, settings.embedding_model_name)
        schema = retriever.retrieve(nl_query, top_k=5)
    except Exception:
        return None
        
    if not schema:
        return None
        
    try:
        nl2sql = _get_nl2sql(settings.gemini_model_name)
        schema_text = nl2sql._schema_to_text(schema)
        
        prompt = f"""You are a Text-to-SQL system.

Convert the natural language query into a valid PostgreSQL SQL query.
Constraints:
- Use ONLY tables and columns from the schema
- Return ONLY SQL
- End the SQL with a semicolon. Do not wrap in markdown fences.

Schema:
{schema_text}

Query:
{nl_query}

=== PREVIOUS ATTEMPT FAILED ===
You previously generated this SQL:
{failed_sql}

It resulted in this error from the PostgreSQL database:
{error_msg}

Please correct the SQL query to fix this error. 
Return ONLY the corrected valid PostgreSQL SQL query. Do not include explanations. Ensure the syntax is correct. End with a semicolon.
SQL:
"""
        output_text = nl2sql._call_gemini(prompt)
        fixed_sql = nl2sql._extract_sql(output_text)
        return fixed_sql
    except Exception as e:
        print(f"[Engine] Failed to fix SQL via Gemini: {e}")
        return None

