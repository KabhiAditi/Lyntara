import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import re
import json
import sqlglot
from typing import Dict, Any, Set
from google import genai

import json
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer
import httpx


class NL2SQL:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _schema_to_text(self, schema: Dict[str, Any]) -> str:
        parts = []

        for table, info in schema.items():
            parts.append(f"Table: {table}")

            cols = info.get("columns", [])
            if cols:
                col_str = ", ".join(
                    f"{c['column_name']} ({c.get('data_type', '')})"
                    for c in cols
                )
                parts.append(f"Columns: {col_str}")

            pks = info.get("primary_keys", [])
            if pks:
                parts.append("Primary Keys: " + ", ".join(pks))

            fks = info.get("foreign_keys", [])
            if fks:
                fk_str = ", ".join(
                    f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}"
                    for fk in fks
                )
                parts.append("Foreign Keys: " + fk_str)

            samples = info.get("sample_rows", [])
            if samples:
                parts.append("Sample Rows:")
                for row in samples[:2]:
                    parts.append(json.dumps(row, ensure_ascii=False))

            parts.append("")

        return "\n".join(parts)

    def _build_prompt(self, query: str, schema: Dict[str, Any]) -> str:
        schema_text = self._schema_to_text(schema)

        return f"""
You are a Text-to-SQL system.

Convert the natural language query into a valid PostgreSQL SQL query.

Constraints:
- Use ONLY tables and columns from the schema
- Do NOT invent table names or column names
- Use joins only if needed
- Return ONLY SQL
- Prefer exact matching using sample values when appropriate
- If a person's full name appears and the schema has separate first_name and last_name columns, split it correctly
- Do not include markdown fences
- If the provided schema does not satisfy the requested query return NULL
- - DATA_MODIFICATION_ENFORCEMENT: When the user intends to add, create, or insert new data, you must never output a bare `SELECT` statement that only calculates values. You must wrap the logic inside an executable modification statement (such as `INSERT INTO ... SELECT`). Ensure the query actively modifies the database state rather than just displaying a calculated row.
- End the SQL with a semicolon

Schema:
{schema_text}

Query:
{query}

SQL:
""".strip()

    def _extract_sql(self, text: str) -> str:
        text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text).strip()

        match = re.search(r"(SELECT\b.*?;)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"(WITH\b.*?;)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"((SELECT|WITH)\b.*)", text, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if not sql.endswith(";"):
                sql += ";"
            return sql

        return text

    def _allowed_schema(self, schema: Dict[str, Any]) -> Dict[str, Set[str]]:
        return {
            table.lower(): {
                col["column_name"].lower()
                for col in info.get("columns", [])
            }
            for table, info in schema.items()
        }

    def _validate(self, sql: str, schema: Dict[str, Any]) -> bool:
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
        except Exception:
            return False

        allowed = self._allowed_schema(schema)

        for table in parsed.find_all(sqlglot.exp.Table):
            if table.name and table.name.lower() not in allowed:
                return False

        for col in parsed.find_all(sqlglot.exp.Column):
            col_name = col.name.lower()
            table_name = col.table.lower() if col.table else None

            if table_name:
                if table_name not in allowed:
                    return False
                if col_name not in allowed[table_name]:
                    return False
            else:
                if not any(col_name in cols for cols in allowed.values()):
                    return False

        return True

    def _call_gemini(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return (response.text or "").strip()

    def generate(self, query: str, schema: Dict[str, Any], retries: int = 2) -> str:
        prompt = self._build_prompt(query, schema)
        last_sql = "-- Failed to generate valid SQL"

        for _ in range(retries):
            output_text = self._call_gemini(prompt)
            sql = self._extract_sql(output_text)
            last_sql = sql

            if sql.strip().lower().startswith("select") and self._validate(sql, schema):
                return sql

        return last_sql



class SchemaRetriever:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "table",
        model_name: str = "all-MiniLM-L6-v2",
        context_file: str = "View_Selection/context.json",
        hf_endpoint: Optional[str] = None,
        check_compatibility: bool = False,
    ):
        # if hf_endpoint:
        #     os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.collection_name = collection_name
        self.context_file = context_file
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.base_url = f"http://{qdrant_host}:{qdrant_port}"

        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def search_schema(self, query_text: str, top_k: int = 5, collection_name: Optional[str] = None):
        collection = collection_name or self.collection_name
        query_vector = self.model.encode(query_text).tolist()
        print("[SchemaRetriever] Similarity search performed")

        url = f"{self.base_url}/collections/{collection}/points/search"
        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        with httpx.Client(timeout=5.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    def get_top_context_object(self, search_results, context_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        context_path = context_file or self.context_file

        points = None
        if isinstance(search_results, dict):
            points = search_results.get("result")

        if not points:
            return None

        top_hit = points[0] or {}
        payload = top_hit.get("payload") or {}
        table_name = (payload or {}).get("table")
        if not table_name:
            return None

        with open(context_path, "r") as f:
            context_data = json.load(f)

        obj = {}
        obj[table_name] = context_data.get(table_name)
        print("[SchemaRetriever] Collected top context object:")
        # if isinstance(obj, dict):
        #     obj = dict(obj)  # avoid mutating file-backed object
        #     obj["table_name"] = table_name
        #     obj["score"] = float(getattr(top_hit, "score", 0.0))
        return obj

    def retrieve(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Convenience method: search + return top context + raw hits."""
        hits = self.search_schema(query_text=query_text, top_k=top_k)
        top_context = self.get_top_context_object(hits)
        # return {
        #     "query": query_text,
        #     "top_context": top_context,
        #     "hits": [
        #         {
        #             "table": (getattr(p, "payload", None) or {}).get("table"),
        #             "score": float(getattr(p, "score", 0.0)),
        #             "payload": getattr(p, "payload", None),
        #         }
        #         for p in (getattr(hits, "points", None) or [])
        #     ],
        # }
        return top_context
