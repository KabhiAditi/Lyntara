import os

# HF_ENDPOINT used to be hardcoded to a mirror here. Importing this module then
# repointed the Hugging Face hub for the whole process, and a revalidation
# request to an unreachable mirror would hang model loading indefinitely.
# It is now opt-in: set HF_ENDPOINT in the environment if you need a mirror.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import re
import json
import sqlglot
from typing import Dict, Any, Optional, Set

import json
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer
import httpx

# Engine/ is imported both as a package (from the server) and as a bare script
# directory (Engine/main.py does `from sql import NL2SQL`). Put the repo root on
# the path so `Engine.llm` resolves either way.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parents[1])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from Engine.llm import STYLE_CHATML, LLMError, LLMProvider, get_provider, get_retries
from Engine.llm import prompts as sql_prompts


class NL2SQL:
    """Natural-language -> SQL, backed by a pluggable LLM provider.

    The provider is resolved from environment config (see Engine.llm.factory),
    so switching between the local fine-tune, a different local model, or a
    hosted API is a .env change rather than a code change.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional["LLMProvider"] = None,
        role: str = "sql",
    ):
        self.role = role
        self._provider = provider
        self.model_name = model_name

    @property
    def provider(self) -> "LLMProvider":
        if self._provider is None:
            self._provider = get_provider(self.role)
        return self._provider

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

    def _alias_map(self, parsed: "sqlglot.exp.Expression") -> Dict[str, str]:
        """Map table aliases to their real table names.

        The Spider fine-tune emits `FROM artist AS T1 JOIN album AS T2`, so
        without this every qualified column looks like it belongs to an unknown
        table `T1` and validation rejects perfectly good SQL.
        """
        aliases: Dict[str, str] = {}
        for table in parsed.find_all(sqlglot.exp.Table):
            if not table.name:
                continue
            real = table.name.lower()
            alias = table.alias_or_name
            if alias:
                aliases[alias.lower()] = real
            aliases.setdefault(real, real)
        return aliases

    def _validate(self, sql: str, schema: Dict[str, Any]) -> bool:
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
        except Exception:
            return False

        if parsed is None:
            return False

        allowed = self._allowed_schema(schema)
        aliases = self._alias_map(parsed)

        # Columns introduced by the query itself (SELECT ... AS x, CTE names)
        # are legitimate references that won't appear in the schema.
        derived = {
            a.alias.lower()
            for a in parsed.find_all(sqlglot.exp.Alias)
            if a.alias
        }
        cte_names = {
            cte.alias_or_name.lower()
            for cte in parsed.find_all(sqlglot.exp.CTE)
            if cte.alias_or_name
        }

        for table in parsed.find_all(sqlglot.exp.Table):
            name = table.name.lower() if table.name else ""
            if name and name not in allowed and name not in cte_names:
                return False

        for col in parsed.find_all(sqlglot.exp.Column):
            col_name = col.name.lower()
            qualifier = col.table.lower() if col.table else None

            if qualifier:
                real_table = aliases.get(qualifier, qualifier)
                if real_table in cte_names or qualifier in cte_names:
                    continue
                if real_table not in allowed:
                    return False
                if col_name not in allowed[real_table]:
                    return False
            else:
                if col_name in derived or col_name in cte_names:
                    continue
                if not any(col_name in cols for cols in allowed.values()):
                    return False

        return True

    def _call_llm(self, prompt: str, **params: Any) -> str:
        """Send a prompt through the configured provider.

        Providers declaring the ChatML style receive the prompt verbatim (no
        server-side templating), which is what the Spider fine-tune requires.
        """
        if self.model_name:
            params.setdefault("model", self.model_name)
        return self.provider.complete(prompt, **params)

    # Backwards-compatible alias for callers that predate the provider layer.
    _call_gemini = _call_llm

    def _uses_finetune_format(self) -> bool:
        return getattr(self.provider, "prompt_style", None) == STYLE_CHATML

    def generate(
        self,
        query: str,
        schema: Dict[str, Any],
        retries: Optional[int] = None,
        db_id: str = "database",
    ) -> str:
        """Generate SQL for ``query`` against ``schema``.

        Prompt format follows the provider: the Spider-tuned local model gets
        the ChatML rendering it was trained on, hosted instruction models get
        the original flat prompt. Output is validated against the schema before
        being accepted.
        """
        finetuned = self._uses_finetune_format()

        if finetuned:
            prompt = sql_prompts.build_chatml_prompt(query, schema, db_id=db_id)
            extract = sql_prompts.extract_sql
        else:
            prompt = self._build_prompt(query, schema)
            extract = self._extract_sql

        # Local 9B models take ~20-30s per call on Apple Silicon, so the default
        # attempt count comes from config rather than being hard-coded at 2.
        attempts = retries if retries is not None else get_retries(self.role)
        last_sql = "-- Failed to generate valid SQL"
        last_error: Optional[Exception] = None

        for attempt in range(max(1, attempts)):
            try:
                # A deterministic first pass; nudge sampling on retry so a
                # repeat attempt can actually produce something different.
                params: Dict[str, Any] = {"temperature": 0.4} if attempt else {}
                output_text = self._call_llm(prompt, **params)
            except LLMError as exc:
                last_error = exc
                print(f"[NL2SQL] provider error on attempt {attempt + 1}: {exc}")
                continue

            sql = extract(output_text)
            if not sql:
                continue
            last_sql = sql

            if sql.strip().lower().startswith("select") and self._validate(sql, schema):
                return sql

        if last_error is not None and last_sql == "-- Failed to generate valid SQL":
            raise last_error

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
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
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
