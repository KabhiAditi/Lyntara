"""Prompt construction for the Spider-fine-tuned Qwen3.5 SQL model.

This module encodes the *training contract* of SLM/V3/qwen_spider_checkpoints.
The adapter was trained on Spider with a specific ChatML rendering; feeding it
anything else throws away the fine-tune. Everything here mirrors
`SLM/V3/QwenV3.ipynb` (cells 17 and 19) and must stay in sync with it.

Two deviations from training worth knowing about:
  * Spider is SQLite; Lyntara targets PostgreSQL. The dialect gap is real and is
    handled downstream by sqlglot validation and the self-repair path.
  * Lyntara's schema dicts carry PostgreSQL type names, which are folded into
    the four Spider type buckets by ``_SPIDER_TYPE`` below.
"""

from __future__ import annotations

import re
from typing import Any

# Exact system prompt from QwenV3.ipynb cell 19. Do not reword.
SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, reason step-by-step about the question, then generate "
    "the correct SQL query. First explain your reasoning, then provide the "
    "final SQL query inside a ```sql code block."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Training targets opened the assistant turn with an EMPTY think block. We
# prefill it so generation continues from the same distribution.
_ASSISTANT_PREFILL = f"{IM_START}assistant\n<think>\n\n</think>\n\n"


def _spider_type(pg_type: str) -> str:
    """Fold a PostgreSQL type name into Spider's four type buckets."""
    t = (pg_type or "").strip().lower()
    if not t:
        return "TEXT"
    if any(k in t for k in ("int", "serial", "numeric", "decimal", "real", "double", "float", "money")):
        return "INTEGER"
    if any(k in t for k in ("timestamp", "date", "time", "interval")):
        return "DATETIME"
    if "bool" in t:
        return "BOOLEAN"
    return "TEXT"


def schema_to_create_tables(schema: dict[str, Any]) -> str:
    """Render Lyntara's schema dict as CREATE TABLE statements.

    Mirrors ``build_create_table_statements`` from QwenV3.ipynb cell 17:
    two-space-indented columns, inline PRIMARY KEY, FOREIGN KEY lines appended
    after the columns, statements joined by a blank line.
    """
    statements: list[str] = []

    for table, info in (schema or {}).items():
        info = info or {}
        columns = info.get("columns") or []
        if not columns:
            continue

        primary_keys = {str(pk).lower() for pk in (info.get("primary_keys") or [])}

        defs: list[str] = []
        for col in columns:
            name = col.get("column_name")
            if not name:
                continue
            col_def = f"  {name} {_spider_type(col.get('data_type', ''))}"
            if str(name).lower() in primary_keys:
                col_def += " PRIMARY KEY"
            defs.append(col_def)

        if not defs:
            continue

        for fk in info.get("foreign_keys") or []:
            column = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column")
            if column and ref_table and ref_col:
                defs.append(f"  FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_col})")

        statements.append(f"CREATE TABLE {table} (\n" + ",\n".join(defs) + "\n);")

    return "\n\n".join(statements)


def build_user_message(question: str, schema: dict[str, Any], db_id: str = "database") -> str:
    """The user turn, byte-for-byte as in training."""
    return f"Database: {db_id}\n\nSchema:\n{schema_to_create_tables(schema)}\n\nQuestion: {question}"


def build_chatml_prompt(
    question: str,
    schema: dict[str, Any],
    db_id: str = "database",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Full ChatML string, ending with the assistant prefill.

    Sent verbatim to the provider with raw/no-template mode.
    """
    return (
        f"{IM_START}system\n{system_prompt}{IM_END}\n"
        f"{IM_START}user\n{build_user_message(question, schema, db_id)}{IM_END}\n"
        f"{_ASSISTANT_PREFILL}"
    )


def build_messages(
    question: str,
    schema: dict[str, Any],
    db_id: str = "database",
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """OpenAI-style messages, for providers that template server-side."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(question, schema, db_id)},
    ]


_SQL_FENCE = re.compile(r"```sql\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a reasoning-then-```sql response.

    Uses the notebook's fence regex first, then falls back to statement
    matching for responses that skip the fence.
    """
    if not text:
        return ""

    text = _THINK_BLOCK.sub("", text)
    for token in (IM_END, IM_START, "<|endoftext|>"):
        text = text.replace(token, "")

    match = _SQL_FENCE.search(text)
    if match:
        sql = match.group(1).strip()
    else:
        cleaned = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
        stmt = re.search(
            r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if not stmt:
            return cleaned.strip()
        sql = stmt.group(0).strip()

    # Keep only the first statement; the model occasionally rambles after it.
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    sql = sql.strip().rstrip(";").strip()
    return f"{sql};" if sql else ""
