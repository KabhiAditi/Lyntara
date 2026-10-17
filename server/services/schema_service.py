"""Schema selection for SQL generation.

Two strategies decide what the model is shown:

  embeddings -- dense retrieval over table descriptions in Qdrant, returning
                the top-k most relevant tables. Scales to wide schemas, but
                a join is only answerable if every table it needs is retrieved.

  full       -- the entire schema is placed in the prompt. No vector database
                is involved, so nothing can be missed, but prompt size grows
                with the schema and large databases will not fit the context
                window.

The mode is config-driven (SCHEMA_MODE) so the two can be compared on
identical inputs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MODE_EMBEDDINGS = "embeddings"
MODE_FULL = "full"


class SchemaUnavailable(RuntimeError):
    """The schema could not be resolved for this database."""


def _context_path(repo_root: Path, database: str | None, fallback: Path) -> Path:
    if database:
        return repo_root / "View_Selection" / f"{database}_context.json"
    return fallback


@lru_cache(maxsize=16)
def _load_context(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise SchemaUnavailable(
            f"No schema context at {path.name}. Run the indexing pipeline for this database first."
        )
    return json.loads(path.read_text())


def load_full_schema(repo_root: Path, database: str | None, fallback: Path) -> dict[str, Any]:
    """Every table in the database."""
    return dict(_load_context(str(_context_path(repo_root, database, fallback))))


def select_schema(
    *,
    repo_root: Path,
    database: str | None,
    query: str,
    mode: str,
    top_k: int,
    collection: str,
    fallback_context: Path,
    embedding_model: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return (schema, warnings) for ``query`` under the configured ``mode``."""
    warnings: list[str] = []
    full = load_full_schema(repo_root, database, fallback_context)

    if mode == MODE_FULL:
        return full, warnings

    # Dense retrieval. Falls back to the full schema rather than failing the
    # request, so a Qdrant outage degrades quality instead of breaking the app.
    try:
        from Engine.sql import SchemaRetriever

        retriever = _get_retriever(collection, str(_context_path(repo_root, database, fallback_context)), embedding_model)
        hits = retriever.search_schema(query, top_k=top_k, collection_name=collection)
        tables = [
            (h.get("payload") or {}).get("table")
            for h in (hits.get("result") or [])
        ]
        picked = {t: full[t] for t in tables if t and t in full}
        if not picked:
            warnings.append("Schema retrieval returned no known tables; using the full schema.")
            return full, warnings
        return picked, warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Schema retrieval failed ({type(exc).__name__}); using the full schema.")
        return full, warnings


@lru_cache(maxsize=8)
def _get_retriever(collection: str, context_file: str, model_name: str):
    from Engine.sql import SchemaRetriever

    return SchemaRetriever(collection_name=collection, context_file=context_file, model_name=model_name)
