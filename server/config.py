from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # Paths
    repo_root: Path
    context_json_path: Path

    # API
    cors_allow_origins: list[str]

    # Schema selection strategy for SQL generation:
    #   "embeddings" -> Qdrant top-k retrieval   "full" -> entire schema in prompt
    schema_mode: str
    schema_top_k: int

    # Qdrant (schema search)
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model_name: str

    # LLM model names (used by Engine/*)
    # Context/JSON enrichment still runs on Gemini; SQL generation is local.
    gemini_model_name: str

    # Local LLM (SQL generation) -- see Engine/llm/factory.py for the full
    # LLM_<ROLE>_* environment contract.
    llm_sql_provider: str
    llm_sql_base_url: str
    llm_sql_model: str
    llm_sql_timeout: float
    llm_sql_retries: int

    # PostgreSQL connectivity check only
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str
    pg_sslmode: str | None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    context_json_path = repo_root / "View_Selection" / "context.json"

    # Default to permissive CORS so opening the HTML via file:// (origin "null")
    # can still call the API during local development.
    cors_allow_origins = _split_csv(os.getenv("CORS_ALLOW_ORIGINS", "*,null"))

    return Settings(
        repo_root=repo_root,
        context_json_path=context_json_path,
        cors_allow_origins=cors_allow_origins,
        # This branch runs WITHOUT embeddings: the whole schema is placed in the
        # prompt and Qdrant is never consulted for SQL generation.
        schema_mode=os.getenv("SCHEMA_MODE", "full").strip().lower(),
        schema_top_k=int(os.getenv("SCHEMA_TOP_K", "5")),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "table"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        llm_sql_provider=os.getenv("LLM_SQL_PROVIDER", "ollama"),
        llm_sql_base_url=os.getenv("LLM_SQL_BASE_URL", "http://localhost:11434"),
        llm_sql_model=os.getenv("LLM_SQL_MODEL", "lyntara-sql:ckpt200"),
        llm_sql_timeout=float(os.getenv("LLM_SQL_TIMEOUT", "120")),
        llm_sql_retries=int(os.getenv("LLM_SQL_RETRIES", "1")),
        pg_host=os.getenv("PGHOST", "localhost"),
        pg_port=int(os.getenv("PGPORT", "5432")),
        pg_database=os.getenv("PGDATABASE", ""),
        pg_user=os.getenv("PGUSER", ""),
        pg_password=os.getenv("PGPASSWORD", ""),
        pg_sslmode=os.getenv("PGSSLMODE"),
    )

