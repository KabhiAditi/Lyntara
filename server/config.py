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

    # Qdrant (schema search)
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model_name: str

    # LLM model names (used by Engine/*)
    gemini_model_name: str

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
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "table"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        pg_host=os.getenv("PGHOST", "localhost"),
        pg_port=int(os.getenv("PGPORT", "5432")),
        pg_database=os.getenv("PGDATABASE", ""),
        pg_user=os.getenv("PGUSER", ""),
        pg_password=os.getenv("PGPASSWORD", ""),
        pg_sslmode=os.getenv("PGSSLMODE"),
    )

