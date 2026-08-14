from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from server.config import get_settings
from server.schemas import (
    ConnectRequest,
    ConnectResponse,
    ErrorEnvelope,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
    ExecuteRequest,
    ExecuteResponse,
    PipelineRequest,
    PipelineStatusResponse,
    DeleteCollectionRequest,
    DeleteCollectionResponse,
)
from server.services.engine_service import run_readonly_pipeline
from server.services.db_service import check_postgres_connectivity, execute_sql
from server.services.pipeline_service import (
    check_pipeline_done,
    delete_collections,
    run_full_pipeline,
)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="NL2SQL API", version="v1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ok=True)

    @app.post(
        "/api/connect",
        response_model=ConnectResponse,
        responses={400: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}},
    )
    def connect(req: ConnectRequest) -> ConnectResponse:
        settings = get_settings()
        warnings: list[str] = []

        qdrant_host = req.qdrant_host or settings.qdrant_host
        qdrant_port = req.qdrant_port or settings.qdrant_port
        
        qdrant_ok = False
        try:
            import httpx
            url = f"http://{qdrant_host}:{qdrant_port}/collections"
            res = httpx.get(url, timeout=2.0)
            if res.status_code == 200:
                qdrant_msg = "Qdrant connection succeeded."
                qdrant_ok = True
            else:
                qdrant_msg = f"Qdrant returned status {res.status_code}"
        except Exception as e:
            qdrant_msg = f"Qdrant check failed: {type(e).__name__}"

        pg_ok = False
        pg_message = "PostgreSQL connection check skipped."
        if req.engine != "postgresql":
            warnings.append(
                f"Engine '{req.engine}' not supported yet; PostgreSQL-only connectivity checks are implemented."
            )
        else:
            if not req.database:
                warnings.append("Database name is empty.")
            pg_res = check_postgres_connectivity(
                host=req.host,
                port=req.port or 5432,
                database=req.database,
                user=req.username,
                password=req.password,
                sslmode=req.sslmode,
            )
            pg_ok = pg_res.ok
            pg_message = pg_res.message

        ok = bool(qdrant_ok and (pg_ok if req.engine == "postgresql" else True))
        parts = []
        parts.append(qdrant_msg)
        parts.append(pg_message)

        return ConnectResponse(
            ok=ok,
            pg_ok=pg_ok,
            qdrant_ok=qdrant_ok,
            message=" ".join(parts),
            warnings=warnings,
        )

    @app.post(
        "/api/query",
        response_model=QueryResponse,
        responses={400: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}},
    )
    def query(req: QueryRequest) -> QueryResponse:
        try:
            payload = run_readonly_pipeline(
                query=req.query, top_k=req.top_k, session_id=req.session_id,
                database_name=req.database,
            )
            return QueryResponse(**payload)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": str(e)})
        except Exception as e:
            raise HTTPException(
                status_code=500, detail={"error": "internal_error", "detail": str(e)}
            )

    @app.get("/api/schema", response_model=SchemaResponse)
    def schema(database: str = Query(default="")) -> SchemaResponse:
        from pathlib import Path
        view_dir = Path("View_Selection")
        ctx = {}
        # Try database-specific context first, fallback to generic
        candidates = []
        if database:
            candidates.append(view_dir / f"{database}_context.json")
        candidates.append(view_dir / "context.json")
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        ctx = json.load(f)
                    break
                except Exception:
                    continue
        return SchemaResponse(ok=True, table_count=len(ctx or {}), schema_data=ctx or {})

    @app.post("/api/execute", response_model=ExecuteResponse)
    def api_execute(req: ExecuteRequest) -> ExecuteResponse:
        cols, rows, err, permission_denied, denied_details = execute_sql(
            host=req.host,
            port=req.port or 5432,
            database=req.database,
            user=req.username,
            password=req.password,
            sql=req.sql,
        )
        if permission_denied:
            return ExecuteResponse(
                ok=False,
                error=err or "Permission denied",
                columns=[],
                rows=[],
                permission_denied=True,
                denied_details=denied_details,
            )
        if err:
            fixed_sql = None
            if req.nl_query and req.database:
                from server.services.engine_service import fix_sql_with_gemini
                fixed_sql = fix_sql_with_gemini(req.database, req.nl_query, req.sql, err)
            return ExecuteResponse(ok=False, error=err, columns=[], rows=[], fixed_sql=fixed_sql)
        return ExecuteResponse(ok=True, columns=cols, rows=rows)

    # ── Pipeline Endpoints ──────────────────────────────

    @app.post("/api/pipeline/status", response_model=PipelineStatusResponse)
    def pipeline_status(req: PipelineRequest) -> PipelineStatusResponse:
        settings = get_settings()
        qdrant_host = req.qdrant_host or settings.qdrant_host
        qdrant_port = req.qdrant_port or settings.qdrant_port
        done = check_pipeline_done(req.database, qdrant_host, qdrant_port)
        if done:
            return PipelineStatusResponse(
                ok=True, already_indexed=True,
                message=f"Database '{req.database}' is already indexed."
            )
        return PipelineStatusResponse(
            ok=True, already_indexed=False,
            message=f"Database '{req.database}' has not been indexed yet."
        )

    @app.post("/api/pipeline/run")
    def pipeline_run(req: PipelineRequest):
        """SSE stream that yields realtime pipeline progress."""
        settings = get_settings()
        qdrant_host = req.qdrant_host or settings.qdrant_host
        qdrant_port = req.qdrant_port or settings.qdrant_port

        def event_stream():
            for event in run_full_pipeline(
                host=req.host,
                port=req.port or 5432,
                database=req.database,
                user=req.username,
                password=req.password,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/pipeline/delete", response_model=DeleteCollectionResponse)
    def pipeline_delete(req: DeleteCollectionRequest) -> DeleteCollectionResponse:
        settings = get_settings()
        qdrant_host = req.qdrant_host or settings.qdrant_host
        qdrant_port = req.qdrant_port or settings.qdrant_port
        try:
            delete_collections(req.database, qdrant_host, qdrant_port)
            return DeleteCollectionResponse(
                ok=True,
                message=f"Collections and cached files for '{req.database}' deleted."
            )
        except Exception as e:
            return DeleteCollectionResponse(
                ok=False,
                message=f"Delete failed: {e}"
            )

    return app


app = create_app()

