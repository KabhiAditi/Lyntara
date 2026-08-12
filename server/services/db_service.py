from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from server.services.rbac_service import validate_sql_security


@dataclass(frozen=True)
class PgConnectResult:
    ok: bool
    message: str


def _connect_postgres(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    sslmode: str | None = None,
    connect_timeout_s: int = 5,
):
    sslmode_part = f" sslmode={sslmode}" if sslmode else ""
    conninfo = (
        f"host={host} port={port} dbname={database} user={user} password={password}"
        f" connect_timeout={connect_timeout_s}{sslmode_part}"
    )
    try:
        import psycopg
        return psycopg.connect(conninfo)
    except ImportError:
        import psycopg2
        return psycopg2.connect(
            dbname=database,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=connect_timeout_s,
            sslmode=sslmode,
        )


def check_postgres_connectivity(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    sslmode: str | None = None,
    connect_timeout_s: int = 3,
) -> PgConnectResult:
    """
    Connectivity check only: open TCP connection + auth, then close.
    No queries are executed.
    """
    try:
        conn = _connect_postgres(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            sslmode=sslmode,
            connect_timeout_s=connect_timeout_s,
        )
        conn.close()
        return PgConnectResult(ok=True, message="PostgreSQL connection succeeded.")
    except Exception as e:
        return PgConnectResult(ok=False, message=f"PostgreSQL connection failed ({type(e).__name__}).")


def execute_sql(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    sql: str,
    sslmode: str | None = None,
    connect_timeout_s: int = 5,
) -> tuple[list[str], list[list[Any]], str | None, bool, dict[str, Any] | None]:
    """
    Execute SQL against PostgreSQL with RBAC security validation.
    The RBAC check is executed BEFORE calling cur.execute(sql).
    Returns (columns, rows, error_message, permission_denied_flag, denied_details_dict).
    """
    try:
        conn = _connect_postgres(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            sslmode=sslmode,
            connect_timeout_s=connect_timeout_s,
        )
    except Exception as e:
        return [], [], f"PostgreSQL connection failed: {str(e)}", False, None

    try:
        with conn:
            # 1. RBAC validation BEFORE SQL execution
            rbac_result = validate_sql_security(conn, user, database, sql)

            # 2. Block execution if authorization check fails
            if not rbac_result.allowed:
                denied_details = {
                    "username": user,
                    "database": database,
                    "operation": rbac_result.operation,
                    "missing_privileges": rbac_result.missing_privileges,
                    "message": rbac_result.message,
                }
                return [], [], rbac_result.message, True, denied_details

            # 3. Execute SQL query only if RBAC check passes
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return columns, [list(r) for r in rows], None, False, None
                conn.commit()
                return [], [], None, False, None
    except Exception as e:
        return [], [], f"Execution failed: {str(e)}", False, None
    finally:
        try:
            conn.close()
        except Exception:
            pass
