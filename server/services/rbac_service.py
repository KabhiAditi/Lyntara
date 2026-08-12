from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence

import sqlglot
import sqlglot.expressions as exp

logger = logging.getLogger("rbac_security")


@dataclass(frozen=True)
class TableRef:
    schema: str
    table: str

    def formatted(self) -> str:
        return f'"{self.schema}"."{self.table}"'


@dataclass(frozen=True)
class RequiredTablePrivilege:
    schema: str
    table: str
    privilege: str  # 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'CREATE', 'ALTER', 'DROP'


@dataclass
class SQLPermissionRequirement:
    operation: str
    referenced_schemas: set[str] = field(default_factory=set)
    referenced_tables: list[TableRef] = field(default_factory=list)
    table_privileges: list[RequiredTablePrivilege] = field(default_factory=list)


@dataclass(frozen=True)
class PermissionCheckResult:
    allowed: bool
    operation: str
    missing_privileges: list[dict[str, Any]]
    message: str


def _extract_table_from_node(node: Any) -> tuple[str, str] | None:
    if node is None:
        return None
    if isinstance(node, exp.Table):
        s = node.db or "public"
        t = node.name
        if t:
            return (s, t)
    elif isinstance(node, exp.Schema):
        tbl = node.find(exp.Table)
        if tbl and tbl.name:
            return (tbl.db or "public", tbl.name)
        if node.this:
            db_arg = node.args.get("db")
            db_str = db_arg.name if db_arg else "public"
            return (db_str, str(node.this))
    return None


def parse_sql_permissions(sql: str) -> SQLPermissionRequirement:
    """
    Parse SQL using sqlglot and extract operation type, referenced schemas,
    referenced tables, and required privileges per table.
    Fails by raising ValueError if parsing is not possible or operation is unsupported.
    """
    cleaned_sql = sql.strip()
    if not cleaned_sql:
        raise ValueError("Empty SQL query")

    try:
        parsed = sqlglot.parse_one(cleaned_sql, read="postgres")
    except Exception as e:
        raise ValueError(f"SQL parsing failed: {e}") from e

    if not parsed:
        raise ValueError("Invalid AST produced from SQL")

    root_key = parsed.key.upper()
    if "TRUNCATE" in root_key:
        op = "TRUNCATE"
    elif root_key in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"):
        op = root_key
    else:
        raise ValueError(f"Unsupported SQL operation: '{root_key}'")

    ctes: set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        if cte.alias:
            ctes.add(cte.alias.lower())

    target_pair: tuple[str, str] | None = None
    if isinstance(parsed, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Alter, exp.Drop)):
        target_pair = _extract_table_from_node(parsed.this)

    schemas: set[str] = set()
    tables: list[TableRef] = []
    req_privs: list[RequiredTablePrivilege] = []
    seen: set[tuple[str, str, str]] = set()

    for table_node in parsed.find_all(exp.Table):
        t_name = table_node.name
        if not t_name or t_name.lower() in ctes:
            continue
        s_name = table_node.db or "public"
        pair = (s_name, t_name)

        if target_pair and pair == target_pair:
            priv = op
        else:
            if op == "TRUNCATE":
                priv = "TRUNCATE"
            else:
                priv = "SELECT"

        key = (s_name, t_name, priv)
        if key not in seen:
            seen.add(key)
            schemas.add(s_name)
            t_ref = TableRef(schema=s_name, table=t_name)
            tables.append(t_ref)
            req_privs.append(RequiredTablePrivilege(schema=s_name, table=t_name, privilege=priv))

    if not req_privs and target_pair:
        s_name, t_name = target_pair
        schemas.add(s_name)
        tables.append(TableRef(schema=s_name, table=t_name))
        req_privs.append(RequiredTablePrivilege(schema=s_name, table=t_name, privilege=op))

    return SQLPermissionRequirement(
        operation=op,
        referenced_schemas=schemas,
        referenced_tables=tables,
        table_privileges=req_privs,
    )


def validate_sql_security(
    conn: Any, username: str, dbname: str, sql: str
) -> PermissionCheckResult:
    """
    Validate SQL security using PostgreSQL built-in privilege functions BEFORE execution.
    Returns PermissionCheckResult detailing if execution is allowed.
    Fails closed if parsing or check fails.
    """
    # 1. Parse SQL
    try:
        reqs = parse_sql_permissions(sql)
    except Exception as parse_err:
        msg = f"Security check failed: SQL parsing error ({parse_err})"
        missing = [{"error": "sql_parse_error", "detail": str(parse_err)}]
        _log_audit(username, dbname, "UNKNOWN", [], False, missing)
        return PermissionCheckResult(
            allowed=False,
            operation="UNKNOWN",
            missing_privileges=missing,
            message=msg,
        )

    op = reqs.operation
    missing_privileges: list[dict[str, Any]] = []
    referenced_tables_str = [t.formatted() for t in reqs.referenced_tables]

    try:
        cur = conn.cursor()

        # 2. Database CONNECT privilege check
        try:
            cur.execute("SELECT has_database_privilege(%s, %s, 'CONNECT');", (username, dbname))
            has_db_priv = cur.fetchone()[0]
        except Exception:
            has_db_priv = False

        if not has_db_priv:
            missing_privileges.append({
                "type": "database",
                "database": dbname,
                "privilege": "CONNECT",
                "user": username,
            })

        # 3. Schema USAGE / CREATE privilege checks
        for s_name in reqs.referenced_schemas:
            try:
                cur.execute("SELECT has_schema_privilege(%s, %s, 'USAGE');", (username, s_name))
                has_usage = cur.fetchone()[0]
            except Exception:
                has_usage = False

            if not has_usage:
                missing_privileges.append({
                    "type": "schema",
                    "schema": s_name,
                    "privilege": "USAGE",
                    "user": username,
                })

            if op == "CREATE":
                try:
                    cur.execute("SELECT has_schema_privilege(%s, %s, 'CREATE');", (username, s_name))
                    has_create = cur.fetchone()[0]
                except Exception:
                    has_create = False

                if not has_create:
                    missing_privileges.append({
                        "type": "schema",
                        "schema": s_name,
                        "privilege": "CREATE",
                        "user": username,
                    })

        # 4. Table privilege checks
        standard_privs = {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE"}
        for req in reqs.table_privileges:
            tbl_qualified = f'"{req.schema}"."{req.table}"'
            priv = req.privilege

            if priv in standard_privs:
                try:
                    cur.execute("SELECT has_table_privilege(%s, %s, %s);", (username, tbl_qualified, priv))
                    has_tbl_priv = cur.fetchone()[0]
                except Exception:
                    has_tbl_priv = False

                if not has_tbl_priv:
                    missing_privileges.append({
                        "type": "table",
                        "schema": req.schema,
                        "table": req.table,
                        "privilege": priv,
                        "user": username,
                    })
            elif priv == "CREATE":
                # Schema CREATE privilege already checked above
                pass
            elif priv in ("ALTER", "DROP"):
                # Check if table exists in PG catalog
                has_alter_drop = False
                try:
                    cur.execute("SELECT to_regclass(%s);", (tbl_qualified,))
                    reg = cur.fetchone()[0]
                    if reg is None:
                        # Table does not exist: check schema CREATE privilege
                        cur.execute("SELECT has_schema_privilege(%s, %s, 'CREATE');", (username, req.schema))
                        has_alter_drop = cur.fetchone()[0]
                    else:
                        # Table exists: check table ownership or schema CREATE
                        cur.execute(
                            """
                            SELECT pg_has_role(%s, (SELECT relowner FROM pg_class WHERE oid = %s::regclass), 'USAGE')
                                   OR has_schema_privilege(%s, %s, 'CREATE');
                            """,
                            (username, tbl_qualified, username, req.schema),
                        )
                        has_alter_drop = cur.fetchone()[0]
                except Exception:
                    has_alter_drop = False

                if not has_alter_drop:
                    missing_privileges.append({
                        "type": "table",
                        "schema": req.schema,
                        "table": req.table,
                        "privilege": priv,
                        "user": username,
                    })

        cur.close()

    except Exception as db_err:
        msg = f"Database privilege check failed: {db_err}"
        missing_privileges.append({"error": "db_check_error", "detail": str(db_err)})
        _log_audit(username, dbname, op, referenced_tables_str, False, missing_privileges)
        return PermissionCheckResult(
            allowed=False,
            operation=op,
            missing_privileges=missing_privileges,
            message=msg,
        )

    allowed = len(missing_privileges) == 0
    msg = (
        f"Access granted for operation {op} on database {dbname}."
        if allowed
        else f"Access denied: User '{username}' lacks required PostgreSQL privilege(s) for operation {op}."
    )

    _log_audit(username, dbname, op, referenced_tables_str, allowed, missing_privileges)

    return PermissionCheckResult(
        allowed=allowed,
        operation=op,
        missing_privileges=missing_privileges,
        message=msg,
    )


def _log_audit(
    username: str,
    database: str,
    operation: str,
    referenced_tables: list[str],
    allowed: bool,
    missing_privileges: list[dict[str, Any]],
) -> None:
    """
    Structured audit logging for authorization decisions.
    Never logs passwords.
    """
    decision = "ALLOW" if allowed else "DENY"
    if allowed:
        logger.info(
            f"RBAC {decision}: User '{username}' op '{operation}' on db '{database}' "
            f"tables={referenced_tables}"
        )
    else:
        logger.warning(
            f"RBAC {decision}: User '{username}' op '{operation}' on db '{database}' "
            f"tables={referenced_tables} missing={missing_privileges}"
        )
