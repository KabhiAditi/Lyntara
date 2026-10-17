#!/usr/bin/env python3
"""Counterfactual: how much accuracy is lost purely to SQLite->PostgreSQL dialect?

The Spider fine-tunes were trained on SQLite. Two SQLite habits are invalid in
PostgreSQL and account for most observed failures:

  1. Double-quoted string literals.  WHERE country = "Canada"
     SQLite tolerates this; PostgreSQL reads double quotes as an *identifier*,
     so it reports: column "Canada" does not exist.

  2. Lax GROUP BY.  SELECT count(*), name FROM t GROUP BY id
     SQLite allows selecting a column absent from GROUP BY; PostgreSQL rejects
     it unless the column is aggregated or grouped.

This script re-scores already-saved predictions with those two repairs applied.
It makes no model calls -- it is a pure replay -- so the difference isolates
"dialect cost" from genuine reasoning errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import psycopg2  # noqa: E402
import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from run_eval import execute, rows_match  # noqa: E402  (same directory)

PG = dict(host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", "5433")),
          dbname=os.getenv("PGDATABASE", "chinook"), user=os.getenv("PGUSER", "postgres"),
          password=os.getenv("PGPASSWORD", "secret"))


def schema_names(db: str) -> tuple[set[str], set[str]]:
    ctx = json.loads((REPO / "View_Selection" / f"{db}_context.json").read_text())
    tables = {t.lower() for t in ctx}
    cols = {c["column_name"].lower() for t in ctx.values() for c in (t.get("columns") or [])}
    return tables, cols


def fix_quoted_literals(tree, tables: set[str], cols: set[str]) -> int:
    """Rewrite double-quoted identifiers that are not real schema names as strings."""
    fixed = 0
    for node in list(tree.find_all(exp.Column)):
        ident = node.this
        if not isinstance(ident, exp.Identifier) or not ident.quoted:
            continue
        name = ident.name
        if name.lower() in cols or name.lower() in tables:
            continue  # a genuine quoted identifier
        node.replace(exp.Literal.string(name))
        fixed += 1
    return fixed


def fix_group_by(tree) -> int:
    """Append selected non-aggregated columns that are missing from GROUP BY."""
    fixed = 0
    for select in list(tree.find_all(exp.Select)):
        group = select.args.get("group")
        if not group:
            continue
        grouped = {g.sql(dialect="postgres").lower() for g in group.expressions}
        for proj in select.expressions:
            target = proj.this if isinstance(proj, exp.Alias) else proj
            if isinstance(target, exp.AggFunc) or target.find(exp.AggFunc):
                continue
            if not isinstance(target, exp.Column):
                continue
            key = target.sql(dialect="postgres").lower()
            if key in grouped:
                continue
            # also accept a bare-name match against a qualified grouping
            if any(key.split(".")[-1] == g.split(".")[-1] for g in grouped):
                continue
            group.append("expressions", target.copy())
            grouped.add(key)
            fixed += 1
    return fixed


def repair(sql: str, tables: set[str], cols: set[str]) -> tuple[str, list[str]]:
    if not sql:
        return sql, []
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return sql, []
    if tree is None:
        return sql, []
    applied = []
    if fix_quoted_literals(tree, tables, cols):
        applied.append("quoted-literal")
    if fix_group_by(tree):
        applied.append("group-by")
    if not applied:
        return sql, []
    return tree.sql(dialect="postgres") + ";", applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "evaluation" / "results"))
    ap.add_argument("--db", default="chinook")
    ap.add_argument("--out", default=str(REPO / "evaluation" / "results" / "_dialect_counterfactual.json"))
    args = ap.parse_args()

    tables, cols = schema_names(args.db)
    conn = psycopg2.connect(**PG)
    report = {}

    files = sorted(p for p in Path(args.results_dir).glob("*.json") if not p.name.startswith("_"))
    for f in files:
        data = json.loads(f.read_text())
        before = sum(r["exec_match"] for r in data["records"])
        after = 0
        repaired_rows = []
        for r in data["records"]:
            sql, applied = repair(r["pred_sql"], tables, cols)
            if r["exec_match"]:
                after += 1
                continue
            if not applied:
                continue
            gold_rows, _ = execute(conn, r["gold_sql"])
            pred_rows, err = execute(conn, sql)
            ok = rows_match(gold_rows, pred_rows, r["gold_sql"])
            after += ok
            repaired_rows.append({"id": r["id"], "applied": applied, "now_correct": ok,
                                  "repaired_sql": sql, "error": err})
        n = len(data["records"])
        report[f.stem] = {
            "n": n,
            "accuracy_before": round(before / n, 4),
            "accuracy_after_dialect_fix": round(after / n, 4),
            "recovered": after - before,
            "repairs": repaired_rows,
        }
        print(f"{f.stem:<34} {before}/{n} -> {after}/{n}   (+{after - before})")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
