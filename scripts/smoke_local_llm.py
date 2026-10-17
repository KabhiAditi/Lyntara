#!/usr/bin/env python3
"""Smoke-test the local SQL model end to end.

Renders the Spider training format from a real Lyntara schema, sends it through
the configured provider, and reports the extracted SQL plus timing.

    python scripts/smoke_local_llm.py
    python scripts/smoke_local_llm.py --model qwen3.5:9b --db chinook
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Engine.llm import LLMError, build_provider, prompts  # noqa: E402

QUESTIONS = [
    "How many albums are there?",
    "List the names of the top 5 artists with the most albums.",
    "What is the total number of tracks per genre?",
]


def load_schema(db: str, tables: int) -> dict:
    path = REPO_ROOT / "View_Selection" / f"{db}_context.json"
    if not path.exists():
        sys.exit(f"No schema at {path}. Run the pipeline for '{db}' first.")
    data = json.loads(path.read_text())
    return {k: data[k] for k in list(data)[:tables]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chinook")
    ap.add_argument("--tables", type=int, default=6)
    ap.add_argument("--role", default="sql")
    ap.add_argument("--model", help="override LLM_<ROLE>_MODEL")
    ap.add_argument("--question", action="append", help="ask a specific question")
    args = ap.parse_args()

    import os

    if args.model:
        os.environ[f"LLM_{args.role.upper()}_MODEL"] = args.model

    provider = build_provider(args.role)
    schema = load_schema(args.db, args.tables)
    questions = args.question or QUESTIONS

    print(f"provider : {provider.name}")
    print(f"style    : {provider.prompt_style}")
    print(f"schema   : {len(schema)} tables from {args.db}\n")

    failures = 0
    for q in questions:
        prompt = prompts.build_chatml_prompt(q, schema, db_id=args.db)
        print(f"Q: {q}")
        start = time.time()
        try:
            raw = provider.complete(prompt)
        except LLMError as exc:
            print(f"   ERROR: {exc}\n")
            failures += 1
            continue
        elapsed = time.time() - start

        sql = prompts.extract_sql(raw)
        approx_tokens = max(1, len(raw) // 4)
        print(f"   time : {elapsed:.1f}s  (~{approx_tokens / elapsed:.1f} tok/s)")
        print(f"   sql  : {sql or '<none extracted>'}")
        if not sql:
            print(f"   raw  : {raw[:300]}")
            failures += 1
        print()

    print("FAILURES:" if failures else "All prompts produced SQL.", failures or "")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
