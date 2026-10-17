#!/usr/bin/env python3
"""Score a model on the Chinook evaluation set.

Measures *execution accuracy*: run the gold SQL and the model's SQL against the
live database and compare the returned rows. This is the metric Spider's
official evaluation uses, and it is far more informative than exact string
match -- two differently-phrased queries that return the same rows are both
correct.

Retrieval modes:
  embeddings  -- Qdrant top-k schema retrieval (Lyntara's default)
  full        -- the entire database schema is placed in the prompt

Usage:
  python evaluation/run_eval.py --model lyntara-sql-1.7b:step200 --mode embeddings
  python evaluation/run_eval.py --model lyntara-sql:ckpt200 --mode full --tag 9b-full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import psycopg2  # noqa: E402
from Engine.llm import LLMError, OllamaProvider, prompts  # noqa: E402

PG = dict(host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", "5433")),
          dbname=os.getenv("PGDATABASE", "chinook"), user=os.getenv("PGUSER", "postgres"),
          password=os.getenv("PGPASSWORD", "secret"))


def normalise(rows) -> list:
    """Make result rows comparable across type/precision differences."""
    out = []
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, Decimal):
                v = round(float(v), 4)
            elif isinstance(v, float):
                v = round(v, 4)
            elif isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", "replace")
            cells.append(v if v is None else str(v))
        out.append(tuple(cells))
    return out


def execute(conn, sql: str):
    cur = conn.cursor()
    try:
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as exc:
        conn.rollback()
        return None, f"{type(exc).__name__}: {str(exc).strip()[:200]}"
    finally:
        cur.close()


def rows_match(gold_rows, pred_rows, gold_sql: str) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    g, p = normalise(gold_rows), normalise(pred_rows)
    if len(g) != len(p):
        return False
    # Order matters only when the gold query asked for an order.
    if re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE):
        return g == p
    return sorted(g, key=repr) == sorted(p, key=repr)


@lru_cache(maxsize=4)
def _context(db: str) -> str:
    return (REPO / "View_Selection" / f"{db}_context.json").read_text()


@lru_cache(maxsize=2)
def _retriever(db: str):
    """Built once. The embedding model is ~1.6 GB; reloading it per question
    made a 20-question run take longer than the generation itself."""
    from Engine.sql import SchemaRetriever

    return SchemaRetriever(
        collection_name=f"{db}-table",
        context_file=str(REPO / "View_Selection" / f"{db}_context.json"),
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5"),
    )


def load_schema(db: str, mode: str, question: str, top_k: int) -> dict:
    full = json.loads(_context(db))
    if mode == "full":
        return full
    r = _retriever(db)
    hits = r.search_schema(question, top_k=top_k, collection_name=f"{db}-table")
    tables = [h["payload"]["table"] for h in (hits.get("result") or []) if h.get("payload", {}).get("table")]
    picked = {t: full[t] for t in tables if t in full}
    return picked or full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["embeddings", "full"], default="embeddings")
    ap.add_argument("--db", default="chinook")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--base-url", default=os.getenv("LLM_SQL_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--tag", help="label for the output file")
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "results"))
    args = ap.parse_args()

    spec = json.loads((REPO / "evaluation" / "chinook_eval_set.json").read_text())
    questions = spec["questions"]

    provider = OllamaProvider(base_url=args.base_url, model=args.model, timeout=args.timeout,
                              temperature=0.1, max_tokens=512, top_p=0.9, num_ctx=args.num_ctx,
                              stop=["<|im_end|>", "<|im_start|>"])
    conn = psycopg2.connect(**PG)

    tag = args.tag or f"{args.model.replace(':', '_').replace('/', '_')}__{args.mode}"
    print(f"model={args.model}  mode={args.mode}  db={args.db}  n={len(questions)}\n")

    records, correct, total_latency = [], 0, 0.0
    for q in questions:
        schema = load_schema(args.db, args.mode, q["question"], args.top_k)
        prompt = prompts.build_chatml_prompt(q["question"], schema, db_id=args.db)
        t0 = time.time()
        err = None
        try:
            raw = provider.complete(prompt)
        except LLMError as exc:
            raw, err = "", str(exc)[:200]
        latency = time.time() - t0
        total_latency += latency

        pred_sql = prompts.extract_sql(raw) if raw else ""
        gold_rows, gold_err = execute(conn, q["gold_sql"])
        pred_rows, pred_err = (execute(conn, pred_sql) if pred_sql else (None, "no SQL extracted"))
        ok = rows_match(gold_rows, pred_rows, q["gold_sql"])
        correct += ok

        records.append({
            "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
            "gold_sql": q["gold_sql"], "pred_sql": pred_sql, "exec_match": ok,
            "latency_s": round(latency, 2), "tables_in_prompt": sorted(schema),
            "pred_error": pred_err, "provider_error": err,
            "raw_chars": len(raw),
        })
        mark = "OK  " if ok else "MISS"
        print(f"  {q['id']:>3} [{q['difficulty'][:4]}] {mark} {latency:>6.1f}s  {(pred_sql or '<none>')[:72]}")
        if not ok and pred_err:
            print(f"        ! {pred_err[:100]}")

    n = len(questions)
    by_diff = {}
    for r in records:
        d = by_diff.setdefault(r["difficulty"], [0, 0])
        d[1] += 1
        d[0] += r["exec_match"]

    summary = {
        "model": args.model, "mode": args.mode, "database": args.db,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": n, "correct": correct,
        "execution_accuracy": round(correct / n, 4),
        "by_difficulty": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
                          for k, v in sorted(by_diff.items())},
        "mean_latency_s": round(total_latency / n, 2),
        "total_latency_s": round(total_latency, 1),
        "no_sql_extracted": sum(1 for r in records if not r["pred_sql"]),
        "sql_errors": sum(1 for r in records if r["pred_error"] and r["pred_sql"]),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False))

    print(f"\n  execution accuracy : {correct}/{n} = {summary['execution_accuracy']:.1%}")
    for k, v in summary["by_difficulty"].items():
        print(f"    {k:<7} {v['correct']}/{v['n']}  ({v['accuracy']:.0%})")
    print(f"  mean latency       : {summary['mean_latency_s']}s")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
