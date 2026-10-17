#!/usr/bin/env python3
"""Drive the Lyntara SPA and capture the full user journey as PNGs.

Each run produces a numbered set of screenshots for one configuration
(model + schema mode), so the two retrieval branches can be compared
side by side visually.

Usage:
  python evaluation/capture_dashboard.py --label 9b-embeddings \
      --question "List the names of the top 5 artists with the most albums."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]

DB = dict(host="localhost", port="5433", name="chinook", user="postgres", password="secret")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="subfolder name for this configuration")
    ap.add_argument("--question", default="List the names of the top 5 artists with the most albums.")
    ap.add_argument("--url", default="http://localhost:5173")
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "screenshots"))
    ap.add_argument("--gen-timeout", type=int, default=300_000, help="ms to wait for SQL generation")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir) / args.label
    out.mkdir(parents=True, exist_ok=True)
    shots: list[str] = []

    def shot(n: int, name: str, page) -> None:
        p = out / f"{n:02d}-{name}.png"
        page.screenshot(path=str(p))
        shots.append(p.name)
        print(f"  [{n:02d}] {name}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page.set_default_timeout(60_000)

        print(f"-> {args.url}")
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("#s-connect.active")
        shot(1, "connect-screen-default", page)

        # Fill real credentials
        page.fill("#inp-host", DB["host"])
        page.fill("#inp-port", DB["port"])
        page.fill("#inp-dbname", DB["name"])
        page.fill("#inp-user", DB["user"])
        page.fill("#inp-pass", DB["password"])
        shot(2, "connect-credentials-filled", page)

        # Test connection -> status line reports Postgres + Qdrant reachability
        page.click(".test-btn")
        page.wait_for_timeout(4000)
        shot(3, "connection-tested", page)

        # Enter the app (skips indexing when the database is already indexed)
        page.click(".connect-btn")
        page.wait_for_selector("#s-loading.active")
        page.wait_for_timeout(1500)
        shot(4, "pipeline-status", page)

        page.wait_for_selector("#s-app.active", timeout=180_000)
        page.wait_for_timeout(1500)
        shot(5, "workspace-ready", page)

        # Schema explorer
        page.click("text=Schema explorer")
        page.wait_for_timeout(1500)
        shot(6, "schema-explorer", page)
        page.click("text=Query workspace")
        page.wait_for_timeout(600)

        # Ask the question
        page.fill("#nl-input", args.question)
        shot(7, "question-entered", page)

        page.click(".run-btn")
        # Generation on a local model is slow; wait for the SQL panel to change
        page.wait_for_function(
            """() => {
                 const el = document.getElementById('sql-display');
                 if (!el) return false;
                 const t = el.textContent || '';
                 return t.length > 0 && !t.includes('total_revenue');
               }""",
            timeout=args.gen_timeout,
        )
        page.wait_for_timeout(1200)
        shot(8, "sql-generated", page)

        # Matched tables panel -- the visible difference between the two modes
        try:
            page.locator("#tables-matched").scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            shot(9, "tables-matched", page)
        except Exception as exc:  # noqa: BLE001
            print(f"  (tables-matched skipped: {type(exc).__name__})")

        # Execute against PostgreSQL
        page.click("button.copy-btn:has-text('Execute')")
        page.wait_for_timeout(8000)
        try:
            page.locator("#res-body").scroll_into_view_if_needed()
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(600)
        shot(10, "query-results", page)

        # Whole page, for the record
        page.screenshot(path=str(out / "11-full-page.png"), full_page=True)
        shots.append("11-full-page.png")
        print("  [11] full-page")

        browser.close()

    print(f"\n{len(shots)} screenshots -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
