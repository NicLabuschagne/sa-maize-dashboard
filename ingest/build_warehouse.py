"""Build the DuckDB warehouse from raw SAFEX and SAGIS files.

Tables
  prices          one row per (symbol, trade_date, expiry)      -- SAFEX contracts
  balance_sheet   one row per (vintage_date, period_type,
                               attribute, grain_class)          -- SAGIS point-in-time
  ingest_log      one row per SAGIS file with parse status

Run:  python ingest/build_warehouse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DB_PATH  # noqa: E402
from ingest.safex_loader import load_safex  # noqa: E402
from ingest.sagis_parser import parse_sagis_file  # noqa: E402


def build_balance_sheet(sagis_dir: Path) -> dict:
    frames, log = [], []
    for f in sorted(sagis_dir.glob("*.xls")):
        r = parse_sagis_file(f)
        log.append({"file": f.name, "ok": r["ok"], "error": r["error"],
                    "n_rows": (r["meta"] or {}).get("n_rows")})
        if r["ok"]:
            frames.append(r["rows"])
    bs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"balance_sheet": bs, "log": pd.DataFrame(log)}


def build(db_path: Path = DB_PATH) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    safex = load_safex(DATA_DIR / "raw")
    sagis = build_balance_sheet(DATA_DIR / "raw" / "sagis")
    prices, bs, log = safex["prices"], sagis["balance_sheet"], sagis["log"]

    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices")
    con.execute("CREATE OR REPLACE TABLE balance_sheet AS SELECT * FROM bs")
    con.execute("CREATE OR REPLACE TABLE ingest_log AS SELECT * FROM log")
    con.close()
    return {"db": str(db_path), "prices": safex["meta"],
            "balance_sheet": {"rows": len(bs), "vintages": int(bs.vintage_date.nunique()) if len(bs) else 0,
                              "first": bs.vintage_date.min() if len(bs) else None,
                              "last": bs.vintage_date.max() if len(bs) else None},
            "sagis_failed": log[~log.ok][["file", "error"]].to_dict("records")}


if __name__ == "__main__":
    res = build()
    print(res["db"])
    print("prices:", res["prices"])
    print("balance_sheet:", res["balance_sheet"])
    print(f"sagis files failed: {len(res['sagis_failed'])}")
    for r in res["sagis_failed"]:
        print("  ", r["file"], "->", r["error"])
