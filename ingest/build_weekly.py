"""Load the SAGIS weekly files into the warehouse.

Tables
  sagis_weekly       one row per (season, week, grain_class, flow)  -- deliveries / exports / imports
  weekly_ingest_log  one row per weekly file with parse status

Run:  python ingest/download_sagis.py --weekly
      python ingest/build_weekly.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DB_PATH  # noqa: E402
from ingest.sagis_weekly_parser import parse_all  # noqa: E402


def build(db_path: Path = DB_PATH, raw_dir: Path = DATA_DIR / "raw" / "sagis_weekly") -> dict:
    r = parse_all(raw_dir)
    weekly, log = r["weekly"], r["log"]
    if weekly.empty:
        return {"db": str(db_path), "rows": 0, "failed": log.to_dict("records")}
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE sagis_weekly AS SELECT * FROM weekly ORDER BY flow, grain_class, week_end")
        con.execute("CREATE OR REPLACE TABLE weekly_ingest_log AS SELECT * FROM log")
    finally:
        con.close()
    return {"db": str(db_path), "rows": len(weekly), "seasons": sorted(weekly.season.unique()),
            "last_week_end": weekly.week_end.max(), "failed": log[~log.ok].to_dict("records")}


if __name__ == "__main__":
    res = build()
    print(res["db"], "rows:", res["rows"], "last week:", res.get("last_week_end"))
    for f in res["failed"]:
        print("  FAILED", f["file"], "->", f["error"])
