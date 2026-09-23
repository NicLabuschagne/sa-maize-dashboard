"""Load SAGIS historic maize import/export parity into the warehouse.

Tables
  sagis_parity   one row per weekly calculation date: parity components, band at Randfontein,
                 prime rate backed out of the financing line, available_date

Run:  python ingest/build_parity.py   (raw workbooks in DATA_DIR/raw/sagis_parity, from
      https://www.sagis.org.za/sagis-historic-information/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DB_PATH  # noqa: E402
from ingest.sagis_parity_parser import parse_parity  # noqa: E402


def build(db_path: Path = DB_PATH, raw_dir: Path = DATA_DIR / "raw" / "sagis_parity") -> dict:
    r = parse_parity(raw_dir)
    parity = r["parity"]
    if parity.empty:
        return {"db": str(db_path), "rows": 0, "error": r["error"]}
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE sagis_parity AS SELECT * FROM parity ORDER BY date")
    finally:
        con.close()
    return {"db": str(db_path), "rows": len(parity), "first": parity.date.min(), "last": parity.date.max(),
            "error": None}


if __name__ == "__main__":
    print(build())
