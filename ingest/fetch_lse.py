"""Fetch macro/reference series from London Strategic Edge into the warehouse `macro` table.

Series (symbol -> role):
    southafriconpriindcp  ZA CPI index (monthly)     -> deflator for real prices
    USD/ZAR               daily close                -> currency, v2 parity work
    CORN/USD              daily close (CBOT proxy)   -> v2 parity work

Requires LSE_API_KEY in the environment.  Run: python ingest/fetch_lse.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH  # noqa: E402

SERIES = {"southafriconpriindcp": "za_cpi"}
CANDLES = {"USD/ZAR": "usdzar", "CORN/USD": "cbot_corn"}


def fetch_macro(api_key: str | None = None, start: str = "2005-01-01") -> dict:
    from lse import LSE  # imported here so the rest of the app never needs the SDK

    key = api_key or os.environ.get("LSE_API_KEY")
    if not key:
        return {"ok": False, "rows": 0, "error": "LSE_API_KEY not set"}
    c = LSE(api_key=key)
    frames = []
    for sym, name in SERIES.items():
        rows = c.series(sym, start=start, limit=5000)
        frames.append(pd.DataFrame(rows).assign(series=name)[["series", "date", "value"]])
    for sym, name in CANDLES.items():
        rows: list[dict] = []
        for year in range(int(start[:4]), pd.Timestamp.today().year + 1):  # 5000-row cap per call
            rows += c.candles(sym, "1d", start=f"{year}-01-01", end=f"{year}-12-31", limit=5000, order="asc")
        d = pd.DataFrame(rows)
        frames.append(pd.DataFrame({"series": name, "date": d["timestamp"].str[:10], "value": d["close"]}))
    m = pd.concat(frames, ignore_index=True)
    m["date"] = pd.to_datetime(m["date"])
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m = m.dropna().drop_duplicates(["series", "date"]).sort_values(["series", "date"]).reset_index(drop=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE macro AS SELECT * FROM m")
    con.close()
    return {"ok": True, "rows": len(m), "error": None,
            "coverage": m.groupby("series").date.agg(["min", "max", "count"]).to_dict("index")}


if __name__ == "__main__":
    res = fetch_macro()
    print(res)
