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

# SAFEX grain marks at 12:00 SAST. South Africa is UTC+2 with no DST, so the snapshot hour is
# fixed at 10:00 UTC. CBOT corn settles 19:20/20:20 UTC, i.e. AFTER the SAFEX close, so the
# same-day settle is not knowable at the mark; the 10:00 UTC bar (overnight Globex) is.
SNAP_HOUR_UTC = 10
SNAP_SYMBOLS = {"USD/ZAR": "usdzar_safexclose", "CORN/USD": "cbot_corn_safexclose"}


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


def fetch_snapshots(api_key: str | None = None, start_year: int = 2009) -> dict:
    """Hourly bars reduced to the bar that closes at the SAFEX mark (10:00 UTC), plus the
    previous CBOT settle as a robustness series. Written to table `macro_snap`."""
    from lse import LSE

    key = api_key or os.environ.get("LSE_API_KEY")
    if not key:
        return {"ok": False, "rows": 0, "error": "LSE_API_KEY not set"}
    c = LSE(api_key=key)
    this_year = pd.Timestamp.today().year
    frames = []
    for sym, name in SNAP_SYMBOLS.items():
        rows: list[dict] = []
        for year in range(start_year, this_year + 1):
            for q0 in (1, 4, 7, 10):  # quarterly: ~2200 hourly bars, under the 5000-row cap
                q1 = q0 + 2
                end = pd.Timestamp(year=year, month=q1, day=1) + pd.offsets.MonthEnd(0)
                rows += c.candles(sym, "1h", start=f"{year}-{q0:02d}-01",
                                  end=end.strftime("%Y-%m-%d"), limit=5000, order="asc")
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        d["ts"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
        d = d.drop_duplicates("ts").sort_values("ts")
        snap = d[d["ts"].dt.hour == SNAP_HOUR_UTC]
        frames.append(pd.DataFrame({"series": name, "date": snap["ts"].dt.normalize(),
                                    "value": snap["close"].to_numpy()}))
    m = pd.concat(frames, ignore_index=True)
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m = m.dropna().drop_duplicates(["series", "date"]).sort_values(["series", "date"]).reset_index(drop=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE macro_snap AS SELECT * FROM m")
    con.close()
    return {"ok": True, "rows": len(m), "error": None,
            "coverage": m.groupby("series").date.agg(["min", "max", "count"]).to_dict("index")}


def fetch_cot(symbols: tuple[str, ...] = ("ZC", "ZW", "ZS"), api_key: str | None = None,
              start: str = "2010-01-01") -> dict:
    """CFTC Commitments of Traders into table `cot`.

    Positions are as of Tuesday; `release_date` is the Friday they became public. Keep both:
    use `date` to ask what positioning was, `release_date` to ask what was knowable.
    """
    from lse import LSE

    key = api_key or os.environ.get("LSE_API_KEY")
    if not key:
        return {"ok": False, "rows": 0, "error": "LSE_API_KEY not set"}
    c = LSE(api_key=key)
    frames = []
    for sym in symbols:
        rows = c.cot(sym, start=start, limit=5000)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return {"ok": False, "rows": 0, "error": "no COT rows returned"}
    d = pd.concat(frames, ignore_index=True)
    keep = ["symbol", "date", "release_date", "open_interest", "noncomm_long", "noncomm_short",
            "noncomm_spread", "comm_long", "comm_short", "nonrept_long", "nonrept_short"]
    d = d[[c_ for c_ in keep if c_ in d.columns]].copy()
    for c_ in ("date", "release_date"):
        if c_ in d:
            d[c_] = pd.to_datetime(d[c_])
    d["net_noncomm"] = d["noncomm_long"] - d["noncomm_short"]
    d["net_noncomm_pct_oi"] = d["net_noncomm"] / d["open_interest"].replace(0, pd.NA)
    d = d.dropna(subset=["date"]).drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE cot AS SELECT * FROM d")
    finally:
        con.close()
    return {"ok": True, "rows": len(d),
            "coverage": d.groupby("symbol").date.agg(["min", "max", "count"]).to_dict("index")}


if __name__ == "__main__":
    print(fetch_macro())
    print(fetch_snapshots())
    print(fetch_cot())
