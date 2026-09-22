"""Load JSE/SAFEX 'Physical Settled Grain contracts' yearly workbooks into one tidy table.

The yearly files overlap in trade dates; overlapping rows are identical, so we
union and dedupe on (trade_date, symbol, expiry). High/Low of 0 mean "no trade"
and are nulled.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = {
    "TradeDate": "trade_date", "ExpiryDate": "expiry_date", "Expiry": "expiry",
    "ShortName": "symbol", "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Change": "change", "Volume": "volume", "OI": "open_interest", "ContractSize": "contract_size",
}


def load_safex(raw_dir: str | Path, symbols: tuple[str, ...] = ("WMAZ", "YMAZ")) -> dict:
    """Return {"ok", "prices": DataFrame, "meta"}; prices sorted by symbol, trade_date, expiry."""
    files = sorted(Path(raw_dir).glob("Physical Settled Grain contracts*.xlsx"))
    if not files:
        return {"ok": False, "prices": None, "meta": {"files": 0}, "error": "no SAFEX files found"}
    frames = []
    for f in files:
        d = pd.read_excel(f, sheet_name="PricingDetail", usecols=list(COLS))
        d = d[d["ShortName"].isin(symbols)].rename(columns=COLS)
        d["source_file"] = f.name
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p["expiry_date"] = pd.to_datetime(p["expiry_date"])
    n_raw = len(p)
    p = p.drop_duplicates(["trade_date", "symbol", "expiry"]).copy()
    for c in ("high", "low"):
        p.loc[p[c] <= 0, c] = pd.NA
    p["days_to_expiry"] = (p["expiry_date"] - p["trade_date"]).dt.days
    p = p.sort_values(["symbol", "trade_date", "expiry_date"]).reset_index(drop=True)
    meta = {"files": len(files), "rows_raw": n_raw, "rows": len(p),
            "first": p.trade_date.min().date(), "last": p.trade_date.max().date()}
    return {"ok": True, "prices": p, "meta": meta, "error": None}
