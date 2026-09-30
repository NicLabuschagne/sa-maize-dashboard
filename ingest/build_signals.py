"""Materialise fitted model output into the warehouse as table `signals`.

The pages compute these in memory; writing them to DuckDB lets the Ask page answer
questions about fair value, residuals and z-scores with plain SQL.

One row per (model, grain_class, vintage_date). `actual` and `fair_value` are in the
units of that model's left-hand side:
    A  log(real price, 90-day constant maturity)
    B  calendar spread, % annualised
    C  white premium, % of yellow
    D  log(SAFEX / world parity)   -- the basis

Run:  python ingest/build_signals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import fairvalue as FV  # noqa: E402
from app.data import features as F  # noqa: E402
from config import DB_PATH  # noqa: E402

KEEP = ["vintage_date", "latest_month", "marketing_year", "my_month", "months_cover",
        "y", "fv", "resid", "z", "close_1", "close_cm", "world_rand", "basis"]


def _tidy(panel: pd.DataFrame, model: str, grain_class: str) -> pd.DataFrame:
    d = pd.DataFrame({"model": model, "grain_class": grain_class}, index=panel.index)
    for c in KEEP:
        d[c] = panel[c] if c in panel.columns else pd.NA
    for h in FV.HORIZONS:
        d[f"fwd_{h}"] = panel[f"fwd_{h}"] if f"fwd_{h}" in panel.columns else pd.NA
    return d.rename(columns={"y": "actual", "fv": "fair_value", "resid": "residual",
                             "close_1": "front_close", "close_cm": "price_cm"})


def build(db_path: Path = DB_PATH) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        snap = con.execute("SELECT * FROM macro_snap").df() if "macro_snap" in tables else pd.DataFrame()
    finally:
        con.close()

    sd, cont = F.sd_monthly(bs), F.continuous(px)
    wy = F.white_yellow_spread(cont)
    frames = []
    for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
        frames.append(_tidy(FV.fit_expanding(FV.panel_price(sd, cont, cpi, cls, sym)).panel, "A", cls))
        frames.append(_tidy(FV.fit_expanding(FV.panel_spread(sd, cont, cls, sym)).panel, "B", cls))
        if len(snap):
            p = FV.panel_parity(sd, cont, snap, cpi, cls, sym)
            if len(p):
                frames.append(_tidy(FV.fit_expanding(p).panel, "D", cls))
    frames.append(_tidy(FV.fit_expanding(FV.panel_white_yellow(sd, wy, bs)).panel, "C", "white_vs_yellow"))

    sig = pd.concat(frames, ignore_index=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE signals AS SELECT * FROM sig")
    finally:
        con.close()
    return {"rows": len(sig),
            "by_model": sig.groupby(["model", "grain_class"]).size().to_dict(),
            "first": sig.vintage_date.min(), "last": sig.vintage_date.max()}


if __name__ == "__main__":
    print(build())
