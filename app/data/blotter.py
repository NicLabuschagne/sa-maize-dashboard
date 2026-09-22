"""Rows for the home-page monitor: one line per tradeable market.

Each line carries the market level, the model's fair value in the same units, the deviation
between them, the standardised deviation (z), months of cover and the next expected S&D release.

Fair value in Model A is fitted on the *real* (CPI-deflated) price, so it is converted back to
nominal rand here: residual = log(actual) - log(fair), therefore nominal_fair = close / exp(residual).
The deflator cancels, so no CPI lookup is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (model, grain_class) -> (display symbol, description, unit)
MARKETS: list[tuple[str, str, str, str, str]] = [
    ("A", "white", "WMAZ", "SAFEX White Maize, front", "R/t"),
    ("A", "yellow", "YMAZ", "SAFEX Yellow Maize, front", "R/t"),
    ("B", "white", "WMAZ 2nd-1st", "White calendar spread", "% ann"),
    ("B", "yellow", "YMAZ 2nd-1st", "Yellow calendar spread", "% ann"),
    ("C", "white_vs_yellow", "WMAZ-YMAZ", "White premium over yellow", "% of yellow"),
    ("D", "white", "WMAZ/parity", "White vs CBOT import parity", "%"),
    ("D", "yellow", "YMAZ/parity", "Yellow vs CBOT import parity", "%"),
]


@dataclass
class Row:
    symbol: str
    description: str
    unit: str
    market: float | None
    fair: float | None
    dev: float | None          # market - fair, in the unit's own terms
    dev_pct: bool              # True when dev should read as a percentage move
    z: float | None
    cover: float | None
    as_of: pd.Timestamp | None


def next_release(vintages: pd.Series, today: pd.Timestamp | None = None) -> tuple[pd.Timestamp | None, int | None]:
    """Project the next SAGIS release from the historical day-of-month pattern."""
    v = pd.to_datetime(pd.Series(vintages)).dropna().sort_values()
    if v.empty:
        return None, None
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    day = int(v.tail(24).dt.day.median())
    nxt = v.max()
    for _ in range(36):                      # walk forward a month at a time until it is in the future
        nxt = (nxt + pd.offsets.MonthBegin(1))
        cand = nxt.replace(day=min(day, nxt.days_in_month))
        if cand > today:
            return cand, int((cand - today).days)
        nxt = cand
    return None, None


def build_rows(signals: pd.DataFrame) -> list[Row]:
    rows: list[Row] = []
    for model, cls, sym, desc, unit in MARKETS:
        d = signals[(signals.model == model) & (signals.grain_class == cls)]
        d = d.dropna(subset=["actual"]).sort_values("vintage_date")
        if d.empty:
            rows.append(Row(sym, desc, unit, None, None, None, False, None, None, None))
            continue
        last = d.iloc[-1]
        resid = last.residual if pd.notna(last.residual) else None
        if model == "A":
            market = float(last.front_close) if pd.notna(last.front_close) else None
            fair = market / np.exp(resid) if (market and resid is not None) else None
            dev = (np.exp(resid) - 1) * 100 if resid is not None else None
            pct = True
        elif model == "D":
            market = (np.exp(last.actual) - 1) * 100
            fair = (np.exp(last.fair_value) - 1) * 100 if pd.notna(last.fair_value) else None
            dev = (np.exp(resid) - 1) * 100 if resid is not None else None
            pct = True
        else:                                 # B and C are already in percentage points
            market = float(last.actual)
            fair = float(last.fair_value) if pd.notna(last.fair_value) else None
            dev = float(resid) if resid is not None else None
            pct = False
        rows.append(Row(sym, desc, unit, market, fair, dev, pct,
                        float(last.z) if pd.notna(last.z) else None,
                        float(last.months_cover) if pd.notna(last.months_cover) else None,
                        last.vintage_date))
    return rows


def fmt(v: float | None, unit: str, places: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if unit == "R/t":
        return f"{v:,.0f}"
    return f"{v:+.{max(places, 1)}f}" if unit != "R/t" else f"{v:,.0f}"
