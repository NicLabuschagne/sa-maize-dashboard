"""SAGIS weekly producer deliveries and RSA imports/exports: loader and season-to-date pace.

Separate from warehouse.py so a deployed app picks it up without a restart.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from config import DB_PATH

FLOWS = {"deliveries": "Producer deliveries", "exports": "Exports", "imports": "Imports"}


@st.cache_data
def load_weekly() -> pd.DataFrame:
    """Empty frame if the table has not been built."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            w = con.execute("SELECT * FROM sagis_weekly").df()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - table not built yet
        return pd.DataFrame(columns=["season", "week", "week_end", "grain_class", "flow",
                                     "tons_week", "tons_prog", "available_date"])
    return with_total(w)


def with_total(w: pd.DataFrame) -> pd.DataFrame:
    """Trade is published per class only; add total = white + yellow where the file lacks it."""
    have = set(w.loc[w.grain_class == "total", "flow"])
    parts = w[w.grain_class.isin(["white", "yellow"]) & ~w.flow.isin(have)]
    if parts.empty:
        return w
    keys = ["season", "week", "week_end", "flow", "available_date"]
    tot = parts.groupby(keys, as_index=False)[["tons_week", "tons_prog"]].sum(min_count=1)
    tot["grain_class"] = "total"
    return pd.concat([w, tot], ignore_index=True)


def as_of(w: pd.DataFrame, date: pd.Timestamp | None) -> pd.DataFrame:
    """Only the weeks that had been published by `date`."""
    return w if date is None else w[w.available_date <= pd.Timestamp(date)]


def season_to_date(w: pd.DataFrame, flow: str, grain_class: str) -> pd.DataFrame:
    """Cumulative tons by week of marketing year, one column per season.

    Built from the weekly column rather than the progressive one, so a holiday week that folds
    two weeks together shows as a flat step, not a gap.
    """
    d = w[(w.flow == flow) & (w.grain_class == grain_class)]
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index="week", columns="season", values="tons_week", aggfunc="sum")
    last = d.groupby("season").week.max()
    cum = piv.fillna(0.0).cumsum()
    for s_, wk in last.items():                       # no path beyond the last published week
        cum.loc[cum.index > wk, s_] = np.nan
    return cum.sort_index(axis=1)


def pace(cum: pd.DataFrame, n_avg: int = 5) -> dict:
    """Latest season to date against last season and the prior-n average at the same week."""
    if cum.empty:
        return {}
    cur = cum.columns[-1]
    wk = int(cum[cur].last_valid_index())
    now = float(cum.at[wk, cur])
    prior = cum.loc[wk, cum.columns[:-1]].dropna()
    last = float(prior.iloc[-1]) if len(prior) else np.nan
    avg = float(prior.iloc[-n_avg:].mean()) if len(prior) else np.nan
    return {"season": cur, "week": wk, "to_date": now, "last_season": last, f"avg_{n_avg}y": avg,
            "vs_last_pct": (now / last - 1) * 100 if last else np.nan,
            "vs_avg_pct": (now / avg - 1) * 100 if avg else np.nan}
