"""Run the band fair-value model for one class and package what the page needs. Cached per class,
so the band percentiles and weekly refits run once per session (and again when the warehouse changes)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import band_fairvalue as BFV
from app.data.warehouse import load_macro_snap, load_sagis_parity
from app.data.weekly import load_weekly
from app.state import CLASS_TO_SYMBOL, derived

WINDOWS = {"May 2017 – today (all out-of-sample)": ("2017-05-01", "2100-01-01"),
           "2020 – today": ("2020-01-01", "2100-01-01"),
           "2023 – today": ("2023-01-01", "2100-01-01")}


@st.cache_data(show_spinner="Building the parity band and fair value…")
def run_band_model(grain_class: str) -> dict:
    """Daily band, stocks-to-use nowcast and out-of-sample fair value, plus scores by window and year."""
    D = derived()
    daily = BFV.build(D["cont"], load_macro_snap(), load_sagis_parity(), D["sd"], load_weekly(),
                      grain_class, CLASS_TO_SYMBOL[grain_class])
    years = {str(y): (f"{y}-01-01", f"{y}-12-31") for y in sorted(daily.dropna(subset=["fair_position"]).date.dt.year.unique())}
    return {"daily": daily, "scores": BFV.out_of_sample_scores(daily, WINDOWS),
            "scores_by_year": BFV.out_of_sample_scores(daily, years)}
