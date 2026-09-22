"""Shared sidebar selectors and cached derived frames used by every page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import features as F
from app.data.warehouse import load_balance_sheet, load_prices

CLASS_TO_SYMBOL = {"white": "WMAZ", "yellow": "YMAZ", "total": None}


@st.cache_data
def derived() -> dict[str, pd.DataFrame]:
    bs = load_balance_sheet()
    p = load_prices()
    cont = F.continuous(p)
    return {"bs": bs, "prices": p, "sd": F.sd_monthly(bs), "cont": cont, "wy": F.white_yellow_spread(cont)}


def sidebar() -> dict:
    st.sidebar.markdown("### Selection")
    country = st.sidebar.selectbox("Country", ["South Africa"], index=0)
    commodity = st.sidebar.selectbox("Commodity", ["Maize"], index=0)
    grain_class = st.sidebar.radio("Class", ["white", "yellow", "total"], index=0, horizontal=True)
    symbol = CLASS_TO_SYMBOL[grain_class]
    st.sidebar.caption("Price series: " + (f"SAFEX {symbol} front month" if symbol else "n/a for total"))
    return {"country": country, "commodity": commodity, "grain_class": grain_class, "symbol": symbol}


def fmt_t(v: float) -> str:
    return f"{v/1e6:.2f} Mt" if abs(v) >= 1e6 else f"{v/1e3:,.0f} kt"
