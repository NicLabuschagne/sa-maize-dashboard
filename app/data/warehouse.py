"""Read-only access to the DuckDB warehouse, cached for Streamlit."""
from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

from config import DB_PATH


def _query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


@st.cache_data
def load_prices(symbols: tuple[str, ...] = ("WMAZ", "YMAZ")) -> pd.DataFrame:
    return _query("SELECT * FROM prices WHERE symbol IN (SELECT UNNEST(?)) ORDER BY symbol, trade_date, expiry_date",
                  [list(symbols)])


@st.cache_data
def load_balance_sheet() -> pd.DataFrame:
    return _query("SELECT * FROM balance_sheet ORDER BY vintage_date, period_type, attribute, grain_class")


@st.cache_data
def load_ingest_log() -> pd.DataFrame:
    return _query("SELECT * FROM ingest_log ORDER BY file")


@st.cache_data
def load_macro() -> pd.DataFrame:
    return _query("SELECT * FROM macro ORDER BY series, date")


@st.cache_data
def load_macro_snap() -> pd.DataFrame:
    """Series snapped at the SAFEX mark (10:00 UTC). Empty frame if the table is absent."""
    try:
        return _query("SELECT * FROM macro_snap ORDER BY series, date")
    except Exception:  # noqa: BLE001 - table not built yet
        return pd.DataFrame(columns=["series", "date", "value"])


@st.cache_data
def load_signals() -> pd.DataFrame:
    return _query("SELECT * FROM signals ORDER BY model, grain_class, vintage_date")
