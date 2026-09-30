"""Raw inputs for the band-position study, and a point-in-time cut of them.

Every table carries the date on which it became usable:

    prices          trade_date      SAFEX close, known at the close
    snapshots       date            CBOT and USD/ZAR at 10:00 UTC, before the 12:00 SAST SAFEX mark
    parity          available_date  SAGIS weekly parity, assumed usable a week after its Friday
    balance_sheet   vintage_date    SAGIS monthly release date
    weekly          available_date  SAGIS weekly deliveries / trade publication date
    cpi             date + 45 days  handled inside fairvalue.real_price (baseline model only)

`inputs_as_of` drops everything not yet usable on a date. The truncation test rebuilds the whole
study from that cut and checks nothing at the cut date changes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pandas as pd

from config import DB_PATH


@dataclass(frozen=True)
class Inputs:
    """The five source tables plus CPI, each with a column saying when it became known."""

    prices: pd.DataFrame
    snapshots: pd.DataFrame
    parity: pd.DataFrame
    balance_sheet: pd.DataFrame
    weekly: pd.DataFrame
    cpi: pd.DataFrame


def _read(connection: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run a query and normalise every timestamp column to nanosecond precision, so joins line up."""
    frame = connection.execute(sql).df()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype("datetime64[ns]")
    return frame


def load_inputs(db_path: Path = DB_PATH) -> Inputs:
    """Read the warehouse tables the study needs. Snapshots are pivoted to one row per date."""
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        prices = _read(connection, "SELECT * FROM prices WHERE symbol IN ('WMAZ', 'YMAZ')")
        snapshots_long = _read(connection, "SELECT * FROM macro_snap")
        parity = _read(connection, "SELECT * FROM sagis_parity ORDER BY date")
        balance_sheet = _read(connection, "SELECT * FROM balance_sheet")
        weekly = _read(connection, "SELECT * FROM sagis_weekly")
        cpi = _read(connection, "SELECT * FROM macro WHERE series = 'za_cpi'")
    finally:
        connection.close()
    snapshots = (snapshots_long.pivot(index="date", columns="series", values="value")
                 .rename(columns={"cbot_corn_safexclose": "cbot_usd_per_bushel",
                                  "usdzar_safexclose": "usdzar"})
                 .sort_index().reset_index())
    snapshots.columns.name = None
    return Inputs(prices, snapshots, parity, balance_sheet, weekly, cpi)


def inputs_as_of(inputs: Inputs, as_of: pd.Timestamp) -> Inputs:
    """Keep only rows that had been published by `as_of` (inclusive). CPI keeps its own 45-day lag
    downstream, so here it is cut on its reference date, which is conservative."""
    as_of = pd.Timestamp(as_of)
    return replace(
        inputs,
        prices=inputs.prices[inputs.prices.trade_date <= as_of],
        snapshots=inputs.snapshots[inputs.snapshots.date <= as_of],
        parity=inputs.parity[inputs.parity.available_date <= as_of],
        balance_sheet=inputs.balance_sheet[inputs.balance_sheet.vintage_date <= as_of],
        weekly=inputs.weekly[inputs.weekly.available_date <= as_of],
        cpi=inputs.cpi[inputs.cpi.date <= as_of],
    )
