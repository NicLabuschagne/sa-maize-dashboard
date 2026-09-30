"""Daily import/export parity bands and SAFEX's position inside them.

Three bands are built for each class:

    hybrid   (primary)  export edge from the SAFEX/world basis floor; width from the SAGIS cost width
                        (freight x ZAR + rail, which explains ~98% of it), scaled to where SAFEX has
                        actually peaked relative to that width
    implied             both edges from basis percentiles, so width is proportional to world price
    sagis               SAGIS Randfontein import and export parity as published

Every percentile uses days strictly before t, so a band value never depends on the price it is
compared with. Positions are never clipped: 0 = export edge, 1 = import edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import features as F


def world_price_rand(snapshots: pd.DataFrame, bushels_per_tonne: float) -> pd.Series:
    """CBOT corn (USD/bushel) in rand per tonne, using both prints at 10:00 UTC, before the SAFEX mark."""
    world = snapshots["cbot_usd_per_bushel"] * bushels_per_tonne * snapshots["usdzar"]
    return pd.Series(world.to_numpy(), index=pd.DatetimeIndex(snapshots["date"]), name="world").dropna()


def front_month(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Front main-month close per trade date, with the dashboard's roll rule (7 days before expiry)."""
    continuous = F.continuous(prices)
    front = continuous[continuous.symbol == symbol][["trade_date", "close_1", "expiry_1", "days_to_expiry_1"]]
    front = front.rename(columns={"trade_date": "date", "close_1": "safex", "expiry_1": "expiry",
                                  "days_to_expiry_1": "days_to_expiry"})
    front["date"] = front["date"].astype("datetime64[ns]")
    return front.dropna(subset=["safex"]).sort_values("date").reset_index(drop=True)


def attach_world_and_costs(front: pd.DataFrame, world: pd.Series, parity: pd.DataFrame,
                           tolerance_days: int) -> pd.DataFrame:
    """Join each SAFEX day to the latest world price (within tolerance) and the latest published SAGIS
    band. SAGIS rows join on `available_date`, so a Friday calculation is used only once published."""
    world_frame = world.rename("world").reset_index().rename(columns={"index": "world_date"})
    world_frame.columns = ["world_date", "world"]
    frame = pd.merge_asof(front, world_frame.sort_values("world_date"), left_on="date", right_on="world_date",
                          direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
    sagis = parity.dropna(subset=["export_randfontein", "import_randfontein"])
    sagis = sagis[["available_date", "export_randfontein", "import_randfontein"]].sort_values("available_date")
    sagis = sagis.rename(columns={"export_randfontein": "sagis_export", "import_randfontein": "sagis_import"})
    frame = pd.merge_asof(frame, sagis, left_on="date", right_on="available_date", direction="backward")
    frame["cost_width"] = frame["sagis_import"] - frame["sagis_export"]
    frame = frame.dropna(subset=["world"]).reset_index(drop=True)
    frame["basis"] = np.log(frame["safex"] / frame["world"])
    return frame.drop(columns=["world_date", "available_date"])


def prior_day_edges(frame: pd.DataFrame, export_quantile: float, import_quantile: float,
                    min_history_days: int) -> pd.DataFrame:
    """Hybrid and implied band edges on each day, from percentiles over days strictly before it.

    Hybrid ceiling: today's export edge is re-applied to every prior day, and the excess of SAFEX over
    it is measured in units of that day's cost width. The 95th percentile of that excess, times today's
    cost width, is how far above the floor SAFEX has historically been able to go. Recomputing the
    excess with today's floor (rather than each day's own floor) keeps both edges on one definition
    and needs only `min_history_days` of history, not twice that.
    """
    safex, world = frame["safex"].to_numpy(), frame["world"].to_numpy()
    basis, width = frame["basis"].to_numpy(), frame["cost_width"].to_numpy()
    n = len(frame)
    columns = {name: np.full(n, np.nan) for name in
               ("floor_basis", "ceiling_basis", "excess_quantile")}
    for t in range(min_history_days, n):
        floor_basis = np.quantile(basis[:t], export_quantile)
        columns["floor_basis"][t] = floor_basis
        columns["ceiling_basis"][t] = np.quantile(basis[:t], import_quantile)
        excess = (safex[:t] - world[:t] * np.exp(floor_basis)) / width[:t]
        if np.isfinite(excess).sum() >= min_history_days:
            columns["excess_quantile"][t] = np.nanquantile(excess, import_quantile)
    out = frame.copy()
    for name, values in columns.items():
        out[name] = values
    out["hybrid_export"] = out["world"] * np.exp(out["floor_basis"])
    out["hybrid_import"] = out["hybrid_export"] + out["excess_quantile"] * out["cost_width"]
    out["implied_export"] = out["hybrid_export"]
    out["implied_import"] = out["world"] * np.exp(out["ceiling_basis"])
    return out


def band_position(price: pd.Series, export_edge: pd.Series, import_edge: pd.Series) -> pd.Series:
    """Where price sits in the band: 0 at the export edge, 1 at the import edge. Not clipped, so days
    outside the band stay visible as values below 0 or above 1."""
    return (price - export_edge) / (import_edge - export_edge)


def daily_band_frame(prices: pd.DataFrame, snapshots: pd.DataFrame, parity: pd.DataFrame, symbol: str,
                     settings: dict) -> pd.DataFrame:
    """One row per SAFEX trading day: price, world price, the three bands and a position in each."""
    band_settings = settings["band"]
    world = world_price_rand(snapshots, settings["units"]["bushels_per_tonne"])
    frame = attach_world_and_costs(front_month(prices, symbol), world, parity,
                                   band_settings["world_price_tolerance_days"])
    frame = prior_day_edges(frame, band_settings["export_quantile"], band_settings["import_quantile"],
                            band_settings["min_history_days"])
    for band in ("hybrid", "implied", "sagis"):
        frame[f"position_{band}"] = band_position(frame["safex"], frame[f"{band}_export"], frame[f"{band}_import"])
    frame["position"] = frame["position_hybrid"]
    return frame


def outside_band_summary(position: pd.Series) -> dict:
    """How often and for how long price sits outside the band.

    A spell is a run of consecutive trading days on the same side (below 0 or above 1). Lengths are in
    trading days. Returned as a dict so it can go straight into a results table.
    """
    values = position.dropna().to_numpy()
    side = np.where(values < 0, -1, np.where(values > 1, 1, 0))
    spells = {-1: [], 1: []}
    run_side, run_length = 0, 0
    for s in np.append(side, 0):
        if s == run_side and s != 0:
            run_length += 1
            continue
        if run_side != 0:
            spells[run_side].append(run_length)
        run_side, run_length = s, 1
    below, above = np.array(spells[-1]), np.array(spells[1])
    return {
        "days": int(len(values)),
        "share_below": float((side == -1).mean()) if len(values) else np.nan,
        "share_above": float((side == 1).mean()) if len(values) else np.nan,
        "spells_below": int(len(below)), "median_spell_below": float(np.median(below)) if len(below) else 0.0,
        "longest_spell_below": int(below.max()) if len(below) else 0,
        "spells_above": int(len(above)), "median_spell_above": float(np.median(above)) if len(above) else 0.0,
        "longest_spell_above": int(above.max()) if len(above) else 0,
    }
