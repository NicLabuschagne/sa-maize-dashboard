"""Export floors revealed by the market, and the criteria used to choose between them (Addendum 2).

    B0 expanding   floor = W x exp(5th percentile of all prior basis)                 (round 2 band)
    B1 season      floor = W x exp(10th pct of harvest-window basis, export seasons only)
    B2 Kalman      floor = W x exp(s) - C, s learnt from days when exports are flowing

W = CBOT x USD/ZAR in R/t; C = SAGIS export deductions (rail, port, financing) in R/t. B2 separates
what is known daily (world price, rand costs) from the one thing that must be learnt: how far SA's
achievable FOB sits from the CBOT-based world price. Every candidate uses ceiling = floor + SAGIS
cost width. No forward return is computed anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.band_position import band
from research.band_position.inputs import Inputs

HARVEST_MONTHS = (5, 6, 7)


# ----------------------------------------------------------------------------- inputs
def parity_frame(inputs: Inputs, grain_class: str, settings: dict) -> pd.DataFrame:
    """Daily SAFEX (constant maturity), world price, USD/ZAR, SAGIS export deductions and cost width,
    plus the round-2 expanding floor (B0). SAGIS values join on `available_date`."""
    symbol = settings["symbols"][grain_class]
    frame = band.daily_band_frame(inputs.prices, inputs.snapshots, inputs.parity, symbol, settings)
    frame = frame[["date", "safex", "world", "basis", "cost_width", "hybrid_export"]]
    frame = frame.rename(columns={"hybrid_export": "floor_b0"})
    snapshots = inputs.snapshots[["date", "usdzar"]].rename(columns={"date": "snap_date"}).dropna()
    frame = pd.merge_asof(frame, snapshots.sort_values("snap_date"), left_on="date", right_on="snap_date",
                          direction="backward", tolerance=pd.Timedelta(days=settings["band"]["world_price_tolerance_days"]))
    sagis = inputs.parity.dropna(subset=["fob_gulf_rand", "export_randfontein"]).copy()
    sagis["export_deductions"] = sagis["fob_gulf_rand"] - sagis["export_randfontein"]
    sagis = sagis[["available_date", "export_deductions"]].sort_values("available_date")
    frame = pd.merge_asof(frame, sagis, left_on="date", right_on="available_date", direction="backward")
    frame["grain_class"] = grain_class
    return frame.drop(columns=["snap_date", "available_date"]).dropna(subset=["usdzar"]).reset_index(drop=True)


def published_pace(weekly: pd.DataFrame, grain_class: str, flow: str, dates: pd.Series, n_weeks: int = 4) -> pd.Series:
    """Mean weekly tonnage (kt) over the latest `n_weeks` weeks *published* by each date.

    Recomputed at each publication date and carried forward, so a day only sees weeks whose
    `available_date` is on or before it.
    """
    rows = weekly[(weekly.grain_class == grain_class) & (weekly.flow == flow)]
    rows = rows.groupby("week_end", as_index=False).agg(tons=("tons_week", "sum"), available=("available_date", "max"))
    rows = rows.sort_values("week_end").reset_index(drop=True)
    events = []
    for publication in np.sort(rows["available"].unique()):
        seen = rows[rows["available"] <= publication]
        events.append({"date": publication, "pace": seen["tons"].iloc[-n_weeks:].mean() / 1e3})
    events = pd.DataFrame(events)
    events["date"] = events["date"].astype("datetime64[ns]")
    left = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]")})
    joined = pd.merge_asof(left, events, on="date", direction="backward")
    return pd.Series(joined["pace"].to_numpy(), index=dates.index, name=f"{flow}_pace")


def marketing_year(dates: pd.Series) -> pd.Series:
    """Marketing year (May–April) as the calendar year it starts in."""
    dates = pd.to_datetime(dates)
    return dates.dt.year - (dates.dt.month < 5).astype(int)


# ----------------------------------------------------------------------------- B1 season rule
def season_rule_floor_basis(frame: pd.DataFrame, weekly: pd.DataFrame, grain_class: str,
                            quantile: float) -> pd.Series:
    """Floor basis per day from the harvest window of the latest *confirmed export season*.

    A season is confirmed when its weekly exports for weeks ending by 31 July exceed its imports for
    the same weeks. The floor becomes usable on the publication date of the last of those weeks
    (trade is the slower series), and carries forward until another season is confirmed.
    """
    rows = weekly[(weekly.grain_class == grain_class) & weekly.flow.isin(["exports", "imports"])]
    usable = []
    for year in sorted(marketing_year(frame["date"]).unique()):
        start, end = pd.Timestamp(f"{year}-05-01"), pd.Timestamp(f"{year}-07-31")
        window = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        weeks = rows[(rows.week_end >= start) & (rows.week_end <= end)]
        if len(window) < 20 or weeks.empty:
            continue
        # the whole window must be in: a cut build mid-harvest must not publish a partial-season floor
        if weeks["week_end"].max() < end - pd.Timedelta(days=6) or frame["date"].max() < end - pd.Timedelta(days=3):
            continue
        flows = weeks.groupby("flow")["tons_week"].sum()
        if flows.get("exports", 0.0) <= flows.get("imports", 0.0):
            continue
        usable.append({"date": weeks["available_date"].max(), "floor_basis": window["basis"].quantile(quantile)})
    if not usable:
        return pd.Series(np.nan, index=frame.index)
    table = pd.DataFrame(usable).sort_values("date")
    table["date"] = table["date"].astype("datetime64[ns]")
    left = pd.DataFrame({"date": frame["date"].astype("datetime64[ns]")})
    joined = pd.merge_asof(left, table, on="date", direction="backward")
    return pd.Series(joined["floor_basis"].to_numpy(), index=frame.index)


# ----------------------------------------------------------------------------- B2 Kalman
@dataclass(frozen=True)
class KalmanSettings:
    """Pre-registered filter settings (Addendum 2). All in log-competitiveness units."""

    daily_drift_sd: float = 0.002
    season_jump_sd: float = 0.15
    below_sd: float = 0.02
    above_sd: float = 0.05
    pace_floor_kt: float = 5.0
    pace_full_kt: float = 30.0
    initial_sd: float = 0.3


def pace_weight(pace: np.ndarray, floor_kt: float, full_kt: float) -> np.ndarray:
    """0 below `floor_kt` (border trickle), 1 at or above `full_kt` (a full export programme), linear
    between. Missing pace counts as 0."""
    weight = (np.nan_to_num(pace, nan=0.0) - floor_kt) / (full_kt - floor_kt)
    return np.clip(weight, 0.0, 1.0)


def kalman_floor(frame: pd.DataFrame, export_pace: pd.Series, params: KalmanSettings) -> pd.DataFrame:
    """One-sided Kalman filter for the export competitiveness factor.

    Observation z = log((SAFEX + C) / W) equals the state when SAFEX is at export parity and is above
    it otherwise. So a price below the current estimate always pulls the floor down (price cannot sit
    under export parity for long). A price above it pulls the floor up only while exports are
    flowing, with weight rising with pace. The floor reported for day t uses the state *before* day
    t's price, so it never depends on the price it is compared with.
    """
    z = np.log((frame["safex"] + frame["export_deductions"]) / frame["world"]).to_numpy()
    weight = pace_weight(export_pace.to_numpy(), params.pace_floor_kt, params.pace_full_kt)
    year = marketing_year(frame["date"]).to_numpy()
    n = len(frame)
    prior_state, prior_sd = np.full(n, np.nan), np.full(n, np.nan)
    state, variance = np.nan, np.nan
    for t in range(n):
        if not np.isfinite(z[t]):
            if np.isfinite(state):
                prior_state[t], prior_sd[t] = state, np.sqrt(variance)
            continue
        if not np.isfinite(state):
            state, variance = z[t], params.initial_sd ** 2
            continue
        variance += params.daily_drift_sd ** 2
        if t > 0 and year[t] != year[t - 1]:
            variance += params.season_jump_sd ** 2
        prior_state[t], prior_sd[t] = state, np.sqrt(variance)
        innovation = z[t] - state
        if innovation < 0:
            noise = params.below_sd ** 2
        elif weight[t] > 0:
            noise = params.above_sd ** 2 / weight[t]
        else:
            continue
        gain = variance / (variance + noise)
        state += gain * innovation
        variance *= 1 - gain
    out = pd.DataFrame({"state": prior_state, "state_sd": prior_sd}, index=frame.index)
    out["floor"] = frame["world"] * np.exp(out["state"]) - frame["export_deductions"]
    return out


# ----------------------------------------------------------------------------- all candidates
def build_candidates(inputs: Inputs, grain_class: str, settings: dict,
                     kalman: KalmanSettings | None = None) -> pd.DataFrame:
    """Daily frame with the three candidate floors (floor_b0, floor_b1, floor_b2), their ceilings
    (floor + cost width) and the published export pace used by B2."""
    kalman = kalman or KalmanSettings()
    frame = parity_frame(inputs, grain_class, settings)
    frame["export_pace"] = published_pace(inputs.weekly, grain_class, "exports", frame["date"])
    frame["floor_basis_b1"] = season_rule_floor_basis(frame, inputs.weekly, grain_class, 0.10)
    frame["floor_b1"] = frame["world"] * np.exp(frame["floor_basis_b1"])
    filtered = kalman_floor(frame, frame["export_pace"], kalman)
    frame["state_b2"], frame["state_sd_b2"], frame["floor_b2"] = filtered["state"], filtered["state_sd"], filtered["floor"]
    for name in ("b0", "b1", "b2"):
        frame[f"ceiling_{name}"] = frame[f"floor_{name}"] + frame["cost_width"]
        frame[f"position_{name}"] = band.band_position(frame["safex"], frame[f"floor_{name}"], frame[f"ceiling_{name}"])
    return frame


# ----------------------------------------------------------------------------- criteria
def strong_export_weeks(weekly: pd.DataFrame, grain_class: str, min_kt: float) -> pd.DatetimeIndex:
    """Week-end dates when the class actually exported at least `min_kt` (ex post, evaluation only)."""
    rows = weekly[(weekly.grain_class == grain_class) & (weekly.flow == "exports")]
    tons = rows.groupby("week_end")["tons_week"].sum()
    return pd.DatetimeIndex(tons[tons >= min_kt * 1e3].index)


def fixed_world_floor(frame: pd.DataFrame, name: str, anchor: pd.Series) -> pd.Series:
    """The candidate's floor re-priced at one day's world price and costs (`anchor`), so changes over
    time reflect only what the method has learnt, not CBOT or ZAR moves."""
    if name == "b2":
        return anchor["world"] * np.exp(frame["state_b2"]) - anchor["export_deductions"]
    basis = np.log(frame[f"floor_{name}"] / frame["world"])
    return anchor["world"] * np.exp(basis)


def band_criteria(frame: pd.DataFrame, weekly: pd.DataFrame, grain_class: str, name: str,
                  start: str, end: str, strong_kt: float = 30.0, settle_usd: float = 10.0) -> dict:
    """C1 below-floor share and longest spell; C2 calibration in strong export weeks (USD/t);
    C3 median weeks to settle within `settle_usd` of the 31 October floor (world price held fixed)."""
    window = frame[(frame["date"] >= start) & (frame["date"] <= end)].dropna(subset=[f"floor_{name}"])
    distance_usd = (window["safex"] - window[f"floor_{name}"]) / window["usdzar"]
    below = band.outside_band_summary(window[f"position_{name}"])
    week_end = window["date"] + pd.to_timedelta((4 - window["date"].dt.weekday) % 7, unit="D")
    strong = week_end.isin(strong_export_weeks(weekly, grain_class, strong_kt))
    rows = weekly[(weekly.grain_class == grain_class) & weekly.flow.isin(["exports", "imports"])]
    season_flows = rows.pivot_table(index="season", columns="flow", values="tons_week", aggfunc="sum")
    settle_weeks = []
    for year in sorted(marketing_year(window["date"]).unique()):
        season = f"{year}/{str(year + 1)[-2:]}"
        if season not in season_flows.index or season_flows.loc[season].get("exports", 0) <= season_flows.loc[season].get("imports", 0):
            continue
        part = window[(window["date"] >= f"{year}-05-01") & (window["date"] <= f"{year}-10-31")]
        if len(part) < 100:
            continue
        anchor = part.iloc[-1]
        repriced = fixed_world_floor(part, name, anchor)
        off = ((repriced - repriced.iloc[-1]).abs() / anchor["usdzar"] > settle_usd).to_numpy()
        last_off = np.flatnonzero(off)
        settled_at = part["date"].iloc[last_off[-1] + 1] if len(last_off) else part["date"].iloc[0]
        settle_weeks.append((settled_at - pd.Timestamp(f"{year}-05-01")).days / 7)
    return {"band": name, "grain_class": grain_class, "days": int(len(window)),
            "share_below_floor": below["share_below"], "longest_below_days": below["longest_spell_below"],
            "strong_export_days": int(strong.sum()),
            "median_abs_distance_usd": float(distance_usd[strong].abs().median()) if strong.any() else np.nan,
            "median_distance_usd": float(distance_usd[strong].median()) if strong.any() else np.nan,
            "export_seasons_scored": len(settle_weeks),
            "median_weeks_to_settle": float(np.median(settle_weeks)) if settle_weeks else np.nan}
