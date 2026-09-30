"""Band fair value: where SAFEX should sit inside the import/export parity band, given stocks.

The model chosen in research/band_position (round 2, Addendum 1), ported here as the app's version.
That folder stays the frozen study record.

    band       export edge = world x exp(5th pct of prior basis)                basis = log(SAFEX / world)
               import edge = export edge + 95th pct of prior excess x SAGIS cost width
    position   (SAFEX - export edge) / (import edge - export edge)       0 = export parity, 1 = import
    STU        weekly-nowcast closing stock / trailing-12m domestic utilisation
    model      position = a + season (2 harmonics) + b x STU, refit weekly on prior weeks only
    fair value R/t = export edge + fair position x band width

SAFEX is the 90-day constant-maturity price and world is CBOT x USD/ZAR at 10:00 UTC. Everything on
a day uses only data published by that day. The fit is descriptive: out-of-sample R^2 ~0.2, but the gap
did not predict 5-40-day moves (research/band_position/REPORT.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import fairvalue as FV
from app.data import features as F

EXPORT_QUANTILE, IMPORT_QUANTILE = 0.05, 0.95
MIN_HISTORY_DAYS = 500
WORLD_TOLERANCE = pd.Timedelta(days=4)
DAYS_PER_MONTH = 30.44
HARMONICS = 2
MIN_SEASONS = 5


# ----------------------------------------------------------------------------- band
def band_frame(cont: pd.DataFrame, snap: pd.DataFrame, parity: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Daily 90-day SAFEX price, world price, SAGIS cost width (as published) and the band edges.

    Percentiles use days strictly before t. The ceiling re-applies today's floor to every prior day,
    so both edges share one definition and need only `MIN_HISTORY_DAYS` of history.
    """
    safex = cont[cont.symbol == symbol][["trade_date", "close_cm"]].dropna()
    safex = safex.rename(columns={"trade_date": "date", "close_cm": "safex"})
    safex["date"] = safex["date"].astype("datetime64[ns]")
    world = FV.world_parity(snap).rename("world").reset_index()
    world.columns = ["world_date", "world"]
    frame = pd.merge_asof(safex.sort_values("date"), world.sort_values("world_date"), left_on="date",
                          right_on="world_date", direction="backward", tolerance=WORLD_TOLERANCE)
    sagis = parity.dropna(subset=["export_randfontein", "import_randfontein"]).copy()
    sagis["cost_width"] = sagis["import_randfontein"] - sagis["export_randfontein"]
    sagis["available_date"] = sagis["available_date"].astype("datetime64[ns]")
    frame = pd.merge_asof(frame, sagis[["available_date", "cost_width"]].sort_values("available_date"),
                          left_on="date", right_on="available_date", direction="backward")
    frame = frame.dropna(subset=["world"]).drop(columns=["world_date", "available_date"]).reset_index(drop=True)
    frame["basis"] = np.log(frame["safex"] / frame["world"])

    price, world_v = frame["safex"].to_numpy(), frame["world"].to_numpy()
    basis, width = frame["basis"].to_numpy(), frame["cost_width"].to_numpy()
    floor_basis, excess_q = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for t in range(MIN_HISTORY_DAYS, len(frame)):
        floor_basis[t] = np.quantile(basis[:t], EXPORT_QUANTILE)
        excess = (price[:t] - world_v[:t] * np.exp(floor_basis[t])) / width[:t]
        if np.isfinite(excess).sum() >= MIN_HISTORY_DAYS:
            excess_q[t] = np.nanquantile(excess, IMPORT_QUANTILE)
    frame["export_edge"] = frame["world"] * np.exp(floor_basis)
    frame["import_edge"] = frame["export_edge"] + excess_q * frame["cost_width"]
    frame["position"] = (frame["safex"] - frame["export_edge"]) / (frame["import_edge"] - frame["export_edge"])
    return frame


# ----------------------------------------------------------------------------- stocks-to-use nowcast
def _weekly_flows(weekly: pd.DataFrame, grain_class: str) -> pd.DataFrame:
    """Deliveries, imports and exports per week end, dated by when the week's trade was published."""
    rows = weekly[(weekly.grain_class == grain_class) & weekly.flow.isin(["deliveries", "imports", "exports"])]
    flows = rows.pivot_table(index="week_end", columns="flow", values="tons_week", aggfunc="sum")
    flows = flows.reindex(columns=["deliveries", "imports", "exports"])
    available = rows.pivot_table(index="week_end", columns="flow", values="available_date", aggfunc="max")
    flows["available_date"] = available.get("exports")
    flows["has_deliveries"] = flows["deliveries"].notna()
    return flows.dropna(subset=["available_date"]).sort_index()


def stocks_to_use_daily(sd: pd.DataFrame, weekly: pd.DataFrame, grain_class: str, dates: pd.Series) -> pd.DataFrame:
    """Daily stocks-to-use, recomputed whenever a monthly release or a week of trade is published.

    Nowcast: latest closing stock + weekly deliveries + imports - exports for weeks after the reported
    month (published by the day) - trailing utilisation pro-rated by weeks. NaN if trade was published
    without deliveries. Monthly figures enter on their release date.
    """
    releases = sd[sd.grain_class == grain_class].sort_values(["vintage_date", "latest_month"])
    releases = releases.drop_duplicates("vintage_date", keep="last").reset_index(drop=True)
    releases["vintage_date"] = releases["vintage_date"].astype("datetime64[ns]")
    releases["month_end"] = releases["latest_month"] + pd.offsets.MonthEnd(0)
    flows = _weekly_flows(weekly, grain_class)
    events = np.union1d(releases["vintage_date"].to_numpy(), flows["available_date"].astype("datetime64[ns]").to_numpy())
    release_dates = releases["vintage_date"].to_numpy()
    rows = []
    for event in pd.DatetimeIndex(events):
        i = np.searchsorted(release_dates, event.to_datetime64(), side="right") - 1
        if i < 0:
            continue
        r = releases.iloc[i]
        weeks = flows[(flows.index > r["month_end"]) & (flows["available_date"] <= event)]
        if weeks.empty:
            stock = r["closing_stock"]
        elif not weeks["has_deliveries"].all():
            stock = np.nan
        else:
            use = r["utilisation_12m"] / 12 * len(weeks) * 7 / DAYS_PER_MONTH
            stock = (r["closing_stock"] + weeks["deliveries"].sum() + weeks["imports"].fillna(0).sum()
                     - weeks["exports"].fillna(0).sum() - use)
        rows.append({"date": event, "vintage_date": r["vintage_date"], "latest_month": r["latest_month"],
                     "stu_total": r["closing_stock"] / r["disappearance_12m"],
                     "stu_domestic": r["closing_stock"] / r["utilisation_12m"],
                     "stu_domestic_nowcast": stock / r["utilisation_12m"], "nowcast_weeks": len(weeks),
                     "last_week_published": weeks.index.max() if len(weeks) else pd.NaT})
    table = pd.DataFrame(rows)
    table["date"] = table["date"].astype("datetime64[ns]")
    left = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]")})
    return pd.merge_asof(left.sort_values("date"), table.sort_values("date"), on="date", direction="backward")


# ----------------------------------------------------------------------------- model
def season_terms(dates: pd.Series) -> np.ndarray:
    """Fourier terms on the fraction of the marketing year (from 1 May) elapsed."""
    dates = pd.to_datetime(dates)
    start = pd.to_datetime((dates.dt.year - (dates.dt.month < 5).astype(int)).astype(str) + "-05-01")
    fraction = ((dates - start).dt.days / 365.25).to_numpy()
    cols = []
    for k in range(1, HARMONICS + 1):
        cols += [np.sin(2 * np.pi * k * fraction), np.cos(2 * np.pi * k * fraction)]
    return np.column_stack(cols)


def design(frame: pd.DataFrame) -> np.ndarray:
    """Constant, season terms and the primary stocks-to-use ratio."""
    return np.column_stack([np.ones(len(frame)), season_terms(frame["date"]), frame["stu_domestic_nowcast"].to_numpy()])


def first_fit_date(band_start: pd.Timestamp) -> pd.Timestamp:
    """The first 1 May preceded by `MIN_SEASONS` complete marketing years of band history."""
    year = band_start.year - (band_start.month < 5)
    first_full = pd.Timestamp(f"{year}-05-01")
    if first_full < band_start:
        first_full += pd.DateOffset(years=1)
    return first_full + pd.DateOffset(years=MIN_SEASONS)


def fair_value_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample fair position for every day from the first fit date.

    Weekly snapshots (the last trading day of each week) are the fitting sample, so slow-moving stocks
    are not counted five times. The model used in week t is fitted on weeks before t, and applied to
    every day of week t. Also returns the expanding-mean benchmark and the fitted slope.
    """
    usable = frame["position"].notna() & (frame["stu_domestic_nowcast"] > 0)
    week = frame["date"].dt.to_period("W-FRI")
    snaps = frame[frame["position"].notna()].groupby(week[frame["position"].notna()]).tail(1)
    snaps = snaps[snaps["stu_domestic_nowcast"] > 0].reset_index(drop=True)
    start = first_fit_date(frame.loc[frame["position"].notna(), "date"].min())
    fair, naive, slope = (np.full(len(frame), np.nan) for _ in range(3))
    snap_weeks = snaps["date"].dt.to_period("W-FRI")
    for t in np.flatnonzero(snaps["date"] >= start):
        train = snaps.iloc[:t]
        beta, *_ = np.linalg.lstsq(design(train), train["position"].to_numpy(), rcond=None)
        rows = np.flatnonzero((week == snap_weeks.iloc[t]).to_numpy() & usable.to_numpy())
        fair[rows] = design(frame.iloc[rows]) @ beta
        naive[rows] = train["position"].mean()
        slope[rows] = beta[-1]
    out = frame.copy()
    out["fair_position"], out["naive_position"], out["stu_slope"] = fair, naive, slope
    out["band_width"] = out["import_edge"] - out["export_edge"]
    out["fair_value"] = out["export_edge"] + out["fair_position"] * out["band_width"]
    out["gap_rand"] = out["safex"] - out["fair_value"]
    out["gap_position"] = out["position"] - out["fair_position"]
    return out


def build(cont: pd.DataFrame, snap: pd.DataFrame, parity: pd.DataFrame, sd: pd.DataFrame, weekly: pd.DataFrame,
          grain_class: str, symbol: str) -> pd.DataFrame:
    """Full daily frame for one class: band, stocks-to-use nowcast and out-of-sample fair value."""
    frame = band_frame(cont, snap, parity, symbol)
    frame = frame.merge(stocks_to_use_daily(sd, weekly, grain_class, frame["date"]), on="date", how="left")
    return fair_value_daily(frame)


def out_of_sample_scores(daily: pd.DataFrame, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Weekly out-of-sample R^2 against the expanding mean, and mean absolute gaps, per window."""
    week = daily["date"].dt.to_period("W-FRI")
    snaps = daily.groupby(week).tail(1).dropna(subset=["fair_position", "position", "naive_position"])
    rows = []
    for label, (start, end) in windows.items():
        part = snaps[(snaps["date"] >= start) & (snaps["date"] <= end)]
        if len(part) < 10:
            continue
        sse = ((part["position"] - part["fair_position"]) ** 2).sum()
        sse_naive = ((part["position"] - part["naive_position"]) ** 2).sum()
        rows.append({"window": label, "weeks": len(part), "oos_r2": 1 - sse / sse_naive,
                     "mean_abs_gap_rand": part["gap_rand"].abs().mean()})
    return pd.DataFrame(rows)
