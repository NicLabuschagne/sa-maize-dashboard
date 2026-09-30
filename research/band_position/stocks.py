"""Point-in-time stocks-to-use, daily, with the SAGIS weekly nowcast between monthly releases.

    stu_total             closing stock / trailing-12m (utilisation + exports)   the dashboard's ratio
    stu_domestic          closing stock / trailing-12m utilisation               no export feedback
    stu_domestic_nowcast  nowcast stock / trailing-12m utilisation                primary

The nowcast is the W1 rule from PREREGISTRATION_WEEKLY.md: start from the latest release's closing
stock, add deliveries and imports and subtract exports for weeks ending after the reported month, and
subtract utilisation pro-rated by weeks (SAGIS publishes use only monthly). The week set is those
whose exports were published by the day, because trade publishes later than deliveries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import features as F

GRAIN_CLASSES = ("white", "yellow")


def monthly_stocks(balance_sheet: pd.DataFrame) -> pd.DataFrame:
    """One row per (release, class) with closing stock and trailing-12m use, from the dashboard's
    `sd_monthly`. Where one release date carries two months, the later month wins."""
    monthly = F.sd_monthly(balance_sheet)
    monthly = monthly[monthly.grain_class.isin(GRAIN_CLASSES)].copy()
    monthly["vintage_date"] = monthly["vintage_date"].astype("datetime64[ns]")
    monthly["month_end"] = monthly["latest_month"] + pd.offsets.MonthEnd(0)
    monthly = monthly.sort_values(["grain_class", "vintage_date", "latest_month"])
    return monthly.drop_duplicates(["grain_class", "vintage_date"], keep="last").reset_index(drop=True)


def _weekly_flows(weekly: pd.DataFrame, grain_class: str) -> pd.DataFrame:
    """Weekly deliveries, imports and exports for one class, one row per week end, with the date the
    week's trade (the slower series) became available."""
    rows = weekly[(weekly.grain_class == grain_class) & weekly.flow.isin(["deliveries", "imports", "exports"])]
    flows = rows.pivot_table(index="week_end", columns="flow", values="tons_week", aggfunc="sum")
    available = rows.pivot_table(index="week_end", columns="flow", values="available_date", aggfunc="max")
    flows = flows.reindex(columns=["deliveries", "imports", "exports"])
    flows["available_date"] = available.get("exports")
    flows["has_deliveries"] = flows["deliveries"].notna()
    return flows.dropna(subset=["available_date"]).sort_index()


def nowcast_stock(release: pd.Series, flows: pd.DataFrame, as_of: pd.Timestamp, until: pd.Timestamp | None,
                  days_per_month: float) -> tuple[float, int]:
    """Stock after the weeks following `release`'s reported month that were published by `as_of`
    (and, if given, ended by `until`). Returns (stock, number of weeks used); NaN if trade was
    published without deliveries, which would bias the stock upward."""
    weeks = flows[(flows.index > release["month_end"]) & (flows["available_date"] <= as_of)]
    if until is not None:
        weeks = weeks[weeks.index <= until]
    if weeks.empty:
        return float(release["closing_stock"]), 0
    if not weeks["has_deliveries"].all():
        return np.nan, len(weeks)
    use = release["utilisation_12m"] / 12 * len(weeks) * 7 / days_per_month
    stock = (release["closing_stock"] + weeks["deliveries"].sum() + weeks["imports"].fillna(0).sum()
             - weeks["exports"].fillna(0).sum() - use)
    return float(stock), len(weeks)


def stocks_to_use_events(monthly: pd.DataFrame, weekly: pd.DataFrame, grain_class: str,
                         days_per_month: float) -> pd.DataFrame:
    """The three stocks-to-use ratios recomputed on every date something new was published for the
    class: a monthly release or a week of trade. Values hold until the next event."""
    releases = monthly[monthly.grain_class == grain_class].reset_index(drop=True)
    flows = _weekly_flows(weekly, grain_class)
    events = np.union1d(releases["vintage_date"].to_numpy(), flows["available_date"].to_numpy())
    release_dates = releases["vintage_date"].to_numpy()
    rows = []
    for event in pd.DatetimeIndex(events):
        position = np.searchsorted(release_dates, event.to_datetime64(), side="right") - 1
        if position < 0:
            continue
        release = releases.iloc[position]
        stock, n_weeks = nowcast_stock(release, flows, event, None, days_per_month)
        rows.append({
            "date": event, "vintage_date": release["vintage_date"], "latest_month": release["latest_month"],
            "stu_total": release["closing_stock"] / release["disappearance_12m"],
            "stu_domestic": release["closing_stock"] / release["utilisation_12m"],
            "stu_domestic_nowcast": stock / release["utilisation_12m"],
            "nowcast_weeks": n_weeks,
        })
    return pd.DataFrame(rows)


def daily_stocks_to_use(dates: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Carry each event's ratios forward onto trading days, from the event date only (no backfill)."""
    left = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]")})
    right = events.sort_values("date").copy()
    right["date"] = right["date"].astype("datetime64[ns]")
    return pd.merge_asof(left.sort_values("date"), right, on="date", direction="backward")


def release_surprises(monthly: pd.DataFrame, weekly: pd.DataFrame, grain_class: str,
                      days_per_month: float) -> pd.DataFrame:
    """For each monthly release: published closing stock minus the weekly nowcast of the same
    month-end, built from the previous release and the weeks published the day before. Scaled by
    trailing utilisation so it is on the stocks-to-use scale. Only consecutive months are used."""
    releases = monthly[monthly.grain_class == grain_class].reset_index(drop=True)
    flows = _weekly_flows(weekly, grain_class)
    rows = []
    for i in range(1, len(releases)):
        previous, current = releases.iloc[i - 1], releases.iloc[i]
        if current["latest_month"] != previous["latest_month"] + pd.offsets.MonthBegin(1):
            continue
        day_before = current["vintage_date"] - pd.Timedelta(days=1)
        expected, n_weeks = nowcast_stock(previous, flows, day_before, current["month_end"], days_per_month)
        rows.append({"vintage_date": current["vintage_date"], "grain_class": grain_class,
                     "published_stock": current["closing_stock"], "nowcast_stock": expected,
                     "nowcast_weeks": n_weeks,
                     "surprise": (current["closing_stock"] - expected) / current["utilisation_12m"]})
    return pd.DataFrame(rows)
