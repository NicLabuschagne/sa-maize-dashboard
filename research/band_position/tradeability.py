"""First look at tradeability: does the gap between fair position and position predict what happens next?

    signal     fair position - position          positive = cheap against the balance sheet
    benchmark  expanding mean position - position   plain mean reversion, no balance sheet
    outcomes   change in band position, and the roll-adjusted SAFEX front-month log return,
               from the close after the signal (d + 1) to h trading days later

Forward windows overlap on daily data, so t-stats use Newey-West with lags = h - 1, and the
effective number of independent observations is reported as n / h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from app.data import fairvalue as FV
from app.data import features as F
from research.band_position import models


def daily_fair_position(daily: pd.DataFrame, stu_column: str, name: str, first_date: pd.Timestamp,
                        harmonics: int) -> pd.DataFrame:
    """Apply each week's model to every trading day of that week.

    The model for week t is fitted on weekly snapshots before t, i.e. data up to the previous week's
    last close, so every day in week t uses only earlier information. On the snapshot day itself this
    equals `models.expanding_predictions`. Also returns the rand fair value implied by the band.
    """
    snapshots = models.weekly_snapshots(daily.dropna(subset=["position"]))
    snapshots = snapshots.dropna(subset=[stu_column])
    snapshots = snapshots[snapshots[stu_column] > 0].reset_index(drop=True)
    week_of_day = daily["date"].dt.to_period("W-FRI")
    week_of_snapshot = snapshots["date"].dt.to_period("W-FRI")
    fair, naive = np.full(len(daily), np.nan), np.full(len(daily), np.nan)
    usable = daily["position"].notna() & (daily[stu_column] > 0)
    for t in np.flatnonzero(snapshots["date"] >= first_date):
        train = snapshots.iloc[:t]
        model = models.fit_model(name, train, stu_column, harmonics)
        rows = np.flatnonzero((week_of_day == week_of_snapshot.iloc[t]).to_numpy() & usable.to_numpy())
        fair[rows] = model.predict(daily.iloc[rows])
        naive[rows] = train["position"].mean()
    out = daily[["date", "safex", "position", "hybrid_export", "hybrid_import"]].copy()
    out["fair_position"] = fair
    out["naive_position"] = naive
    out["fair_value"] = out["hybrid_export"] + out["fair_position"] * (out["hybrid_import"] - out["hybrid_export"])
    out["signal"] = out["fair_position"] - out["position"]
    out["benchmark_signal"] = out["naive_position"] - out["position"]
    return out


def forward_outcomes(frame: pd.DataFrame, prices: pd.DataFrame, symbol: str, horizons: list[int],
                     entry_lag: int) -> pd.DataFrame:
    """Forward change in position and forward SAFEX return, from `entry_lag` trading days after each
    date to `entry_lag + h`. Positions count trading days in each series, never calendar days."""
    out = frame.copy()
    position = out["position"].to_numpy()
    index = FV.roll_adjusted_index(F.continuous(prices), symbol)
    log_index = np.log(index.to_numpy())
    entry_in_index = index.index.searchsorted(pd.DatetimeIndex(out["date"]).as_unit("ns")) + entry_lag
    for h in horizons:
        start, end = np.arange(len(out)) + entry_lag, np.arange(len(out)) + entry_lag + h
        ok = end < len(out)
        change = np.full(len(out), np.nan)
        change[ok] = position[end[ok]] - position[start[ok]]
        out[f"fwd_position_{h}"] = change
        ok = entry_in_index + h < len(log_index)
        ret = np.full(len(out), np.nan)
        ret[ok] = log_index[entry_in_index[ok] + h] - log_index[entry_in_index[ok]]
        out[f"fwd_return_{h}"] = ret
    return out


def ic_statistics(signal: pd.Series, outcome: pd.Series, horizon: int) -> dict:
    """Spearman IC with a Newey-West t-stat.

    The t-stat comes from regressing the outcome's ranks on the signal's ranks (both scaled to unit
    variance), so its slope is the rank IC. Lags = h - 1 because h-day windows overlap by h - 1 days.
    """
    data = pd.DataFrame({"signal": signal, "outcome": outcome}).dropna()
    n = len(data)
    if n < 60:
        return {"n": n, "effective_n": n / horizon, "ic": np.nan, "nw_t": np.nan}
    x = stats.rankdata(data["signal"])
    y = stats.rankdata(data["outcome"])
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 1)})
    return {"n": n, "effective_n": n / horizon, "ic": float(stats.spearmanr(data["signal"], data["outcome"]).statistic),
            "nw_t": float(fit.tvalues[1])}


def ic_table(frame: pd.DataFrame, horizons: list[int], windows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """IC of the fair-value signal and the benchmark, for both outcomes, every horizon and window."""
    rows = []
    for label, start, end in windows:
        part = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        for signal in ("signal", "benchmark_signal"):
            for outcome in ("position", "return"):
                for h in horizons:
                    rows.append({"window": label, "signal": signal, "outcome": outcome, "horizon": h,
                                 **ic_statistics(part[signal], part[f"fwd_{outcome}_{h}"], h)})
    return pd.DataFrame(rows)


def ic_by_year(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """IC per calendar year at one horizon, so recent performance can be read year by year."""
    rows = []
    for year, part in frame.dropna(subset=["signal"]).groupby(frame["date"].dt.year):
        for outcome in ("position", "return"):
            rows.append({"year": int(year), "outcome": outcome, "horizon": horizon,
                         **ic_statistics(part["signal"], part[f"fwd_{outcome}_{horizon}"], horizon)})
    return pd.DataFrame(rows)
