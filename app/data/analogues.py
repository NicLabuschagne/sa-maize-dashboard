"""Analogue analysis: condition on a state known at the SAGIS release date, then look at what the
front-month price did afterwards — as paths (fan chart) and as a distribution at one horizon,
compared with the unconditional distribution over the same release calendar.

Consecutive releases in the same state are one *episode*; both counts are reported because
overlapping forward windows make n_obs an overstatement of independent evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PATH_DAYS = 40
HORIZON_CHOICES = {"5d": 5, "10d": 10, "1m": 21, "2m": 42, "3m": 63}
MIN_PRIOR_YEARS = 5


# ----------------------------------------------------------------------------- state features
def pct_rank_same_month(p: pd.DataFrame, col: str = "months_cover", min_prior: int = MIN_PRIOR_YEARS) -> pd.Series:
    """Point-in-time percentile of `col` against prior years' values for the same marketing-year month."""
    out = np.full(len(p), np.nan)
    vals, mm = p[col].to_numpy(), p["my_month"].to_numpy()
    for i in range(len(p)):
        prior = vals[:i][mm[:i] == mm[i]]
        prior = prior[~np.isnan(prior)]
        if len(prior) >= min_prior:
            out[i] = (prior < vals[i]).mean()
    return pd.Series(out, index=p.index, name=f"{col}_pct_same_month")


def yoy_log_change(p: pd.DataFrame, col: str = "months_cover") -> pd.Series:
    """log(value) − log(value at the same marketing-year month one season earlier), as published."""
    s = p.set_index("latest_month")[col]
    prev = s.reindex(s.index - pd.DateOffset(years=1))
    return pd.Series(np.log(s.to_numpy()) - np.log(prev.to_numpy()), index=p.index, name=f"{col}_yoy")


def episodes(mask: pd.Series) -> pd.Series:
    """Label runs of consecutive True with an episode id (NaN where False)."""
    m = mask.fillna(False).astype(bool)
    starts = m & ~m.shift(fill_value=False)
    ids = starts.cumsum().where(m)
    return ids


# ----------------------------------------------------------------------------- paths
def forward_paths(idx: pd.Series, dates: pd.Series, n_days: int = PATH_DAYS, lag_days: int = 1) -> pd.DataFrame:
    """Rows = entry dates, cols = 0..n_days; cumulative log return from the entry close (day 0 = 0)."""
    entry = pd.DatetimeIndex(dates).as_unit("ns") + pd.Timedelta(days=lag_days)
    pos = idx.index.searchsorted(entry)
    vals = np.log(idx.to_numpy())
    rows = []
    for p0 in pos:
        end = p0 + n_days + 1
        if end <= len(vals):
            rows.append(vals[p0:end] - vals[p0])
        else:
            rows.append(np.full(n_days + 1, np.nan))
    return pd.DataFrame(rows, index=dates.to_numpy(), columns=range(n_days + 1))


def fan(paths: pd.DataFrame, qs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    return paths.quantile(list(qs)).T.rename(columns={q: f"q{int(q * 100)}" for q in qs})


def horizon_stats(cond: pd.Series, uncond: pd.Series, n_episodes: int) -> dict:
    """Summary of terminal log returns, conditional vs unconditional, plus a KS test of the two samples."""
    c, u = cond.dropna(), uncond.dropna()
    ks = stats.ks_2samp(c, u) if len(c) >= 5 and len(u) >= 5 else None
    return {
        "n_obs": int(len(c)), "n_episodes": int(n_episodes), "n_uncond": int(len(u)),
        "cond_mean": float(c.mean()), "cond_median": float(c.median()), "cond_p_neg": float((c < 0).mean()),
        "cond_p10": float(c.quantile(0.1)), "cond_p90": float(c.quantile(0.9)),
        "uncond_mean": float(u.mean()), "uncond_median": float(u.median()), "uncond_p_neg": float((u < 0).mean()),
        "uncond_p10": float(u.quantile(0.1)), "uncond_p90": float(u.quantile(0.9)),
        "ks_stat": float(ks.statistic) if ks else np.nan, "ks_p": float(ks.pvalue) if ks else np.nan,
    }


# ----------------------------------------------------------------------------- event builder
def build_mask(p: pd.DataFrame, event: str, threshold: float, direction: str, months: list[int] | None) -> pd.Series:
    """event in {'z', 'cover_pct', 'cover_yoy', 'spread_z'}; direction in {'high', 'low'}."""
    col = {"z": "z", "cover_pct": "months_cover_pct_same_month", "cover_yoy": "months_cover_yoy",
           "spread_z": "spread_z"}[event]
    s = p[col]
    m = (s >= threshold) if direction == "high" else (s <= threshold)
    if months:
        m &= p["my_month"].isin(months)
    return m & s.notna()


def seasonal_surprise(p: pd.DataFrame, col: str = "months_cover", min_prior: int = 4) -> pd.Series:
    """Divergence of a series from what is normal for that point in the marketing year.

    log(value) minus the mean of log(value) at the same marketing-year month in *prior* seasons
    only, so it is point-in-time. This is the closest stand-in the warehouse supports for the
    "my balance sheet vs the official estimate" state: it measures how far the published figure
    sits from its own seasonal expectation, rather than how the price is valued.
    """
    out = np.full(len(p), np.nan)
    vals, mm = np.log(p[col].to_numpy(dtype=float)), p["my_month"].to_numpy()
    for i in range(len(p)):
        prior = vals[:i][mm[:i] == mm[i]]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= min_prior and np.isfinite(vals[i]):
            out[i] = vals[i] - prior.mean()
    return pd.Series(out, index=p.index, name=f"{col}_surprise")


COLUMNS = {
    "z": ("Fair-value z — price vs stocks (model A)", "valuation"),
    "spread_z": ("Calendar-spread z (model B)", "valuation"),
    "basis_z": ("Parity-basis z (model D)", "valuation"),
    "months_cover_pct_same_month": ("Cover percentile vs prior seasons, same month", "divergence"),
    "months_cover_yoy": ("Cover, log change vs same month last season", "divergence"),
    "months_cover_surprise": ("Cover surprise vs seasonal norm", "divergence"),
}


def build_mask_multi(p: pd.DataFrame, conditions: list[tuple[str, str, float]],
                     months: list[int] | None = None) -> pd.Series:
    """AND together several (column, direction, threshold) conditions.

    A compound state - "rich on stocks AND the balance sheet is tighter than its seasonal norm" -
    is a different and far more specific trade than either leg alone.
    """
    m = pd.Series(True, index=p.index)
    for col, direction, thr in conditions:
        s = p[col]
        m &= (s >= thr) if direction == "high" else (s <= thr)
        m &= s.notna()
    if months:
        m &= p["my_month"].isin(months)
    return m
