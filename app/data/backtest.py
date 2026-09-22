"""Backtest engine for analogue events, with the overfitting controls built in.

An interactive event builder wired to a backtest is structurally a machine for producing
false positives: try forty configurations and one will look excellent on twenty episodes.
So the statistics here are not optional extras -

  * every configuration run in a session is counted as a trial;
  * the Sharpe haircut (Bailey & Lopez de Prado) subtracts the expected maximum Sharpe
    under the null given that trial count;
  * the Deflated Sharpe Ratio converts the observed Sharpe into a probability that it is
    genuinely above zero, accounting for trials, sample length, skew and kurtosis;
  * a holdout period is withheld and scored once.

Costs are applied in R/t at the entry price, because in a thin market they decide the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


# ----------------------------------------------------------------------------- costs
@dataclass(frozen=True)
class CostModel:
    """Round-trip cost in rand per tonne. Defaults are indicative SAFEX grain levels.

    SAFEX maize marks once a day with a thin book, so the half-spread dominates. These are
    assumptions, not measured fills: change them and watch the conclusion move.
    """
    half_spread_r_t: float = 3.0      # half the quoted bid/ask, paid on entry and on exit
    brokerage_r_t: float = 0.30       # ~R30 per 100 t contract per side
    slippage_r_t: float = 2.0         # market impact beyond the touch

    @property
    def one_way_r_t(self) -> float:
        return self.half_spread_r_t + self.brokerage_r_t + self.slippage_r_t

    @property
    def round_trip_r_t(self) -> float:
        return 2 * self.one_way_r_t

    def log_drag(self, entry_price: float) -> float:
        """Round-trip cost as a log-return drag at this price level."""
        if not entry_price or entry_price <= 0:
            return 0.0
        return float(np.log1p(self.round_trip_r_t / entry_price))


# ----------------------------------------------------------------------------- engine
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.Series                  # equal-weight portfolio log return per day
    stats: dict
    config: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.daily.cumsum().apply(np.exp)


def _exit_index(path: np.ndarray, direction: int, horizon: int,
                stop: float | None, target: float | None) -> int:
    """First index at which the trade closes: a stop or target hit, else the horizon."""
    signed = direction * path
    for i in range(1, min(horizon, len(path) - 1) + 1):
        if stop is not None and signed[i] <= -abs(stop):
            return i
        if target is not None and signed[i] >= abs(target):
            return i
    return min(horizon, len(path) - 1)


def run_backtest(idx: pd.Series, entries: pd.DataFrame, horizon: int = 10,
                 costs: CostModel | None = None, stop: float | None = None,
                 target: float | None = None, size_by_z: bool = False,
                 entry_lag_days: int = 1) -> BacktestResult:
    """Fade the signal: short when rich (z > 0), long when cheap.

    `idx` is the roll-adjusted price index. `entries` needs columns vintage_date and z, and
    optionally front_close for the cost calculation. Trades may overlap; the portfolio holds
    every open trade at equal weight.
    """
    costs = costs or CostModel()
    idx = idx.sort_index()
    v = np.log(idx.to_numpy())
    dates = idx.index

    ent = entries.dropna(subset=["z"]).sort_values("vintage_date").reset_index(drop=True)
    pos = dates.searchsorted(pd.DatetimeIndex(ent.vintage_date).as_unit("ns")
                             + pd.Timedelta(days=entry_lag_days))

    rows = []
    for k, p0 in enumerate(pos):
        if p0 >= len(v) - 1:
            continue
        z = float(ent.z.iloc[k])
        direction = -1 if z > 0 else 1                      # fade the dislocation
        path = v[p0:min(p0 + horizon + 1, len(v))] - v[p0]
        h = _exit_index(path, direction, horizon, stop, target)
        if h < 1:
            continue
        px = float(ent.front_close.iloc[k]) if "front_close" in ent and pd.notna(
            ent.front_close.iloc[k]) else float(idx.iloc[p0])
        gross = direction * float(path[h])
        drag = costs.log_drag(px)
        size = min(abs(z), 3.0) if size_by_z else 1.0
        rows.append({
            "entry_date": dates[p0], "exit_date": dates[p0 + h], "held_days": h,
            "direction": "short" if direction < 0 else "long", "dir": direction,
            "z": z, "size": size, "entry_px": px,
            "gross_ret": gross, "cost": drag, "net_ret": (gross - drag) * size,
            "i0": p0, "i1": p0 + h,
        })
    trades = pd.DataFrame(rows)
    daily = _portfolio_daily(trades, v, dates, costs)
    return BacktestResult(trades, daily, summarise(trades, daily),
                          {"horizon": horizon, "stop": stop, "target": target,
                           "size_by_z": size_by_z, "cost_rt_r_t": costs.round_trip_r_t})


def _portfolio_daily(trades: pd.DataFrame, v: np.ndarray, dates: pd.DatetimeIndex,
                     costs: CostModel) -> pd.Series:
    """Equal weight across whatever is open that day; costs charged on the exit day."""
    out = np.zeros(len(v))
    n_open = np.zeros(len(v))
    if trades.empty:
        return pd.Series(out, index=dates, name="ret")
    for t in trades.itertuples():
        step = np.diff(v[t.i0:t.i1 + 1]) * t.dir * t.size
        out[t.i0 + 1:t.i1 + 1] += step
        n_open[t.i0 + 1:t.i1 + 1] += t.size
        out[t.i1] -= t.cost * t.size                        # round trip charged at exit
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(n_open > 0, out / np.maximum(n_open, 1e-9), 0.0)
    return pd.Series(r, index=dates, name="ret")


def max_drawdown(daily: pd.Series) -> float:
    eq = daily.cumsum()
    return float((eq - eq.cummax()).min())


def summarise(trades: pd.DataFrame, daily: pd.Series) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_episodes": 0, "hit_rate": np.nan, "avg_net": np.nan,
                "total_ret": 0.0, "sharpe": np.nan, "sharpe_daily": np.nan,
                "max_dd": 0.0, "time_in_market": 0.0, "avg_hold": np.nan,
                "cost_share": np.nan, "skew": np.nan, "kurtosis": np.nan, "n_obs": 0}
    active = daily[daily != 0]
    sd = active.std(ddof=1)
    sr_d = float(active.mean() / sd) if sd and sd > 0 else np.nan
    gross = trades.gross_ret.abs().mean()
    return {
        "n_trades": int(len(trades)),
        "n_episodes": int(_episode_count(trades)),
        "hit_rate": float((trades.net_ret > 0).mean()),
        "avg_net": float(trades.net_ret.mean()),
        "total_ret": float(daily.sum()),
        "sharpe": float(sr_d * np.sqrt(TRADING_DAYS)) if sr_d == sr_d else np.nan,
        "sharpe_daily": sr_d,
        "max_dd": max_drawdown(daily),
        "time_in_market": float((daily != 0).mean()),
        "avg_hold": float(trades.held_days.mean()),
        "cost_share": float(trades.cost.mean() / gross) if gross else np.nan,
        "skew": float(active.skew()) if len(active) > 3 else np.nan,
        "kurtosis": float(active.kurtosis() + 3) if len(active) > 3 else np.nan,
        "n_obs": int(len(active)),
    }


def _episode_count(trades: pd.DataFrame) -> int:
    """Consecutive monthly entries in the same direction are one idea, not many."""
    if trades.empty:
        return 0
    d = trades.sort_values("entry_date")
    gap = d.entry_date.diff().dt.days.fillna(999) > 45
    flip = d.dir.ne(d.dir.shift())
    return int((gap | flip).cumsum().nunique())


# ----------------------------------------------------------------------------- overfitting
def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """E[max Sharpe] over `n_trials` independent draws from a zero-mean null."""
    if n_trials < 2 or not sharpe_std or sharpe_std <= 0:
        return 0.0
    g = np.euler_gamma
    return float(sharpe_std * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                               + g * norm.ppf(1 - 1 / (n_trials * np.e))))


def sharpe_haircut(observed: float, n_trials: int, sharpe_std: float) -> dict:
    exp_max = expected_max_sharpe(n_trials, sharpe_std)
    return {"observed": observed, "expected_max": exp_max, "haircut": observed - exp_max,
            "n_trials": n_trials}


def deflated_sharpe(sr_daily: float, n_obs: int, skew: float, kurtosis: float,
                    n_trials: int, sharpe_std: float) -> float:
    """P(true Sharpe > 0) given the trial count, sample length, skew and kurtosis.

    Bailey & Lopez de Prado (2014). All Sharpes are per observation, not annualised.
    Above ~0.95 is the usual bar for treating a result as more than selection noise.
    """
    if n_obs < 10 or sr_daily != sr_daily:
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, sharpe_std)
    skew = 0.0 if skew != skew else skew
    kurtosis = 3.0 if kurtosis != kurtosis else kurtosis
    denom = 1 - skew * sr_daily + (kurtosis - 1) / 4 * sr_daily ** 2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr_daily - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)))


def pbo(is_sharpes: np.ndarray, oos_sharpes: np.ndarray) -> float:
    """Probability of backtest overfitting: how often the in-sample winner lands below the
    out-of-sample median. Above 0.5 means the selection procedure is worse than a coin."""
    is_s, oos_s = np.asarray(is_sharpes, float), np.asarray(oos_sharpes, float)
    if is_s.ndim != 2 or is_s.shape != oos_s.shape or is_s.shape[1] < 2:
        return float("nan")
    ranks = []
    for row_is, row_oos in zip(is_s, oos_s):
        if np.all(np.isnan(row_is)) or np.all(np.isnan(row_oos)):
            continue
        best = int(np.nanargmax(row_is))
        order = np.argsort(np.argsort(-np.nan_to_num(row_oos, nan=-np.inf)))
        ranks.append(order[best] / (len(row_oos) - 1))
    return float(np.mean(np.array(ranks) > 0.5)) if ranks else float("nan")
