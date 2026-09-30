"""Edge trading test on the B2 band (Addendum 4).

The desk trades the arb (long SAFEX, short CBOT x ZAR) when SAFEX is within $10/t of export parity.
This module asks whether the balance-sheet gap tells you *which* of those moments to take.

    zone     export: (SAFEX - B2 floor) / USDZAR <= $10    import: (B2 ceiling - SAFEX) / USDZAR <= $10
    signals  S1 detrended gap, S2 raw gap, S3 unpriced fair move (positive = cheap)
    outcome  20-day arb log return from the close on d+1: SAFEX roll-adjusted front minus CBOT x ZAR
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from app.data import fairvalue as FV
from app.data import features as F
from app.data.backtest import TRADING_DAYS, CostModel, deflated_sharpe, max_drawdown
from research.band_position import revealed, stocks, tradeability
from research.band_position.inputs import Inputs

HORIZON = 20
ENTRY_LAG = 1
TRIGGER_USD = 10.0
DETREND_WINDOW, DETREND_MIN = 250, 200
CBOT_ROUND_TRIP_USD = 0.50
SIGNALS = {"S1 detrended gap": "s1", "S2 raw gap": "s2", "S3 unpriced fair move": "s3"}


# ----------------------------------------------------------------------------- frame
def arb_returns(frame: pd.DataFrame, prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Daily log returns of the tradeable legs on the frame's calendar: SAFEX roll-adjusted front,
    world price (CBOT x ZAR), their difference (the arb), and a flag on days the front contract rolls."""
    continuous = F.continuous(prices)
    index = FV.roll_adjusted_index(continuous, symbol).rename("safex_index").reset_index()
    index.columns = ["date", "safex_index"]
    expiry = continuous[continuous.symbol == symbol][["trade_date", "expiry_1"]].rename(columns={"trade_date": "date"})
    expiry["date"] = expiry["date"].astype("datetime64[ns]")
    out = pd.merge_asof(frame[["date", "world"]].sort_values("date"), index.sort_values("date"), on="date",
                        direction="backward", tolerance=pd.Timedelta(days=4))
    out = pd.merge_asof(out, expiry.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta(days=4))
    out["safex_ret"] = np.log(out["safex_index"]).diff()
    out["world_ret"] = np.log(out["world"]).diff()
    out["arb_ret"] = out["safex_ret"] - out["world_ret"]
    out["roll_day"] = out["expiry_1"].ne(out["expiry_1"].shift()) & out["expiry_1"].shift().notna()
    return out[["date", "safex_ret", "world_ret", "arb_ret", "roll_day"]]


def build_edge_frame(inputs: Inputs, grain_class: str, settings: dict, first_date: pd.Timestamp) -> pd.DataFrame:
    """Daily B2 band, fair position refit on B2 positions, gap signals, zones, arb returns and the
    forward 20-day arb return. Everything on day d uses information available at the close on d."""
    frame = revealed.build_candidates(inputs, grain_class, settings)
    monthly = stocks.monthly_stocks(inputs.balance_sheet)
    events = stocks.stocks_to_use_events(monthly, inputs.weekly, grain_class, settings["stocks"]["days_per_month"])
    frame = frame.merge(stocks.daily_stocks_to_use(frame["date"], events), on="date", how="left")
    stu_column = settings["stocks"]["primary"]
    model_input = frame.assign(position=frame["position_b2"], hybrid_export=frame["floor_b2"],
                               hybrid_import=frame["ceiling_b2"])
    fair = tradeability.daily_fair_position(model_input, stu_column, settings["models"]["primary"], first_date,
                                            settings["models"]["fourier_harmonics"])
    frame["fair_position"] = fair["fair_position"].to_numpy()
    frame["gap"] = fair["signal"].to_numpy()
    frame["benchmark"] = fair["benchmark_signal"].to_numpy()
    gap = frame["gap"]
    frame["s1"] = gap - gap.rolling(DETREND_WINDOW, min_periods=DETREND_MIN).mean().shift(1)
    frame["s2"] = gap
    frame["s3"] = gap - gap.shift(HORIZON)
    frame["export_distance_usd"] = (frame["safex"] - frame["floor_b2"]) / frame["usdzar"]
    frame["import_distance_usd"] = (frame["ceiling_b2"] - frame["safex"]) / frame["usdzar"]
    frame["export_zone"] = frame["export_distance_usd"] <= TRIGGER_USD
    frame["import_zone"] = frame["import_distance_usd"] <= TRIGGER_USD
    frame = frame.merge(arb_returns(frame, inputs.prices, settings["symbols"][grain_class]), on="date", how="left")
    cumulative = frame["arb_ret"].fillna(0).cumsum().to_numpy()
    start, end = np.arange(len(frame)) + ENTRY_LAG, np.arange(len(frame)) + ENTRY_LAG + HORIZON
    forward = np.full(len(frame), np.nan)
    ok = end < len(frame)
    forward[ok] = cumulative[end[ok]] - cumulative[start[ok]]
    frame["fwd_arb_20"] = forward
    frame["grain_class"] = grain_class
    return frame


# ----------------------------------------------------------------------------- IC tests
def spell_count(mask: pd.Series) -> int:
    """Number of separate runs of consecutive True days: entries into the zone."""
    values = mask.fillna(False).astype(bool).to_numpy()
    return int((values & ~np.r_[False, values[:-1]]).sum())


def zone_ic(frame: pd.DataFrame, zone: str, start: str, end: str) -> pd.DataFrame:
    """IC of each signal and the benchmark against the forward arb return, inside one zone."""
    part = frame[(frame["date"] >= start) & (frame["date"] <= end) & frame[zone]]
    rows = []
    for label, column in {**SIGNALS, "benchmark (mean reversion)": "benchmark"}.items():
        result = tradeability.ic_statistics(part[column], part["fwd_arb_20"], HORIZON)
        p_value = float(2 * stats.norm.sf(abs(result["nw_t"]))) if np.isfinite(result["nw_t"]) else np.nan
        rows.append({"signal": label, "zone": zone, "start": start, "end": end, **result, "p_value": p_value,
                     "zone_spells": spell_count(frame.loc[(frame["date"] >= start) & (frame["date"] <= end), zone])})
    return pd.DataFrame(rows)


def zone_mean_return(frame: pd.DataFrame, zone: str, start: str, end: str) -> dict:
    """Mean 20-day arb return on zone days (the desk rule's raw edge), with a Newey-West t."""
    part = frame[(frame["date"] >= start) & (frame["date"] <= end) & frame[zone]].dropna(subset=["fwd_arb_20"])
    if len(part) < 30:
        return {"zone": zone, "start": start, "end": end, "n": len(part), "mean_fwd_arb_20": np.nan, "nw_t": np.nan}
    fit = sm.OLS(part["fwd_arb_20"].to_numpy(), np.ones(len(part))).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON - 1})
    return {"zone": zone, "start": start, "end": end, "n": int(len(part)), "effective_n": len(part) / HORIZON,
            "mean_fwd_arb_20": float(fit.params[0]), "nw_t": float(fit.tvalues[0]),
            "hit_rate": float((part["fwd_arb_20"] > 0).mean())}


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def decide(ic_dev: pd.DataFrame, ic_holdout: pd.DataFrame) -> pd.DataFrame:
    """Addendum 4 rule for yellow's export zone: holdout IC > 0 with Holm p < 0.05, development IC > 0,
    holdout IC above the benchmark's."""
    holdout = ic_holdout.set_index("signal")
    development = ic_dev.set_index("signal")
    names = list(SIGNALS)
    adjusted = holm([holdout.at[n, "p_value"] for n in names])
    benchmark = holdout.at["benchmark (mean reversion)", "ic"]
    rows = []
    for name, p_holm in zip(names, adjusted):
        rows.append({"signal": name, "dev_ic": development.at[name, "ic"], "holdout_ic": holdout.at[name, "ic"],
                     "holdout_nw_t": holdout.at[name, "nw_t"], "holdout_p": holdout.at[name, "p_value"],
                     "holdout_p_holm": p_holm, "benchmark_holdout_ic": benchmark,
                     "holdout_effective_n": holdout.at[name, "effective_n"],
                     "passes": bool(holdout.at[name, "ic"] > 0 and p_holm < 0.05
                                    and development.at[name, "ic"] > 0 and holdout.at[name, "ic"] > benchmark)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- backtest
def rule_positions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Target position decided at each close: +1 long the arb, -1 short, 0 flat."""
    export_zone, import_zone = frame["export_zone"].fillna(False), frame["import_zone"].fillna(False)
    in_sample = frame["fair_position"].notna()
    return {
        "T0 desk rule (export zone)": (export_zone & in_sample).astype(float),
        "T1 export zone & S1 > 0": (export_zone & (frame["s1"] > 0)).astype(float),
        "T2 export zone & S2 > 0": (export_zone & (frame["s2"] > 0)).astype(float),
        "T3 export zone & S3 > 0": (export_zone & (frame["s3"] > 0)).astype(float),
        "T4 short arb (import zone)": -(import_zone & in_sample).astype(float),
    }


def backtest_rule(frame: pd.DataFrame, target: pd.Series, costs: CostModel) -> pd.DataFrame:
    """Daily net arb P&L (log) of holding `target`, decided at the close on d and traded at the close
    on d + 1, so it first earns the return from d + 1 to d + 2. Costs: SAFEX round trip split on
    entry and exit plus one per roll held; CBOT leg $0.50/t round trip, as a fraction of SAFEX price."""
    held = target.shift(ENTRY_LAG + 1).fillna(0.0)       # exposure earning the return on each day
    traded = target.shift(ENTRY_LAG).fillna(0.0)         # position after the trade at each close
    change = traded.diff().abs().fillna(traded.abs())
    one_way = costs.one_way_r_t + CBOT_ROUND_TRIP_USD / 2 * frame["usdzar"]
    cost = change * one_way / frame["safex"]
    cost += (frame["roll_day"].fillna(False) & (held != 0)).astype(float) * costs.round_trip_r_t / frame["safex"]
    gross = held * frame["arb_ret"].fillna(0.0)
    return pd.DataFrame({"date": frame["date"], "held": held, "gross": gross, "cost": cost, "net": gross - cost})


def summarise_rule(daily: pd.DataFrame, start: str, end: str) -> dict:
    """Performance over a fixed window, flat days counted as zero."""
    part = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    net = part["net"]
    sd = net.std(ddof=1)
    entries = (part["held"] != 0) & (part["held"].shift().fillna(0) == 0)
    episode = entries.cumsum().where(part["held"] != 0)
    per_trade = net.groupby(episode).sum() if episode.notna().any() else pd.Series(dtype=float)
    return {"start": start, "end": end, "days": int(len(part)),
            "ann_return": float(net.mean() * TRADING_DAYS), "ann_vol": float(sd * np.sqrt(TRADING_DAYS)),
            "sharpe": float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "sharpe_daily": float(net.mean() / sd) if sd > 0 else np.nan,
            "max_drawdown": max_drawdown(net), "time_in_market": float((part["held"] != 0).mean()),
            "entries": int(entries.sum()), "hit_rate_per_trade": float((per_trade > 0).mean()) if len(per_trade) else np.nan,
            "avg_days_per_trade": float((part["held"] != 0).sum() / entries.sum()) if entries.sum() else np.nan,
            "cost_total": float(part["cost"].sum()), "skew": float(net.skew()), "kurtosis": float(net.kurtosis() + 3)}


def add_deflated_sharpe(table: pd.DataFrame) -> pd.DataFrame:
    """Deflated Sharpe for each rule over the full window, counting all rules as trials."""
    full = table[table["period"] == "full"].copy()
    sharpe_std = float(full["sharpe_daily"].std(ddof=1))
    full["deflated_sharpe"] = [deflated_sharpe(r.sharpe_daily, r.days, r.skew, r.kurtosis, len(full), sharpe_std)
                               for r in full.itertuples()]
    return table.merge(full[["rule", "deflated_sharpe"]], on="rule", how="left")
