"""Band extremes with and without the fundamental gap (Addendum 5).

On the dashboard's band and fair position: when SAFEX reaches the export extreme (position < 0.1)
or the import extreme (> 0.9), does the arb move back? And is that move better when the fair position
agrees (gap = fair - position >= +0.2 for a long, <= -0.2 for a short)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.data import band_fairvalue as BFV
from app.data import features as F
from app.data.backtest import CostModel
from research.band_position import edge_test, tradeability

LONG_ENTRY, LONG_REARM = 0.1, 0.2
SHORT_ENTRY, SHORT_REARM = 0.9, 0.8
GAP_CONFIRM = 0.2
HORIZONS = (20, 40, 60)
PRIMARY_HORIZON = 40
ENTRY_LAG = 1
MIN_SEASONS = 3
CBOT_ROUND_TRIP_USD = 0.50


def build_daily(tables: dict, grain_class: str, symbol: str) -> pd.DataFrame:
    """Dashboard band and fair position (first fit May 2015), plus daily arb and SAFEX log returns."""
    cont = F.continuous(tables["prices"])
    daily = BFV.build(cont, tables["macro_snap"], tables["sagis_parity"], F.sd_monthly(tables["balance_sheet"]),
                      tables["sagis_weekly"], grain_class, symbol, min_seasons=MIN_SEASONS)
    daily["gap"] = daily["fair_position"] - daily["position"]
    snap = tables["macro_snap"].pivot(index="date", columns="series", values="value").reset_index()
    snap.columns.name = None
    snap = snap.rename(columns={"usdzar_safexclose": "usdzar"})[["date", "usdzar"]]
    snap["date"] = snap["date"].astype("datetime64[ns]")
    daily = pd.merge_asof(daily.sort_values("date"), snap.sort_values("date").dropna(), on="date",
                          direction="backward", tolerance=pd.Timedelta(days=4))
    returns = edge_test.arb_returns(daily, tables["prices"], symbol)
    return daily.merge(returns, on="date", how="left").reset_index(drop=True)


def find_events(daily: pd.DataFrame, side: str) -> pd.DataFrame:
    """Crossings into an extreme with hysteresis, only where a fair position exists.

    long:  position < 0.1 after >= 0.1 the day before; re-armed once position has been above 0.2.
    short: position > 0.9 after <= 0.9 the day before; re-armed once position has been below 0.8.
    """
    position = daily["position"].to_numpy()
    fair_known = daily["fair_position"].notna().to_numpy()
    rows, armed = [], True
    for i in range(1, len(daily)):
        if not (np.isfinite(position[i]) and np.isfinite(position[i - 1])):
            continue
        if side == "long":
            if position[i] > LONG_REARM:
                armed = True
            crossed = position[i] < LONG_ENTRY <= position[i - 1]
        else:
            if position[i] < SHORT_REARM:
                armed = True
            crossed = position[i] > SHORT_ENTRY >= position[i - 1]
        if crossed and armed and fair_known[i]:
            gap = daily["gap"].iat[i]
            confirmed = gap >= GAP_CONFIRM if side == "long" else gap <= -GAP_CONFIRM
            rows.append({"row": i, "date": daily["date"].iat[i], "side": side, "position": position[i],
                         "fair_position": daily["fair_position"].iat[i], "gap": gap, "confirmed": bool(confirmed)})
            armed = False
    return pd.DataFrame(rows)


def event_outcomes(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Forward arb and SAFEX returns from the close on d + 1, sign-adjusted for shorts, and the split of
    each position change into SAFEX moving (edges fixed at entry) versus the edges moving."""
    if events.empty:
        return events
    arb = daily["arb_ret"].fillna(0).cumsum().to_numpy()
    safex_ret = daily["safex_ret"].fillna(0).cumsum().to_numpy()
    price, position = daily["safex"].to_numpy(), daily["position"].to_numpy()
    export_edge, width = daily["export_edge"].to_numpy(), daily["band_width"].to_numpy()
    out = events.copy()
    sign = np.where(out["side"] == "long", 1.0, -1.0)
    for h in HORIZONS:
        start, end = out["row"].to_numpy() + ENTRY_LAG, out["row"].to_numpy() + ENTRY_LAG + h
        ok = end < len(daily)
        for name, values in (("arb", arb), ("safex", safex_ret)):
            forward = np.full(len(out), np.nan)
            forward[ok] = values[end[ok]] - values[start[ok]]
            out[f"fwd_{name}_{h}"] = forward * sign
        change, price_part = np.full(len(out), np.nan), np.full(len(out), np.nan)
        change[ok] = position[end[ok]] - position[start[ok]]
        price_part[ok] = (price[end[ok]] - export_edge[start[ok]]) / width[start[ok]] - position[start[ok]]
        out[f"position_change_{h}"] = change
        out[f"from_safex_{h}"] = price_part
        out[f"from_edges_{h}"] = change - price_part
    return out


def group_table(events: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Mean, median, hit rate and event-level t of the forward arb return, per side, group and period."""
    rows = []
    for side in ("long", "short"):
        for period, (start, end) in periods.items():
            part = events[(events["side"] == side) & (events["date"] >= start) & (events["date"] <= end)]
            for group, mask in (("edge alone (all)", pd.Series(True, index=part.index)),
                                ("confirmed by gap", part["confirmed"]), ("not confirmed", ~part["confirmed"])):
                g = part[mask]
                for h in HORIZONS:
                    r = g[f"fwd_arb_{h}"].dropna()
                    t = float(stats.ttest_1samp(r, 0).statistic) if len(r) >= 3 else np.nan
                    rows.append({"side": side, "period": period, "group": group, "horizon": h, "events": len(r),
                                 "mean_arb": r.mean() if len(r) else np.nan, "median_arb": r.median() if len(r) else np.nan,
                                 "hit_rate": (r > 0).mean() if len(r) else np.nan, "t_events": t,
                                 "mean_safex": g[f"fwd_safex_{h}"].dropna().mean() if len(g) else np.nan,
                                 "median_from_safex": g[f"from_safex_{h}"].median() if len(g) else np.nan,
                                 "median_from_edges": g[f"from_edges_{h}"].median() if len(g) else np.nan})
    return pd.DataFrame(rows)


def zone_gap_ic(daily: pd.DataFrame, start: str, end: str) -> dict:
    """On every long-zone day (position < 0.1), IC of the gap against the forward 40-day arb return."""
    arb = daily["arb_ret"].fillna(0).cumsum().to_numpy()
    i = np.arange(len(daily))
    ok = i + ENTRY_LAG + PRIMARY_HORIZON < len(daily)
    forward = np.full(len(daily), np.nan)
    forward[ok] = arb[i[ok] + ENTRY_LAG + PRIMARY_HORIZON] - arb[i[ok] + ENTRY_LAG]
    part = daily.assign(fwd=forward)
    part = part[(part["date"] >= start) & (part["date"] <= end) & (part["position"] < LONG_ENTRY)]
    return {"start": start, "end": end, **tradeability.ic_statistics(part["gap"], part["fwd"], PRIMARY_HORIZON)}


def trade_view(daily: pd.DataFrame, events: pd.DataFrame, costs: CostModel) -> pd.DataFrame:
    """Net 40-day result per event: sign-adjusted arb return minus SAFEX round trip, CBOT leg and a
    SAFEX round trip for every roll inside the holding period, as fractions of the entry price."""
    out = events.dropna(subset=[f"fwd_arb_{PRIMARY_HORIZON}"]).copy()
    rolls = daily["roll_day"].fillna(False).astype(int).cumsum().to_numpy()
    start = out["row"].to_numpy() + ENTRY_LAG
    end = start + PRIMARY_HORIZON
    price = daily["safex"].to_numpy()[start]
    usdzar = daily["usdzar"].to_numpy()[start]
    n_rolls = rolls[end] - rolls[start]
    cost = (costs.round_trip_r_t * (1 + n_rolls) + CBOT_ROUND_TRIP_USD * usdzar) / price
    out["cost"] = cost
    out["net"] = out[f"fwd_arb_{PRIMARY_HORIZON}"] - cost
    return out


def decide(table: pd.DataFrame) -> dict:
    """Addendum 5 rule on yellow's long side at 40 days."""
    t = table[(table.side == "long") & (table.horizon == PRIMARY_HORIZON)].set_index(["period", "group"])

    def mean(period: str, group: str) -> float:
        return float(t.at[(period, group), "mean_arb"]) if (period, group) in t.index else np.nan

    checks = {
        "development: confirmed > not confirmed": mean("development", "confirmed by gap") > mean("development", "not confirmed"),
        "holdout: confirmed > not confirmed": mean("holdout", "confirmed by gap") > mean("holdout", "not confirmed"),
        "holdout: confirmed mean > 0": mean("holdout", "confirmed by gap") > 0,
    }
    return {"checks": {k: bool(v) for k, v in checks.items()}, "gap_adds_value": bool(all(checks.values()))}
