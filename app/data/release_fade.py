"""Release-fade portfolio backtest: fade each leg's release-day fair-value dislocation for a fixed hold.

At every SAGIS release, a leg whose out-of-sample z is at least `threshold` in size is faded: short
when rich, long when cheap. Entry is at the next session's open (outright legs, optional) or its
12:00 mark, and the exit is the mark `hold` sessions later. An optional hard stop on the daily mark
closes a trade early. One position per leg at a time: a release that fires while the leg is still
in a trade is skipped.

The signal is known before entry. z uses the release-day 12:00 mark, and SAGIS publishes at ~14:30.

What each leg trades, as a return on one leg's notional:
  outright  roll-adjusted front month
  parity    SAFEX minus CBOT x USD/ZAR at the 12:00 snapshot (the hedge only exists at the mark)
  spread    2nd minus 1st month, R/t change over the front price; roll days carry no P&L
  cross     WMAZ minus YMAZ front month, equal notional
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.data import fairvalue as FV
from app.data.backtest import TRADING_DAYS, CostModel


@dataclass(frozen=True)
class Leg:
    label: str
    model: str
    grain_class: str
    kind: str                # outright | parity | spread | cross
    symbol: str              # price whose level sets the cost drag
    n_safex_legs: int
    hedge_r_t: float = 0.0   # extra round-trip cost for a CBOT/FX hedge

    @property
    def can_open(self) -> bool:
        return self.kind == "outright"


LEGS: dict[str, Leg] = {
    "YMAZ outright": Leg("YMAZ outright", "A", "yellow", "outright", "YMAZ", 1),
    "WMAZ outright": Leg("WMAZ outright", "A", "white", "outright", "WMAZ", 1),
    "YMAZ parity": Leg("YMAZ parity", "D", "yellow", "parity", "YMAZ", 1, hedge_r_t=2.0),
    "YMAZ spread": Leg("YMAZ spread", "B", "yellow", "spread", "YMAZ", 2),
    "WMAZ spread": Leg("WMAZ spread", "B", "white", "spread", "WMAZ", 2),
    "WM/YM cross": Leg("WM/YM cross", "C", "white_vs_yellow", "cross", "YMAZ", 2),
}
DEFAULT_LEGS = ("YMAZ outright", "YMAZ parity")


def leg_costs(leg: Leg, multiplier: float = 1.0) -> CostModel:
    base = CostModel()
    n = leg.n_safex_legs * multiplier
    return CostModel(half_spread_r_t=base.half_spread_r_t * n + leg.hedge_r_t / 2 * multiplier,
                     brokerage_r_t=base.brokerage_r_t * n, slippage_r_t=base.slippage_r_t * n)


# ----------------------------------------------------------------------------- instruments
def _ns(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    return s.sort_index()


def spread_index(cont: pd.DataFrame, sym: str) -> pd.Series:
    c = cont[cont.symbol == sym].sort_values("trade_date")
    same = (c.expiry_1.shift() == c.expiry_1) & (c.expiry_2.shift() == c.expiry_2)
    r = np.where(same, c.spread_2_1.diff() / c.close_1.shift(), 0.0)
    return _ns(pd.Series(np.exp(np.nancumsum(r)), index=c.trade_date.to_numpy()))


def open_to_close(px: pd.DataFrame, cont: pd.DataFrame, sym: str) -> pd.Series:
    """log(mark / open) of the front contract each day."""
    c = cont[cont.symbol == sym][["trade_date", "expiry_1", "close_1"]]
    p = px[px.symbol == sym][["trade_date", "expiry", "open"]].rename(columns={"expiry": "expiry_1"})
    m = c.merge(p, on=["trade_date", "expiry_1"], how="left")
    m = m[(m.open > 0) & (m.close_1 > 0)]
    return _ns(pd.Series(np.log(m.close_1 / m.open).to_numpy(), index=m.trade_date.to_numpy()))


def instruments(cont: pd.DataFrame, px: pd.DataFrame, snap: pd.DataFrame) -> dict:
    """Tradeable index per leg, open-to-mark moves for the outrights, and front marks for costs."""
    wm, ym = _ns(FV.roll_adjusted_index(cont, "WMAZ")), _ns(FV.roll_adjusted_index(cont, "YMAZ"))
    world = _ns(FV.world_parity(snap)).reindex(ym.index).ffill(limit=3)
    idx = {"YMAZ outright": ym, "WMAZ outright": wm,
           "YMAZ parity": np.exp(np.log(ym) - np.log(world)).dropna(),
           "YMAZ spread": spread_index(cont, "YMAZ"), "WMAZ spread": spread_index(cont, "WMAZ"),
           "WM/YM cross": np.exp(np.log(wm) - np.log(ym.reindex(wm.index))).dropna()}
    oc = {"YMAZ": open_to_close(px, cont, "YMAZ"), "WMAZ": open_to_close(px, cont, "WMAZ")}
    marks = {s: _ns(cont[cont.symbol == s].set_index("trade_date")["close_1"]) for s in ("WMAZ", "YMAZ")}
    return {"idx": idx, "oc": oc, "marks": marks}


# ----------------------------------------------------------------------------- engine
def run_leg(idx: pd.Series, entries: pd.DataFrame, hold: int, costs: CostModel,
            oc: pd.Series | None = None, marks: pd.Series | None = None,
            stop_pct: float | None = None) -> dict:
    """entries: vintage_date, z (already filtered to |z| >= threshold). Returns trades and the
    leg's daily log return (flat days 0). `oc` given = enter at the open."""
    dates, v = idx.index, np.log(idx.to_numpy())
    ent = entries.sort_values("vintage_date").reset_index(drop=True)
    pos = dates.searchsorted(pd.DatetimeIndex(ent.vintage_date).as_unit("ns") + pd.Timedelta(days=1))
    daily = np.zeros(len(v))
    rows, busy_until = [], -1
    for k, p0 in enumerate(pos):
        if p0 <= busy_until or p0 + 1 >= len(v):
            continue
        d = -1 if ent.z.iloc[k] > 0 else 1
        seg0 = 0.0
        if oc is not None:
            seg0 = float(oc.get(dates[p0], np.nan))
            if np.isnan(seg0):
                continue
        p1 = min(p0 + hold, len(v) - 1)
        steps = np.r_[d * seg0, np.diff(v[p0:p1 + 1]) * d]
        cum = np.cumsum(steps)
        reason = "time"
        if stop_pct:
            hit = np.nonzero(cum <= np.log1p(-stop_pct / 100))[0]
            if len(hit):
                p1, reason = p0 + int(hit[0]), "stop"
                steps, cum = steps[:hit[0] + 1], cum[:hit[0] + 1]
        px0 = float(marks.asof(ent.vintage_date.iloc[k])) if marks is not None else float(idx.iloc[p0])
        cost = costs.log_drag(px0)
        daily[p0:p1 + 1] += steps
        daily[p1] -= cost
        busy_until = p1
        rows.append({"release": ent.vintage_date.iloc[k], "entry": dates[p0], "exit": dates[p1],
                     "side": "short" if d < 0 else "long", "z": float(ent.z.iloc[k]),
                     "sessions": p1 - p0, "exit_on": reason,
                     "net_pct": float(np.expm1(cum[-1] - cost) * 100),
                     "worst_pct": float(np.expm1(min(0.0, cum.min())) * 100)})
    return {"trades": pd.DataFrame(rows), "daily": pd.Series(daily, index=dates)}


def run_portfolio(signals: pd.DataFrame, inst: dict, legs: dict[str, float], start, end,
                  threshold: float = 1.0, hold: int = 10, open_entry: bool = True,
                  cost_mult: float = 1.0, stop_pct: float | None = None) -> dict:
    """legs: name -> weight (normalised to sum to 1). Returns daily simple returns per leg and for
    the portfolio over [start, end], plus the trade log."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    w = pd.Series(legs, dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    sig = signals.assign(vintage_date=pd.to_datetime(signals.vintage_date).astype("datetime64[ns]"))
    daily, trades = {}, []
    for name in w.index:
        leg = LEGS[name]
        e = sig[(sig.model == leg.model) & (sig.grain_class == leg.grain_class)
                & sig.vintage_date.between(start, end) & (sig.z.abs() >= threshold)]
        oc = inst["oc"][leg.symbol] if (open_entry and leg.can_open) else None
        r = run_leg(inst["idx"][name], e[["vintage_date", "z"]], hold, leg_costs(leg, cost_mult),
                    oc=oc, marks=inst["marks"][leg.symbol], stop_pct=stop_pct)
        daily[name] = np.expm1(r["daily"].loc[start:end])
        trades.append(r["trades"].assign(leg=name))
    dd = pd.DataFrame(daily).fillna(0.0)
    port = (dd * w.reindex(dd.columns)).sum(axis=1)
    t = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return {"legs": dd, "portfolio": port, "trades": t, "weights": w}


# ----------------------------------------------------------------------------- metrics
def metrics(r: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    """r: daily simple returns, flat days included."""
    nan = float("nan")
    if len(r) < 2 or r.std() == 0:
        return {"Trades": 0 if trades is None else len(trades)}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    dd = eq / eq.cummax() - 1
    under = dd < 0
    runs = under.groupby((~under).cumsum()).sum()
    downside = float(np.sqrt((np.minimum(r, 0) ** 2).mean()))
    max_dd = float(dd.min())
    out = {"CAGR %": cagr * 100, "Vol %": float(r.std() * np.sqrt(TRADING_DAYS) * 100),
           "Sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)),
           "Sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside else nan,
           "Max DD %": max_dd * 100, "Calmar": cagr / abs(max_dd) if max_dd else nan,
           "Longest DD (sessions)": int(runs.max()) if len(runs) else 0,
           "Total %": float((eq.iloc[-1] - 1) * 100), "In market %": float((r != 0).mean() * 100)}
    if trades is not None and len(trades):
        win, loss = trades[trades.net_pct > 0], trades[trades.net_pct <= 0]
        out.update({"Trades": len(trades), "Hit %": len(win) / len(trades) * 100,
                    "Avg win %": float(win.net_pct.mean()) if len(win) else nan,
                    "Avg loss %": float(loss.net_pct.mean()) if len(loss) else nan,
                    "Profit factor": float(win.net_pct.sum() / abs(loss.net_pct.sum())) if len(loss) else nan,
                    "Worst trade %": float(trades.net_pct.min()),
                    "Worst open loss %": float(trades.worst_pct.min())})
    return out


def curves(res: dict) -> pd.DataFrame:
    c = pd.DataFrame({"Portfolio": (1 + res["portfolio"]).cumprod(),
                      **{k: (1 + res["legs"][k]).cumprod() for k in res["legs"].columns}})
    c["drawdown"] = c["Portfolio"] / c["Portfolio"].cummax() - 1
    return c
