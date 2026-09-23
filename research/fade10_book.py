"""The 10-day release fade across every leg: outright, parity, calendar spreads, white/yellow cross.

    python research/fade10_book.py

One rule for all legs, nothing re-tuned: at a SAGIS release, if the leg's out-of-sample z (from the
`signals` table) is at least 1 in size, fade it for 10 trading days, entered at the first close at
least one day after the release.

What each leg trades, as a return on one leg's notional:
  A  YMAZ outright          roll-adjusted front month
  D  YMAZ parity            SAFEX YMAZ minus CBOT x USD/ZAR at 10:00 UTC (long/short SAFEX vs a CBOT + FX hedge)
  B  WMAZ / YMAZ spread     2nd minus 1st month, R/t change over front price; roll days carry no P&L
  C  white/yellow cross     WMAZ minus YMAZ front month, equal notional

Costs: the default SAFEX round trip per SAFEX leg (two legs for spreads and the cross), plus R2/t for
the CBOT and FX hedge on the parity leg. Assumed, not measured fills.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import fairvalue as FV  # noqa: E402
from app.data import features as F  # noqa: E402
from app.data.backtest import TRADING_DAYS, CostModel, max_drawdown, run_backtest  # noqa: E402
from config import DB_PATH  # noqa: E402

T, HORIZON = 1.0, 10
PERIODS = {"last 5y": ("2021-09-01", "2026-12-31"), "2013-2021 check": ("2013-01-01", "2021-08-31")}
ONE = CostModel()
TWO = CostModel(half_spread_r_t=2 * ONE.half_spread_r_t, brokerage_r_t=2 * ONE.brokerage_r_t,
                slippage_r_t=2 * ONE.slippage_r_t)
HEDGE = CostModel(half_spread_r_t=ONE.half_spread_r_t + 1.0, brokerage_r_t=ONE.brokerage_r_t,
                  slippage_r_t=ONE.slippage_r_t)          # + R1/t per side for CBOT and FX


def _ns(s: pd.Series) -> pd.Series:
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    return s.sort_index()


def spread_index(cont: pd.DataFrame, sym: str) -> pd.Series:
    """Calendar-spread P&L as an index: daily change in (F2 - F1) over F1, only when both legs are
    the same contracts as the day before."""
    c = cont[cont.symbol == sym].sort_values("trade_date")
    same = (c.expiry_1.shift() == c.expiry_1) & (c.expiry_2.shift() == c.expiry_2)
    r = np.where(same, c.spread_2_1.diff() / c.close_1.shift(), 0.0)
    return _ns(pd.Series(np.exp(np.nancumsum(r)), index=c.trade_date.to_numpy()))


def legs(cont: pd.DataFrame, snap: pd.DataFrame) -> dict[str, tuple[pd.Series, str, str, CostModel]]:
    """name -> (tradeable index, signal model, signal grain_class, costs)."""
    wm, ym = _ns(FV.roll_adjusted_index(cont, "WMAZ")), _ns(FV.roll_adjusted_index(cont, "YMAZ"))
    world = _ns(FV.world_parity(snap)).reindex(ym.index).ffill(limit=3)
    parity = np.exp(np.log(ym) - np.log(world)).dropna()
    cross = np.exp(np.log(wm) - np.log(ym.reindex(wm.index))).dropna()
    return {
        "YMAZ outright (A)": (ym, "A", "yellow", ONE),
        "YMAZ parity (D)": (parity, "D", "yellow", HEDGE),
        "WMAZ spread (B)": (spread_index(cont, "WMAZ"), "B", "white", TWO),
        "YMAZ spread (B)": (spread_index(cont, "YMAZ"), "B", "yellow", TWO),
        "WM/YM cross (C)": (cross, "C", "white_vs_yellow", TWO),
    }


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def main() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sig = con.execute("SELECT * FROM signals").df()
        px = con.execute("SELECT * FROM prices").df()
        snap = con.execute("SELECT * FROM macro_snap").df()
    finally:
        con.close()
    sig["vintage_date"] = pd.to_datetime(sig["vintage_date"]).astype("datetime64[ns]")
    cont = F.continuous(px)
    ym_close = cont[cont.symbol == "YMAZ"].set_index("trade_date")["close_1"]
    ym_close.index = pd.DatetimeIndex(ym_close.index).as_unit("ns")

    rows, books = [], {}
    for period, (a, b) in PERIODS.items():
        daily = {}
        for name, (idx, model, cls, costs) in legs(cont, snap).items():
            e = sig[(sig.model == model) & (sig.grain_class == cls) & sig.vintage_date.between(a, b)
                    & (sig.z.abs() >= T)][["vintage_date", "z", "front_close"]].copy()
            if e.front_close.isna().all():                   # cross: cost on the yellow price
                e["front_close"] = ym_close.reindex(e.vintage_date, method="ffill").to_numpy()
            r = run_backtest(idx, e, horizon=HORIZON, costs=costs)
            t = r.trades
            d = r.daily.loc[a:b]
            daily[name] = d
            rows.append({"period": period, "leg": name, "trades": len(t),
                         "episodes": r.stats["n_episodes"],
                         "hit": float((t.net_ret > 0).mean()) if len(t) else np.nan,
                         "avg_net_pct": float(t.net_ret.mean() * 100) if len(t) else np.nan,
                         "cost_pct_of_gross": r.stats["cost_share"] * 100 if len(t) else np.nan,
                         "ann_vol_pct": float(d.std() * np.sqrt(TRADING_DAYS) * 100),
                         "_d": d})
        dd = pd.DataFrame(daily).fillna(0.0)
        start = dd.ne(0).any(axis=1).idxmax()               # common window: first trade of any leg
        dd = dd.loc[start:]
        for row in rows:
            if row["period"] == period:
                d = dd[row["leg"]]
                row.update({"sharpe": sharpe(d), "max_dd_pct": max_drawdown(d) * 100,
                            "total_pct": float(d.sum() * 100)})
                row.pop("_d")
        books[period] = dd
    res = pd.DataFrame(rows)
    return {"table": res, "books": books}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    out = main()
    cols = ["period", "leg", "trades", "episodes", "hit", "avg_net_pct", "cost_pct_of_gross",
            "sharpe", "max_dd_pct", "total_pct", "ann_vol_pct"]
    print(out["table"][cols].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    for p, dd in out["books"].items():
        print(f"\n{p}: daily P&L correlation")
        print(dd.corr().round(2).to_string())
