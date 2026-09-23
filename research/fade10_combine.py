"""How correlated are the WMAZ and YMAZ 10-day release fades, and what does combining them give?

    python research/fade10_combine.py

Same configuration as threshold_fade_diagnostics.py: |z| >= 1 at a release, fade, 10-trading-day
hold, default SAFEX costs. No new parameters.
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
PERIODS = {"last 5y": ("2021-09-01", "2026-12-31"), "2013-2021": ("2013-01-01", "2021-08-31"),
           "full 2013-2026": ("2013-01-01", "2026-12-31")}


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def main() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sig = con.execute("SELECT * FROM signals WHERE model = 'A'").df()
        px = con.execute("SELECT * FROM prices").df()
    finally:
        con.close()
    sig["vintage_date"] = pd.to_datetime(sig["vintage_date"]).astype("datetime64[ns]")
    cont = F.continuous(px)
    out = {}
    for period, (a, b) in PERIODS.items():
        books, trades, mkt = {}, {}, {}
        for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
            idx = FV.roll_adjusted_index(cont, sym)
            e = sig[(sig.grain_class == cls) & sig.vintage_date.between(a, b) & (sig.z.abs() >= T)]
            r = run_backtest(idx, e[["vintage_date", "z", "front_close"]], horizon=HORIZON, costs=CostModel())
            books[sym] = r.daily.loc[a:b]
            trades[sym] = r.trades.assign(release=e.sort_values("vintage_date").vintage_date.to_numpy()[:len(r.trades)])
            mkt[sym] = np.log(idx).diff().loc[a:b]
        d = pd.concat(books, axis=1).fillna(0.0)
        d = d.loc[max(t.entry_date.min() for t in trades.values()):]       # from the first trade on
        both_on = (d != 0).all(axis=1)
        tw, ty = trades["WMAZ"], trades["YMAZ"]
        same = tw.merge(ty, on="release", suffixes=("_w", "_y"))
        vol = d.std()
        inv = (1 / vol) / (1 / vol).sum()
        port_eq, port_iv = d.mean(axis=1), (d * inv).sum(axis=1)
        rho = float(d.corr().iloc[0, 1])
        sw, sy = sharpe(d.WMAZ), sharpe(d.YMAZ)
        out[period] = {
            "trades W / Y": f"{len(tw)} / {len(ty)}",
            "releases traded by both": len(same),
            "…same direction": int((same.dir_w == same.dir_y).sum()),
            "corr of trade returns (shared releases)": float(same.net_ret_w.corr(same.net_ret_y)) if len(same) > 2 else np.nan,
            "corr daily P&L (all days)": rho,
            "corr daily P&L (both in market)": float(d[both_on].corr().iloc[0, 1]) if both_on.sum() > 10 else np.nan,
            "corr of the markets themselves": float(pd.concat(mkt, axis=1).corr().iloc[0, 1]),
            "Sharpe WMAZ": sw, "Sharpe YMAZ": sy,
            "Sharpe combined, equal notional": sharpe(port_eq),
            "Sharpe combined, inverse vol": sharpe(port_iv),
            "formula (equal vol): (S1+S2)/sqrt(2(1+rho))": (sw + sy) / np.sqrt(2 * (1 + rho)),
            "ann vol combined %": float(port_eq.std() * np.sqrt(TRADING_DAYS) * 100),
            "max DD combined %": max_drawdown(port_eq) * 100,
            "max DD WMAZ / YMAZ %": f"{max_drawdown(d.WMAZ)*100:.1f} / {max_drawdown(d.YMAZ)*100:.1f}",
        }
    return {"table": pd.DataFrame(out)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = main()["table"]
    print(t.to_string(float_format=lambda v: f"{v:+.2f}"))
