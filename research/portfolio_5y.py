"""Combined release-fade portfolio, last five years.

    python research/portfolio_5y.py

Legs (same rule as every test here: |z| >= 1 at a SAGIS release, fade, 10-trading-day hold):
  YMAZ outright (Model A)  entered at the next session's OPEN (release is 14:30, after the 12:00 mark)
  YMAZ parity   (Model D)  entered at the next session's 12:00 mark, when the CBOT/FX hedge can go on

Weights: fixed 50/50 of capital. No look-ahead in sizing.
Stop: none on price. Every trade exits at the close 10 sessions after entry (time stop).
Costs: default SAFEX round trip, +R2/t for the parity hedge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.data import features as F  # noqa: E402
from app.data.backtest import TRADING_DAYS  # noqa: E402
from config import DB_PATH  # noqa: E402
from fade10_book import legs  # noqa: E402
from fade10_open_entry import front_open  # noqa: E402

T, HORIZON = 1.0, 10
START = "2021-09-01"
BOOK = {"YMAZ outright (A)": True, "YMAZ parity (D)": False}      # leg -> enter at open?
WEIGHTS = {"YMAZ outright (A)": 0.5, "YMAZ parity (D)": 0.5}


def run_leg(idx: pd.Series, oc: pd.Series, ent: pd.DataFrame, costs, open_entry: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Trades with their worst open loss (MAE), and the leg's daily log-return series."""
    dates, v = idx.index, np.log(idx.to_numpy())
    pos = dates.searchsorted(pd.DatetimeIndex(ent.vintage_date).as_unit("ns") + pd.Timedelta(days=1))
    daily = pd.Series(0.0, index=dates)
    rows = []
    for k, p0 in enumerate(pos):
        p1 = p0 + HORIZON
        if p1 >= len(v):
            continue
        d = -1 if ent.z.iloc[k] > 0 else 1
        seg0 = float(oc.get(dates[p0], np.nan)) if open_entry else 0.0
        if np.isnan(seg0):
            continue
        steps = np.r_[d * seg0, np.diff(v[p0:p1 + 1]) * d]             # day 1 open->close, then closes
        cost = costs.log_drag(float(ent.front_close.iloc[k]))
        daily.iloc[p0:p1 + 1] += steps
        daily.iloc[p1] -= cost
        cum = np.cumsum(steps)
        rows.append({"release": ent.vintage_date.iloc[k].date(), "entry": dates[p0].date(),
                     "exit": dates[p1].date(), "side": "short" if d < 0 else "long",
                     "z": round(float(ent.z.iloc[k]), 2), "net_pct": (cum[-1] - cost) * 100,
                     "mae_pct": min(0.0, float(cum.min())) * 100, "mfe_pct": max(0.0, float(cum.max())) * 100})
    return pd.DataFrame(rows), daily


def metrics(r: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    """r = daily simple returns of the portfolio."""
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    under = dd < 0
    runs = under.groupby((~under).cumsum()).sum()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    out = {"CAGR %": cagr * 100, "Ann vol %": r.std() * np.sqrt(TRADING_DAYS) * 100,
           "Sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
           "Sortino": r.mean() / downside * np.sqrt(TRADING_DAYS),
           "Max DD %": dd.min() * 100, "Calmar": cagr / abs(dd.min()),
           "Longest DD (days)": int(runs.max()) if len(runs) else 0,
           "Total return %": (eq.iloc[-1] - 1) * 100, "Time in market %": (r != 0).mean() * 100}
    if trades is not None and len(trades):
        w, l_ = trades[trades.net_pct > 0], trades[trades.net_pct <= 0]
        out.update({"Trades": len(trades), "Hit rate %": len(w) / len(trades) * 100,
                    "Avg win %": w.net_pct.mean(), "Avg loss %": l_.net_pct.mean(),
                    "Profit factor": w.net_pct.sum() / abs(l_.net_pct.sum()) if len(l_) else np.inf,
                    "Worst trade %": trades.net_pct.min(), "Worst MAE %": trades.mae_pct.min(),
                    "Median MAE %": trades.mae_pct.median()})
    return out


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
    oc = front_open(px, cont, "YMAZ")
    L = legs(cont, snap)

    trades, daily = {}, {}
    for leg, open_entry in BOOK.items():
        idx, model, cls, costs = L[leg]
        e = sig[(sig.model == model) & (sig.grain_class == cls) & (sig.vintage_date >= START)
                & (sig.z.abs() >= T)].sort_values("vintage_date").reset_index(drop=True)
        t, d = run_leg(idx, oc, e, costs, open_entry)
        trades[leg] = t.assign(leg=leg)
        daily[leg] = np.expm1(d.loc[START:])                              # log -> simple
    dd = pd.DataFrame(daily).fillna(0.0)
    # window: 1 Sep 2021 to the last price date, flat days included
    port = sum(dd[k] * w for k, w in WEIGHTS.items())
    allt = pd.concat(trades.values(), ignore_index=True)
    table = pd.DataFrame({"Portfolio 50/50": metrics(port, allt),
                          **{k: metrics(dd[k], trades[k]) for k in BOOK}})
    corr = float(dd.corr().iloc[0, 1])
    curves = pd.DataFrame({"portfolio": (1 + port).cumprod(),
                           **{k: (1 + dd[k]).cumprod() for k in BOOK}})
    curves["drawdown"] = curves.portfolio / curves.portfolio.cummax() - 1
    return {"table": table, "trades": allt, "curves": curves, "corr": corr}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    out = main()
    print(out["table"].round(2).to_string())
    print(f"\ndaily P&L correlation between legs: {out['corr']:+.2f}")
    print(out["trades"].round(2).to_string(index=False))
    c = out["curves"].resample("W-FRI").last()
    Path(sys.argv[1] if len(sys.argv) > 1 else "curves.json").write_text(json.dumps({
        "dates": [d.strftime("%Y-%m-%d") for d in c.index],
        **{k: [round(float(x), 4) for x in c[k]] for k in c.columns}}))
