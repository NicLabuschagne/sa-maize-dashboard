"""10-day release fade entered at the next session's OPEN instead of its close.

    python research/fade10_open_entry.py

Timing: SAFEX marks at 12:00 SAST and SAGIS publishes at ~14:30 SAST, so the release is known before
the next session opens. The signal is unchanged: z uses the release-day 12:00 mark, which predates
the release.

  close entry   close of the first session after the release -> close 10 sessions later (as before)
  open entry    OPEN of that same session -> the same exit close

The only difference is the first session's open-to-close move, which is the part of the
post-release reaction the close-entry tests leave out. Same entries and exits, so the comparison
isolates exactly that.

Parity leg: the CBOT x USD/ZAR hedge is only available at the 12:00 SAST snapshot, so on open entry
the first three hours (09:00-12:00) of the SAFEX leg are unhedged. Stated, not modelled away.

Costs: default SAFEX round trip (+R2/t CBOT/FX on parity). SAFEX's morning book is thinner than at
the 12:00 mark, so open entry is also shown at double cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.data import features as F  # noqa: E402
from app.data.backtest import TRADING_DAYS, CostModel, max_drawdown  # noqa: E402
from config import DB_PATH  # noqa: E402
from fade10_book import HEDGE, ONE, PERIODS, legs  # noqa: E402

T, HORIZON = 1.0, 10
LEGS = ("YMAZ outright (A)", "YMAZ parity (D)")


def front_open(px: pd.DataFrame, cont: pd.DataFrame, sym: str) -> pd.Series:
    """Opening price of the contract that is the front month on each date."""
    c = cont[cont.symbol == sym][["trade_date", "expiry_1", "close_1"]]
    p = px[px.symbol == sym][["trade_date", "expiry", "open"]].rename(columns={"expiry": "expiry_1"})
    m = c.merge(p, on=["trade_date", "expiry_1"], how="left")
    m = m[(m.open > 0) & (m.close_1 > 0)]
    s = pd.Series(np.log(m.close_1 / m.open).to_numpy(), index=pd.DatetimeIndex(m.trade_date).as_unit("ns"))
    return s.sort_index()                          # log(close/open) of the front contract that day


def trades_for(idx: pd.Series, oc: pd.Series, ent: pd.DataFrame, costs: CostModel,
               open_entry: bool) -> tuple[pd.DataFrame, pd.Series]:
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
        if open_entry and np.isnan(seg0):
            continue                                                  # no open print that day
        path = np.diff(v[p0:p1 + 1]) * d
        cost = costs.log_drag(float(ent.front_close.iloc[k]))
        if daily.iloc[p0:p1 + 1].ne(0).any():
            raise ValueError("overlapping trades - daily P&L would need netting")
        daily.iloc[p0] += d * seg0
        daily.iloc[p0 + 1:p1 + 1] += path
        daily.iloc[p1] -= cost
        rows.append({"entry": dates[p0], "dir": d, "open_to_close": d * seg0,
                     "rest": float(path.sum()), "net": d * seg0 + float(path.sum()) - cost})
    return pd.DataFrame(rows), daily


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def main() -> pd.DataFrame:
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
    double = {"YMAZ outright (A)": CostModel(2 * ONE.half_spread_r_t, 2 * ONE.brokerage_r_t, 2 * ONE.slippage_r_t),
              "YMAZ parity (D)": CostModel(2 * HEDGE.half_spread_r_t, 2 * HEDGE.brokerage_r_t, 2 * HEDGE.slippage_r_t)}

    rows = []
    for period, (a, b) in PERIODS.items():
        books = {}
        for leg in LEGS:
            idx, model, cls, costs = L[leg]
            e = sig[(sig.model == model) & (sig.grain_class == cls) & sig.vintage_date.between(a, b)
                    & (sig.z.abs() >= T)].sort_values("vintage_date").reset_index(drop=True)
            for mode, oe, c in (("close entry", False, costs), ("open entry", True, costs),
                                ("open entry, 2x cost", True, double[leg])):
                t, d = trades_for(idx, oc, e, c, oe)
                books[(leg, mode)] = (t, d.loc[a:b])
        start = min(d.ne(0).idxmax() for _, d in books.values())      # one common window per period
        for (leg, mode), (t, d) in books.items():
            d = d.loc[start:]
            rows.append({"period": period, "leg": leg, "entry": mode, "trades": len(t),
                         "hit": float((t.net > 0).mean()),
                         "avg_net_pct": float(t.net.mean() * 100),
                         "avg_open_to_close_pct": float(t.open_to_close.mean() * 100),
                         "day1_hit": float((t.open_to_close > 0).mean()) if mode != "close entry" else np.nan,
                         "sharpe": sharpe(d), "max_dd_pct": max_drawdown(d) * 100,
                         "total_pct": float(d.sum() * 100)})
        combo = {}
        for mode in ("close entry", "open entry", "open entry, 2x cost"):
            dd = pd.concat([books[(leg, mode)][1].loc[start:] for leg in LEGS], axis=1).fillna(0.0)
            port = (dd / dd.std()).mean(axis=1) * dd.std().mean()     # inverse-vol, as before
            combo[mode] = port
            rows.append({"period": period, "leg": "A + D (inverse vol)", "entry": mode,
                         "trades": sum(len(books[(leg, mode)][0]) for leg in LEGS),
                         "sharpe": sharpe(port), "max_dd_pct": max_drawdown(port) * 100,
                         "total_pct": float(port.sum() * 100)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    r = main()
    print(r.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
