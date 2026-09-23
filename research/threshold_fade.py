"""State-based fade of the Model A release signal on WMAZ and YMAZ.

    python research/threshold_fade.py

Rules, fixed before the first run:

  entry   at a release, flat, |z| >= T  -> fade (short if z > 0, long if z < 0), at the first close
          at least one day after the release (same convention as every other test here)
  exit    at the first later release where the entry-signed z is back below EXIT_Z (inside the
          neutral band on Home); if z has instead crossed past -T, exit and reverse
  stop    time stop after MAX_RELEASES releases in the trade
  costs   default SAFEX round trip at entry+exit, plus a round trip at every contract roll held

T in {1.0, 1.5}: two configurations. The last five years is the period the idea came from (the
Home replay), so it is in-sample to that inspection. 2013-2021 is run with identical rules as the
out-of-sample check.
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
from app.data.backtest import TRADING_DAYS, CostModel, max_drawdown  # noqa: E402
from config import DB_PATH  # noqa: E402

THRESHOLDS = (1.0, 1.5)
EXIT_Z = 0.5
MAX_RELEASES = 6
ENTRY_LAG_DAYS = 1
PERIODS = {"last 5y (Sep 2021 – Sep 2026)": ("2021-09-01", "2026-12-31"),
           "check (2013 – Aug 2021)": ("2013-01-01", "2021-08-31")}
OUT = Path(__file__).with_name("results_threshold_fade.csv")


def run_product(sig: pd.DataFrame, idx: pd.Series, expiry: pd.Series, T: float,
                costs: CostModel, start: str, end: str) -> dict:
    """Walk the releases in order, carrying at most one position."""
    s = sig[(sig.vintage_date >= start) & (sig.vintage_date <= end)].dropna(subset=["z"])
    s = s.sort_values("vintage_date").reset_index(drop=True)
    dates, v = idx.index, np.log(idx.to_numpy())
    exp_arr = expiry.reindex(dates).ffill().to_numpy()
    at = dates.searchsorted(pd.DatetimeIndex(s.vintage_date).as_unit("ns") + pd.Timedelta(days=ENTRY_LAG_DAYS))

    trades, pos = [], None
    for k in range(len(s)):
        z, i = float(s.z.iloc[k]), int(at[k])
        if i >= len(v):
            break
        if pos is not None:
            pos["n_rel"] += 1
            signed = z * pos["zsign"]
            reason = ("reverted" if signed < EXIT_Z else
                      "time stop" if pos["n_rel"] >= MAX_RELEASES else None)
            if reason:
                trades.append(close(pos, i, reason, v, dates, exp_arr, costs))
                pos = None
                if reason == "reverted" and -signed >= T:          # crossed to the other side
                    pos = open_(s, k, i, z, v)
                    pos["reversal"] = True
                continue
        if pos is None and abs(z) >= T:
            pos = open_(s, k, i, z, v)
    if pos is not None:                                             # mark to market at the end
        last = min(dates.searchsorted(pd.Timestamp(end), side="right") - 1, len(v) - 1)
        trades.append(close(pos, last, "open at end", v, dates, exp_arr, costs))
    t = pd.DataFrame(trades)
    return {"trades": t, "daily": daily_returns(t, v, dates, start, end)}


def open_(s: pd.DataFrame, k: int, i: int, z: float, v: np.ndarray) -> dict:
    return {"release": s.vintage_date.iloc[k], "i0": i, "z0": z, "zsign": np.sign(z),
            "dir": -int(np.sign(z)), "n_rel": 0, "px0": float(s.front_close.iloc[k]), "reversal": False}


def close(pos: dict, i1: int, reason: str, v: np.ndarray, dates: pd.DatetimeIndex,
          exp_arr: np.ndarray, costs: CostModel) -> dict:
    i0 = pos["i0"]
    path = pos["dir"] * (v[i0:i1 + 1] - v[i0])
    rolls = int((pd.Series(exp_arr[i0:i1 + 1]).ne(pd.Series(exp_arr[i0:i1 + 1]).shift()).sum()) - 1)
    cost = costs.log_drag(pos["px0"]) * (1 + max(rolls, 0))
    gross = float(path[-1])
    return {"entry_release": pos["release"].date(), "entry_date": dates[i0].date(), "exit_date": dates[i1].date(),
            "direction": "short" if pos["dir"] < 0 else "long", "z_entry": round(pos["z0"], 2),
            "releases_held": pos["n_rel"], "days": i1 - i0, "rolls": max(rolls, 0), "exit": reason,
            "gross_pct": gross * 100, "cost_pct": cost * 100, "net_pct": (gross - cost) * 100,
            "worst_pct": float(path.min()) * 100, "entry_px": pos["px0"],
            "net_r_t": (np.exp(gross) - 1) * pos["px0"] - costs.round_trip_r_t * (1 + max(rolls, 0)),
            "i0": i0, "i1": i1, "dir": pos["dir"], "cost": cost}


def daily_returns(t: pd.DataFrame, v: np.ndarray, dates: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    out = pd.Series(0.0, index=dates)
    for r in t.itertuples():
        step = np.diff(v[r.i0:r.i1 + 1]) * r.dir
        out.iloc[r.i0 + 1:r.i1 + 1] += step
        out.iloc[r.i1] -= r.cost
    return out.loc[start:end]


def stats(daily: pd.Series, t: pd.DataFrame) -> dict:
    sd = daily.std(ddof=1)
    return {"trades": len(t), "hit_rate": float((t.net_pct > 0).mean()) if len(t) else np.nan,
            "avg_net_pct": float(t.net_pct.mean()) if len(t) else np.nan,
            "avg_net_r_t": float(t.net_r_t.mean()) if len(t) else np.nan,
            "avg_days": float(t.days.mean()) if len(t) else np.nan,
            "worst_trade_dd_pct": float(t.worst_pct.min()) if len(t) else np.nan,
            "total_pct": float(daily.sum() * 100),
            "ann_ret_pct": float(daily.mean() * TRADING_DAYS * 100),
            "ann_vol_pct": float(sd * np.sqrt(TRADING_DAYS) * 100),
            "sharpe": float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "max_dd_pct": max_drawdown(daily) * 100,
            "time_in_mkt": float((daily != 0).mean())}


def main() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sig = con.execute("SELECT * FROM signals WHERE model = 'A'").df()
        px = con.execute("SELECT * FROM prices").df()
    finally:
        con.close()
    cont = F.continuous(px)
    sig["vintage_date"] = pd.to_datetime(sig["vintage_date"]).astype("datetime64[ns]")
    costs = CostModel()

    rows, trade_log = [], []
    for period, (start, end) in PERIODS.items():
        for T in THRESHOLDS:
            port = []
            for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
                idx = FV.roll_adjusted_index(cont, sym)
                c = cont[cont.symbol == sym].set_index("trade_date")["expiry_1"]
                c.index = pd.DatetimeIndex(c.index).as_unit("ns")
                r = run_product(sig[sig.grain_class == cls], idx, c, T, costs, start, end)
                rows.append({"period": period, "T": T, "product": sym, **stats(r["daily"], r["trades"])})
                trade_log.append(r["trades"].assign(period=period, T=T, product=sym))
                port.append(r["daily"])
            both = pd.concat(port, axis=1).fillna(0).mean(axis=1)       # equal notional, both books
            all_t = pd.concat(trade_log[-2:], ignore_index=True)
            rows.append({"period": period, "T": T, "product": "WMAZ+YMAZ", **stats(both, all_t)})
    res = pd.DataFrame(rows)
    log = pd.concat(trade_log, ignore_index=True).drop(columns=["i0", "i1", "dir", "cost"])
    res.to_csv(OUT, index=False)
    return {"summary": res, "trades": log}


if __name__ == "__main__":
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    out = main()
    print(out["summary"].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print(out["trades"].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
