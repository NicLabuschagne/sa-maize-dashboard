"""Diagnostics for threshold_fade.py: why z reverts (price vs fair value), and the same entries
with a fixed 10-trading-day hold.

    python research/threshold_fade_diagnostics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb, numpy as np, pandas as pd
from config import DB_PATH
from app.data import fairvalue as FV, features as F
from app.data.backtest import run_backtest, CostModel
import threshold_fade as TF
con = duckdb.connect(str(DB_PATH), read_only=True)
sig = con.execute("select * from signals where model='A'").df(); px = con.execute("select * from prices").df()
sig["vintage_date"] = pd.to_datetime(sig.vintage_date).astype("datetime64[ns]")
cont = F.continuous(px)
# 1. why does z come back? split the change in residual into price move and fair-value move
out = TF.main()["trades"]
rows = []
for r in out[(out["T"] == 1.0)].itertuples():
    s = sig[sig.grain_class == ("white" if r.product == "WMAZ" else "yellow")].set_index("vintage_date").sort_index()
    a = s.loc[pd.Timestamp(r.entry_release)]
    b = s[s.index > pd.Timestamp(r.entry_release)].iloc[r.releases_held - 1]
    rows.append({"period": r.period[:7], "product": r.product, "entry": r.entry_release, "exit": r.exit,
                 "z0": a.z, "z1": b.z, "d_price_pct": (b.actual - a.actual) * 100, "d_fair_pct": (b.fair_value - a.fair_value) * 100,
                 "net_pct": r.net_pct})
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(2).to_string(index=False))
rev = d[d.exit == "reverted"]
print("\nreverted trades: share of the residual's move that came from fair value moving toward price:",
      round(float((rev.d_fair_pct * np.sign(rev.z0)).sum() / ((rev.d_fair_pct - rev.d_price_pct) * np.sign(rev.z0)).sum()), 2))
# 2. same entries (|z|>=1 at a release), fixed 10-trading-day hold
print()
for period, (a_, b_) in TF.PERIODS.items():
    for T in (1.0, 1.5):
        res = []
        for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
            e = sig[(sig.grain_class == cls) & (sig.vintage_date >= a_) & (sig.vintage_date <= b_) & (sig.z.abs() >= T)]
            e = e.rename(columns={"front_close": "front_close"})[["vintage_date", "z", "front_close"]]
            r = run_backtest(FV.roll_adjusted_index(cont, sym), e, horizon=10, costs=CostModel())
            st = r.stats
            res.append(f"{sym}: n={st['n_trades']} hit={st['hit_rate']:.0%} avg={st['avg_net']*100:+.2f}% sharpe={st['sharpe']:+.2f}")
        print(f"{period[:12]:12s} T={T}  10-day hold  " + " | ".join(res))
