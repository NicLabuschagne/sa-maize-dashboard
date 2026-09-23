"""Development-only: the same Model A and B fair values scored weekly between releases rather than on
release day. Tests whether more observations (breadth) help. Holdout rows are never scored.

    python research/explore_sampling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb, numpy as np, pandas as pd
from config import DB_PATH
from app.data import fairvalue as FV, features as F
from evaluate_hypotheses import ic, boot_p
SPLIT = pd.Timestamp("2020-05-01")
con = duckdb.connect(str(DB_PATH), read_only=True)
bs, px, cpi, snap = (con.execute(f"select * from {t}").df() for t in ("balance_sheet", "prices", "macro", "macro_snap"))
sd, cont = F.sd_monthly(bs), F.continuous(px)

def daily_eval(fit, level: pd.Series, fwd: pd.Series, label):
    q = fit.panel.dropna(subset=["fv"]).copy()
    q["sd"] = q["resid"].expanding(min_periods=12).std().shift(1)
    # after release t, FV and scale are fixed until the next release; entry lag 1 day as before
    st = q[["vintage_date", "fv", "sd"]].assign(vintage_date=lambda x: pd.to_datetime(x.vintage_date).astype("datetime64[ns]"))
    d = pd.DataFrame({"date": level.index.astype("datetime64[ns]"), "y": level.to_numpy(), "fwd": fwd.to_numpy()})
    d = pd.merge_asof(d, st.assign(avail=st.vintage_date + pd.Timedelta(days=1)).sort_values("avail"),
                      left_on="date", right_on="avail", direction="backward")
    d["s"] = -(d.y - d.fv) / d.sd
    d = d.dropna(subset=["s", "fwd"])
    wk = d.iloc[::5]
    dev = wk[wk.date < SPLIT]
    ev = q[pd.to_datetime(q.vintage_date) < SPLIT]
    ev_ic, n_ev = ic(-ev["z"], ev["fwd_10d"])
    w_ic, n_w = ic(dev["s"], dev["fwd"])
    print(f"{label:22s} release-day IC {ev_ic:+.3f} (n={n_ev}, p={boot_p(-ev['z'], ev['fwd_10d']):.3f})   "
          f"weekly IC {w_ic:+.3f} (n={n_w}, p={boot_p(dev['s'], dev['fwd']):.3f})")

for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
    fa = FV.fit_expanding(FV.panel_price(sd, cont, cpi, cls, sym))
    c = cont[cont.symbol == sym].sort_values("trade_date").set_index("trade_date")
    dates = pd.Series(c.index, index=c.index)
    real = FV.real_price(c["close_1"].reset_index(drop=True), dates.reset_index(drop=True), cpi)
    lvl = pd.Series(np.log(real.to_numpy()), index=pd.DatetimeIndex(c.index))
    idx = FV.roll_adjusted_index(cont, sym)
    f10 = pd.Series(np.log(idx.shift(-11) / idx.shift(-1)).reindex(lvl.index).to_numpy(), index=lvl.index)
    daily_eval(fa, lvl, f10, f"A outright {sym}")
    fb = FV.fit_expanding(FV.panel_spread(sd, cont, cls, sym))
    sp = c["spread_2_1_pct_ann"]
    fs = (sp.shift(-11) - sp.shift(-1))
    daily_eval(fb, pd.Series(sp.to_numpy(), index=lvl.index), pd.Series(fs.to_numpy(), index=lvl.index), f"B spread {sym}")
