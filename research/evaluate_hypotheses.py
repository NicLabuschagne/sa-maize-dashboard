"""Evaluate the pre-registered signal hypotheses (see PREREGISTRATION.md). Run once.

    python research/evaluate_hypotheses.py

Every signal is point-in-time: model fits are expanding and end the release before, seasonal norms
use prior seasons only, and the trend reading is taken at the release-day close, before entry.
Signals are oriented so that a positive value means "expect the price to rise": a good IC is > 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import fairvalue as FV  # noqa: E402
from app.data import features as F  # noqa: E402
from app.data import trend as T  # noqa: E402
from config import DB_PATH  # noqa: E402

SPLIT = pd.Timestamp("2020-05-01")
TARGET = "fwd_10d"
HUBER_C = 1.345
OUT = Path(__file__).with_name("results_hypotheses.csv")


# ----------------------------------------------------------------------------- signal builders
def huber_beta(X: np.ndarray, y: np.ndarray, c: float = HUBER_C, iters: int = 50) -> np.ndarray:
    """Huber M-estimate by iteratively reweighted least squares, MAD scale."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        r = y - X @ beta
        s = np.median(np.abs(r - np.median(r))) / 0.6745 or 1e-9
        u = np.abs(r / s)
        w = np.where(u <= c, 1.0, c / u)
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        if np.allclose(new, beta, atol=1e-10):
            break
        beta = new
    return beta


def huber_z(p: pd.DataFrame) -> pd.Series:
    """Model A refitted with a robust loss, same expanding scheme and standardisation as OLS."""
    q = p.dropna(subset=["y", "x"]).reset_index(drop=True)
    X, y = FV._design(q, []), q["y"].to_numpy()
    fv = np.full(len(q), np.nan)
    for t in range(FV.MIN_OBS, len(q)):
        fv[t] = X[t] @ huber_beta(X[:t], y[:t])
    resid = pd.Series(y - fv)
    z = resid / resid.expanding(min_periods=12).std().shift(1)
    return pd.Series(z.to_numpy(), index=q["vintage_date"].to_numpy())


def cover_news(sd: pd.DataFrame, grain_class: str, min_prior: int = 4) -> pd.Series:
    """This release's change in log cover, less the average change for that marketing-year month
    in prior seasons. Uses the full S&D history from 2002, not just the priced era."""
    s = sd[sd.grain_class == grain_class].dropna(subset=["months_cover"]).sort_values("vintage_date")
    s = s[s.months_cover > 0].reset_index(drop=True)
    dlc = np.log(s.months_cover).diff().to_numpy()
    mm = s.my_month.to_numpy()
    news = np.full(len(s), np.nan)
    for i in range(len(s)):
        prior = dlc[:i][(mm[:i] == mm[i])]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= min_prior and np.isfinite(dlc[i]):
            news[i] = dlc[i] - prior.mean()
    return pd.Series(news, index=pd.to_datetime(s.vintage_date).to_numpy())


def trend_at(agg: pd.Series, dates: pd.Series) -> np.ndarray:
    """Trend position standardised by its own expanding sd, read at the release-day close."""
    tz = (agg / agg.expanding(min_periods=252).std()).dropna()
    tz.index = pd.DatetimeIndex(tz.index).as_unit("ns")
    left = pd.DataFrame({"d": pd.DatetimeIndex(dates).as_unit("ns")})
    right = pd.DataFrame({"t": tz.index, "v": tz.to_numpy()})
    return pd.merge_asof(left, right, left_on="d", right_on="t", direction="backward")["v"].to_numpy()


def signals_for(cls: str, sym: str, sd, cont, cpi, snap) -> pd.DataFrame:
    A = FV.fit_expanding(FV.panel_price(sd, cont, cpi, cls, sym)).panel
    A["vintage_date"] = pd.to_datetime(A["vintage_date"])
    D = FV.fit_expanding(FV.panel_parity(sd, cont, snap, cpi, cls, sym)).panel
    zD = pd.Series(D["z"].to_numpy(), index=pd.to_datetime(D["vintage_date"]))
    news = cover_news(sd, cls)
    hz = huber_z(FV.panel_price(sd, cont, cpi, cls, sym))
    agg = T.aggregate(T.trend_panel(FV.roll_adjusted_index(cont, sym)))

    out = pd.DataFrame({"vintage_date": A["vintage_date"], "fwd_5d": A["fwd_5d"],
                        "fwd_10d": A["fwd_10d"], "z_A": A["z"]})
    out["S0"] = -out["z_A"]
    out["H1"] = -news.reindex(out.vintage_date).to_numpy()
    zd = zD.reindex(out.vintage_date).to_numpy()
    out["H2"] = np.where(np.isfinite(zd), (-out["z_A"] - zd) / 2, np.nan)
    out["H3"] = -out["z_A"] + trend_at(agg, out.vintage_date)
    out["H4"] = -hz.reindex(out.vintage_date).to_numpy()
    return out


# ----------------------------------------------------------------------------- scoring
def ic(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return np.nan, int(m.sum())
    return float(stats.spearmanr(a[m], b[m]).statistic), int(m.sum())


def boot_p(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return FV._block_bootstrap_p(a[m].to_numpy(), b[m].to_numpy(), block=6, n_boot=4000)


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def main() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
        snap = con.execute("SELECT * FROM macro_snap").df()
    finally:
        con.close()
    sd, cont = F.sd_monthly(bs), F.continuous(px)

    rows, sig = [], {}
    for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
        s = signals_for(cls, sym, sd, cont, cpi, snap)
        sig[cls] = s
        dev, hold = s[s.vintage_date < SPLIT], s[s.vintage_date >= SPLIT]
        for h in ("H1", "H2", "H3", "H4"):
            same = hold[h].notna()                      # baseline scored on the same releases
            rows.append({"product": sym, "hypothesis": h,
                         "dev_ic": ic(dev[h], dev[TARGET])[0],
                         "hold_ic": ic(hold[h], hold[TARGET])[0],
                         "hold_n": ic(hold[h], hold[TARGET])[1],
                         "baseline_hold_ic": ic(hold.loc[same, "S0"], hold.loc[same, TARGET])[0],
                         "hold_ic_5d": ic(hold[h], hold["fwd_5d"])[0],
                         "p_raw": boot_p(hold[h], hold[TARGET])})

    # H5: relative value, white minus yellow
    w, y = sig["white"], sig["yellow"]
    pair = w[["vintage_date", "z_A", TARGET]].merge(y[["vintage_date", "z_A", TARGET]],
                                                     on="vintage_date", suffixes=("_w", "_y"))
    pair["H5"] = pair["z_A_y"] - pair["z_A_w"]
    pair["tgt"] = pair[f"{TARGET}_w"] - pair[f"{TARGET}_y"]
    pdev, phold = pair[pair.vintage_date < SPLIT], pair[pair.vintage_date >= SPLIT]
    # The pre-registration did not name a baseline for the pair. Interpretation fixed here, before
    # reading the result: the mean of the two outright baselines' holdout ICs.
    base_pair = np.mean([ic(s_[s_.vintage_date >= SPLIT]["S0"], s_[s_.vintage_date >= SPLIT][TARGET])[0]
                         for s_ in (w, y)])
    rows.append({"product": "WMAZ-YMAZ", "hypothesis": "H5",
                 "dev_ic": ic(pdev["H5"], pdev["tgt"])[0],
                 "hold_ic": ic(phold["H5"], phold["tgt"])[0], "hold_n": ic(phold["H5"], phold["tgt"])[1],
                 "baseline_hold_ic": base_pair, "hold_ic_5d": np.nan,
                 "p_raw": boot_p(phold["H5"], phold["tgt"])})

    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p_raw"].to_numpy())
    r["beats_baseline"] = r["hold_ic"] > r["baseline_hold_ic"]
    r["significant"] = r["p_holm"] < 0.05
    r["stable_sign"] = np.sign(r["dev_ic"]) == np.sign(r["hold_ic"])
    r["PASS"] = r.beats_baseline & r.significant & r.stable_sign

    base = []
    for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
        s_ = sig[cls]
        base.append({"product": sym, "hypothesis": "S0 baseline",
                     "dev_ic": ic(s_[s_.vintage_date < SPLIT]["S0"], s_[s_.vintage_date < SPLIT][TARGET])[0],
                     "hold_ic": ic(s_[s_.vintage_date >= SPLIT]["S0"], s_[s_.vintage_date >= SPLIT][TARGET])[0],
                     "hold_n": ic(s_[s_.vintage_date >= SPLIT]["S0"], s_[s_.vintage_date >= SPLIT][TARGET])[1],
                     "p_raw": boot_p(s_[s_.vintage_date >= SPLIT]["S0"], s_[s_.vintage_date >= SPLIT][TARGET])})
    out = pd.concat([pd.DataFrame(base), r], ignore_index=True)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res = main()
    cols = ["product", "hypothesis", "dev_ic", "hold_ic", "baseline_hold_ic", "hold_n",
            "hold_ic_5d", "p_raw", "p_holm", "beats_baseline", "significant", "stable_sign", "PASS"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
