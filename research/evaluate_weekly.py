"""Evaluate the pre-registered weekly-data hypotheses (see PREREGISTRATION_WEEKLY.md). Run once.

    python research/evaluate_weekly.py

Weekly data enters only through rows whose `available_date` is on or before the release date, so
every nowcast and pace reading is what could have been computed that morning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.data import fairvalue as FV  # noqa: E402
from app.data import features as F  # noqa: E402
from app.data import weekly as W  # noqa: E402
from config import DB_PATH  # noqa: E402
from evaluate_hypotheses import boot_p, holm, ic  # noqa: E402

SPLIT = pd.Timestamp("2020-05-01")
TARGET = "fwd_10d"
DAYS_PER_MONTH = 30.44
N_PRIOR, MIN_PRIOR = 5, 3
OUT = Path(__file__).with_name("results_weekly.csv")
SIGN = {"A": 1.0, "B": -1.0, "C": 1.0, "D": 1.0}


# ----------------------------------------------------------------------------- weekly features
def nowcast_sd(sd: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    """sd with months_cover replaced by the W1 nowcast; `n_weeks` records how many weeks went in."""
    out = sd.copy()
    wk = wk.sort_values("week_end")
    stock, nweeks = [], []
    for r in out.itertuples():
        month_end = r.latest_month + pd.offsets.MonthEnd(0)
        c = wk[(wk.grain_class == r.grain_class) & (wk.week_end > month_end)]
        weeks = c[(c.flow == "exports") & (c.available_date <= r.vintage_date)].week_end.unique()
        c = c[c.week_end.isin(weeks)]
        if len(weeks) and not (c.flow == "deliveries").any():
            stock.append(np.nan)                     # trade without deliveries would bias the nowcast
            nweeks.append(len(weeks))
            continue
        flow = c.groupby("flow").tons_week.sum()
        draw = r.utilisation_12m / 12 * len(weeks) * 7 / DAYS_PER_MONTH
        stock.append(r.closing_stock + flow.get("deliveries", 0.0) + flow.get("imports", 0.0)
                     - flow.get("exports", 0.0) - draw)
        nweeks.append(len(weeks))
    out["closing_stock_nc"] = stock
    out["n_weeks"] = nweeks
    out["months_cover"] = out["closing_stock_nc"] / (out["disappearance_12m"] / 12)
    return out


def export_pace(sd: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    """W2 raw pace per (vintage_date, grain_class): season-to-date exports vs prior-season mean at the
    same week, over trailing-12m disappearance."""
    rows = []
    for cls in ("white", "yellow"):
        ex = wk[(wk.flow == "exports") & (wk.grain_class == cls)].sort_values("week_end")
        cum = W.season_to_date(ex, "exports", cls)
        seasons = list(cum.columns)
        for r in sd[sd.grain_class == cls].itertuples():
            seen = ex[ex.available_date <= r.vintage_date]
            if seen.empty:
                continue
            last = seen.iloc[-1]
            s, k = last.season, int(last.week)
            i = seasons.index(s)
            std = seen[seen.season == s].tons_week.sum()
            prior = cum.loc[:k, seasons[max(0, i - N_PRIOR):i]].iloc[-1].dropna() if i else pd.Series(dtype=float)
            pace = (std - prior.mean()) / r.disappearance_12m if len(prior) >= MIN_PRIOR else np.nan
            rows.append({"vintage_date": r.vintage_date, "grain_class": cls, "pace": pace})
    return pd.DataFrame(rows)


def standardise(s: pd.Series) -> pd.Series:
    return s / s.expanding(min_periods=12).std().shift(1)


# ----------------------------------------------------------------------------- per-model signals
def model_panels(sd: pd.DataFrame, cont, wy, bs, cpi, snap) -> dict[str, pd.DataFrame]:
    """Fitted panels for the six model-products, keyed like 'A:WMAZ'."""
    out = {}
    for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
        out[f"A:{sym}"] = FV.fit_expanding(FV.panel_price(sd, cont, cpi, cls, sym)).panel
        out[f"B:{sym}"] = FV.fit_expanding(FV.panel_spread(sd, cont, cls, sym)).panel
    out["C:WMAZ-YMAZ"] = FV.fit_expanding(FV.panel_white_yellow(sd, wy, bs)).panel
    out["D:YMAZ"] = FV.fit_expanding(FV.panel_parity(sd, cont, snap, cpi, "yellow", "YMAZ")).panel
    return out


def pace_for(key: str, vintages: pd.Series, pace: pd.DataFrame) -> np.ndarray:
    pv = pace.pivot_table(index="vintage_date", columns="grain_class", values="pace")
    pv.index = pd.DatetimeIndex(pv.index).as_unit("ns")
    v = pd.DatetimeIndex(vintages).as_unit("ns")
    if key.startswith("C:"):
        raw = pv["white"] - pv["yellow"]
    else:
        raw = pv["white" if key.endswith("WMAZ") else "yellow"]
    return raw.reindex(v).to_numpy()


def build(sd, sd_nc, pace, cont, wy, bs, cpi, snap) -> dict[str, pd.DataFrame]:
    base, nc = model_panels(sd, cont, wy, bs, cpi, snap), model_panels(sd_nc, cont, wy, bs, cpi, snap)
    out = {}
    for key, b in base.items():
        b = b.assign(vintage_date=pd.to_datetime(b["vintage_date"]).astype("datetime64[ns]"))
        d = pd.DataFrame({"vintage_date": b["vintage_date"], TARGET: b[TARGET], "S0": -b["z"]})
        n = nc[key].assign(vintage_date=pd.to_datetime(nc[key]["vintage_date"]).astype("datetime64[ns]"))
        d = d.merge(pd.DataFrame({"vintage_date": n["vintage_date"], "W1": -n["z"]}), on="vintage_date", how="left")
        d["pace"] = pace_for(key, d["vintage_date"], pace)
        d["pace_z"] = standardise(d["pace"])
        d["W2"] = d["S0"] + SIGN[key[0]] * d["pace_z"]
        out[key] = d
    return out


# ----------------------------------------------------------------------------- scoring
def main() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
        snap = con.execute("SELECT * FROM macro_snap").df()
        wk = W.with_total(con.execute("SELECT * FROM sagis_weekly").df())
    finally:
        con.close()
    sd, cont = F.sd_monthly(bs), F.continuous(px)
    wy = F.white_yellow_spread(cont)
    sd_nc = nowcast_sd(sd, wk)
    pace = export_pace(sd, wk)
    sig = build(sd, sd_nc, pace, cont, wy, bs, cpi, snap)

    rows = []
    for key, d in sig.items():
        dev, hold = d[d.vintage_date < SPLIT], d[d.vintage_date >= SPLIT]
        for h in ("W1", "W2"):
            same = hold[h].notna() & hold["S0"].notna()
            rows.append({"model": key, "hypothesis": h,
                         "dev_ic": ic(dev[h], dev[TARGET])[0],
                         "hold_ic": ic(hold[h], hold[TARGET])[0],
                         "hold_n": ic(hold[h], hold[TARGET])[1],
                         "baseline_hold_ic": ic(hold.loc[same, "S0"], hold.loc[same, TARGET])[0],
                         "baseline_dev_ic": ic(dev["S0"], dev[TARGET])[0],
                         "p_raw": boot_p(hold[h], hold[TARGET])})
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p_raw"].to_numpy())
    r["beats_baseline"] = r["hold_ic"] > r["baseline_hold_ic"]
    r["significant"] = r["p_holm"] < 0.05
    r["stable_sign"] = np.sign(r["dev_ic"]) == np.sign(r["hold_ic"])
    r["PASS"] = r.beats_baseline & r.significant & r.stable_sign

    # diagnostics, not part of the decision: how much the nowcast moved cover, how many weeks went in
    diag = sd_nc.dropna(subset=["months_cover"]).assign(
        shift_pct=lambda x: (x["closing_stock_nc"] / x["closing_stock"] - 1) * 100)
    r.attrs["diag"] = diag.groupby("grain_class")[["n_weeks", "shift_pct"]].describe().round(2)
    r.to_csv(OUT, index=False)
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    res = main()
    cols = ["model", "hypothesis", "dev_ic", "hold_ic", "baseline_hold_ic", "hold_n", "p_raw", "p_holm",
            "beats_baseline", "significant", "stable_sign", "PASS"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(res.attrs["diag"].to_string())
