"""Development-sample exploration of model changes for outright, spread and parity.

    python research/explore_dev.py

Scores releases before 2020-05-01 ONLY. The holdout is never scored here: it is reserved for the
finalists this script selects, which are then pre-registered and tested once.

Selection rule, fixed before the first run: per strategy type, the candidate with the highest
mean development IC goes forward if that mean beats the baseline's by at least 0.05 and it does
not fall below the baseline on any product. At most one finalist per strategy type.
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
from config import DB_PATH  # noqa: E402
from evaluate_hypotheses import ic  # noqa: E402

SPLIT = pd.Timestamp("2020-05-01")
MIN_GAIN = 0.05
PAR_TOL = pd.Timedelta("21D")
OUT = Path(__file__).with_name("results_explore_dev.csv")


def attach_parity(p: pd.DataFrame, par: pd.DataFrame) -> pd.DataFrame:
    """Latest parity calculation published by each release date."""
    cols = ["available_date", "export_randfontein", "import_randfontein", "prime_rate"]
    q = par[cols].dropna(subset=["export_randfontein", "import_randfontein"]).copy()
    q["available_date"] = q["available_date"].astype("datetime64[ns]")
    p = p.copy()
    p["vintage_date"] = pd.to_datetime(p["vintage_date"]).astype("datetime64[ns]")
    out = pd.merge_asof(p.sort_values("vintage_date"), q.sort_values("available_date"),
                        left_on="vintage_date", right_on="available_date", direction="backward",
                        tolerance=PAR_TOL)
    return out.reset_index(drop=True)


def signal(p: pd.DataFrame, y: pd.Series, extra: list[str] | None = None) -> pd.DataFrame:
    q = p.assign(y=y)
    if extra and "x2" in extra:
        q["x2"] = q["x"] ** 2
    fit = FV.fit_expanding(q, extra=extra).panel
    return pd.DataFrame({"vintage_date": pd.to_datetime(fit["vintage_date"]), "s": -fit["z"],
                         "fwd_5d": fit["fwd_5d"], "fwd_10d": fit["fwd_10d"], "fwd_1m": fit["fwd_1m"]})


def band_pos(p: pd.DataFrame) -> pd.Series:
    return (p["close_1"] - p["export_randfontein"]) / (p["import_randfontein"] - p["export_randfontein"])


def candidates(sd, cont, cpi, snap, par) -> list[dict]:
    rows = []
    for cls, sym in (("white", "WMAZ"), ("yellow", "YMAZ")):
        # ---- outright: target = SAFEX log return
        a = attach_parity(FV.panel_price(sd, cont, cpi, cls, sym), par)
        mid = np.sqrt(a["export_randfontein"] * a["import_randfontein"])
        for name, y, extra in (("A0 baseline", a["y"], None),
                               ("A band position", band_pos(a), None),
                               ("A basis to band mid", np.log(a["close_1"] / mid), None),
                               ("A basis to export parity", np.log(a["close_1"] / a["export_randfontein"]), None),
                               ("A convex cover (x + x²)", a["y"], ["x2"])):
            rows.append({"type": "outright", "product": sym, "candidate": name, "sig": signal(a, y, extra)})

        # ---- calendar spread: target = change in annualised spread
        b = attach_parity(FV.panel_spread(sd, cont, cls, sym), par)
        for name, y, extra in (("B0 baseline", b["y"], None),
                               ("B carry-adjusted (spread − prime)", b["y"] - b["prime_rate"], None),
                               ("B convex cover (x + x²)", b["y"], ["x2"])):
            rows.append({"type": "spread", "product": sym, "candidate": name, "sig": signal(b, y, extra)})

    # ---- parity: YMAZ, target = SAFEX minus CBOT×ZAR return (the tradeable hedge)
    d = attach_parity(FV.panel_parity(sd, cont, snap, cpi, "yellow", "YMAZ"), par)
    mid = np.sqrt(d["export_randfontein"] * d["import_randfontein"])
    for name, y in (("D0 baseline", d["y"]),
                    ("D band position", band_pos(d)),
                    ("D basis to band mid", np.log(d["close_1"] / mid)),
                    ("D basis to export parity", np.log(d["close_1"] / d["export_randfontein"]))):
        rows.append({"type": "parity", "product": "YMAZ", "candidate": name, "sig": signal(d, y)})
    return rows


def main() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
        snap = con.execute("SELECT * FROM macro_snap").df()
        par = con.execute("SELECT * FROM sagis_parity").df()
    finally:
        con.close()
    sd, cont = F.sd_monthly(bs), F.continuous(px)

    rows = []
    for c in candidates(sd, cont, cpi, snap, par):
        dev = c["sig"][c["sig"].vintage_date < SPLIT]          # holdout rows dropped before scoring
        rows.append({"type": c["type"], "product": c["product"], "candidate": c["candidate"],
                     "dev_n": ic(dev["s"], dev["fwd_10d"])[1],
                     "dev_ic_5d": ic(dev["s"], dev["fwd_5d"])[0],
                     "dev_ic_10d": ic(dev["s"], dev["fwd_10d"])[0],
                     "dev_ic_1m": ic(dev["s"], dev["fwd_1m"])[0]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)

    finalists = []
    for t, g in r.groupby("type"):
        base = g[g.candidate.str.contains("baseline")].set_index("product")["dev_ic_10d"]
        alt = g[~g.candidate.str.contains("baseline")]
        m = alt.groupby("candidate").dev_ic_10d.mean().sort_values(ascending=False)
        if m.empty:
            continue
        best = m.index[0]
        per = alt[alt.candidate == best].set_index("product")["dev_ic_10d"]
        ok = (m.iloc[0] >= base.mean() + MIN_GAIN) and (per >= base.reindex(per.index)).all()
        finalists.append({"type": t, "best": best, "mean_dev_ic": m.iloc[0], "baseline_mean": base.mean(),
                          "selected": bool(ok)})
    return {"table": r, "finalists": pd.DataFrame(finalists)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res = main()
    print(res["table"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print()
    print(res["finalists"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
