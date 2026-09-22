"""Run the fair-value models for one grain class and package everything the page needs.
Cached per class so each fit and bootstrap runs once per session."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import fairvalue as FV
from app.data.warehouse import load_macro, load_macro_snap
from app.state import CLASS_TO_SYMBOL, derived

STABILITY_SPLITS = {
    "all": lambda p: pd.Series(True, index=p.index),
    "2009–2016": lambda p: p.vintage_date < "2017-01-01",
    "2017–2026": lambda p: p.vintage_date >= "2017-01-01",
    "ex 2015/16 & 2016/17 seasons": lambda p: ~p.marketing_year.isin(["2015/16", "2016/17"]),
    "ex Feb–May (crop-estimate window)": lambda p: ~p.latest_month.dt.month.isin([1, 2, 3, 4]),
}


def _stability(q: pd.DataFrame, horizons: tuple[str, ...] = ("10d", "1m")) -> pd.DataFrame:
    rows = []
    for name, f in STABILITY_SPLITS.items():
        ic = FV.ic_table(q, {"z": "z"}, {h: FV.HORIZONS[h] for h in horizons}, mask=f(q))
        for _, r in ic.iterrows():
            rows.append({"sample": name, "horizon": r.horizon, "n": r.n, "IC": r.IC, "p": r.p})
    return pd.DataFrame(rows)


def _swap_forwards(q: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Copy of `q` whose fwd_* columns are taken from `{prefix}_*`, for scoring an alternative outcome."""
    out = q[["z"]].copy()
    for h in FV.HORIZONS:
        out[f"fwd_{h}"] = q[f"{prefix}_{h}"]
    return out


@st.cache_data(show_spinner="Fitting fair-value models…")
def run_models(grain_class: str) -> dict:
    D = derived()
    sd, cont, wy, bs = D["sd"], D["cont"], D["wy"], D["bs"]
    cpi = load_macro()
    snap = load_macro_snap()
    sym = CLASS_TO_SYMBOL[grain_class]
    out: dict = {}

    # Model A — outright real price vs cover ----------------------------------------------
    pa = FV.panel_price(sd, cont, cpi, grain_class, sym)
    fa = FV.fit_expanding(pa)
    q = fa.panel
    q["seasonal_1m"] = FV.seasonal_benchmark(q, "1m")
    q["neg_x"] = -q["x"]
    out["A"] = {"fit": fa, "ic": FV.ic_table(q, {"z (fair-value residual)": "z", "raw −log cover": "neg_x",
                                                 "seasonal mean (1m)": "seasonal_1m"}),
                "terciles": FV.tercile_table(q), "stability": _stability(q)}

    # Model B — calendar spread vs cover ---------------------------------------------------
    pb = FV.panel_spread(sd, cont, grain_class, sym)
    fb = FV.fit_expanding(pb)
    fb.panel["neg_x"] = -fb.panel["x"]
    out["B"] = {"fit": fb, "ic": FV.ic_table(fb.panel, {"z (spread residual)": "z", "raw −log cover": "neg_x"}),
                "terciles": FV.tercile_table(fb.panel), "stability": _stability(fb.panel)}

    # Model C — white premium (class-independent) ------------------------------------------
    pc = FV.panel_white_yellow(sd, wy, bs)
    fc = FV.fit_expanding(pc)
    fc_mix = FV.fit_expanding(pc, extra=["demand_mix"])
    out["C"] = {"fit": fc, "fit_mix": fc_mix,
                "ic": FV.ic_table(fc.panel, {"z (premium residual)": "z"}),
                "ic_mix": FV.ic_table(fc_mix.panel, {"z (with demand mix)": "z"}),
                "terciles": FV.tercile_table(fc.panel)}

    # Model D — import/export parity basis, scored as relative value ------------------------
    pdp = FV.panel_parity(sd, cont, snap, cpi, grain_class, sym)
    if len(pdp):
        fd = FV.fit_expanding(pdp)
        qd = fd.panel
        ic_rel = FV.ic_table(qd, {"basis z → SAFEX minus parity": "z"})
        ic_out = FV.ic_table(_swap_forwards(qd, "fwd_outright"), {"basis z → outright price": "z"})
        out["D"] = {"fit": fd, "ic": pd.concat([ic_rel, ic_out], ignore_index=True),
                    "terciles": FV.tercile_table(qd), "world": FV.world_parity(snap),
                    "stability": _stability(qd)}

        # Tested and rejected: Model A plus the contemporaneous world price ------------------
        paw = FV.panel_price_with_world(sd, cont, snap, cpi, grain_class, sym)
        fw_base, fw_world = FV.fit_expanding(paw), FV.fit_expanding(paw, extra=["lw"])
        out["A_world"] = {"base": fw_base, "with_world": fw_world,
                          "ic_base": FV.ic_table(fw_base.panel, {"cover only": "z"}),
                          "ic_world": FV.ic_table(fw_world.panel, {"cover + world price": "z"})}
    return out
