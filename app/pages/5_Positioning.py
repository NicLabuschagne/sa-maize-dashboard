import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import fairvalue as FV
from app.data import trend as T
from app.data.warehouse import load_cot, load_macro
from app.state import derived, sidebar

st.set_page_config(page_title="Positioning", layout="wide")
sel = sidebar()
cls, sym = sel["grain_class"], sel["symbol"]
st.title("Positioning — trend-model overlay")

if cls == "total":
    st.info("Pick **white** or **yellow** in the sidebar.")
    st.stop()

st.warning(
    "**This is a model of what a trend follower would hold, not a measurement of what anyone does hold.** "
    "For CBOT we can check it against the CFTC's Commitments of Traders and quote a number. The JSE "
    "publishes open interest but no breakdown by trader category, so the SAFEX version is an "
    "unvalidated port — and the tests at the bottom of this page find no trend-following footprint "
    "in SAFEX turnover at all. Use it as a risk overlay: where systematic momentum sits relative to "
    "your fundamental view. Do not use it as a flow estimate, and do not feed it into fair value.",
    icon="⚠️")

D = derived()


@st.cache_data
def safex_trend(symbol: str) -> tuple[pd.Series, pd.DataFrame]:
    idx = FV.roll_adjusted_index(derived()["cont"], symbol)
    panel = T.trend_panel(idx)
    return idx, panel


@st.cache_data
def corn_validation() -> dict:
    m = load_macro()
    m = m[m.series == "cbot_corn"].sort_values("date")
    if m.empty:
        return {}
    px = pd.Series(m.value.to_numpy(), index=pd.DatetimeIndex(m.date).as_unit("ns"))
    panel = T.trend_panel(px)
    agg = T.aggregate(panel)
    cot = load_cot()
    cot = cot[cot.symbol == "ZC"]
    if cot.empty:
        return {}
    out = T.validate_against_cot(agg, cot)
    per = {}
    c = cot.dropna(subset=["net_noncomm_pct_oi"]).copy()
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    for col in panel.columns:
        s = panel[col].dropna()
        j = pd.merge_asof(c.sort_values("date"),
                          pd.DataFrame({"d": pd.DatetimeIndex(s.index).as_unit("ns"),
                                        "v": s.to_numpy()}).sort_values("d"),
                          left_on="date", right_on="d", direction="backward",
                          tolerance=pd.Timedelta("5D")).dropna(subset=["v"])
        per[col] = j["v"].corr(j.net_noncomm_pct_oi)
    out["per_signal"] = per
    return out


idx, panel = safex_trend(sym)
agg = T.aggregate(panel)
disp = T.dispersion(panel).dropna()
last = T.latest_state(panel)
d = disp.iloc[-1]
now = float(agg.dropna().iloc[-1])

# ---------------------------------------------------------------- 1. current reading
st.markdown("#### Current reading")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Aggregate trend", f"{now:+.2f}",
          "net long" if now > 0.1 else "net short" if now < -0.1 else "flat",
          delta_color="off")
k2.metric("Signals long / short", f"{int(d.n_long)} / {int(d.n_short)}",
          f"{int(d.n_flat)} flat of {int(d.n)}")
k3.metric("Agreement", f"{d.agreement:+.0%}",
          help="(long − short) ÷ total. Near ±100% means trend money is aligned and a reversal "
               "has fuel behind it; near zero means no consensus to squeeze.")
k4.metric("As of", f"{pd.Timestamp(agg.dropna().index[-1]):%d %b %Y}")

c1, c2 = st.columns([1, 1])
with c1:
    colours = [P.STATUS["bad"] if v < 0 else P.STATUS["good"] for v in last.position]
    fig = go.Figure(go.Bar(x=last.position, y=last.signal, orientation="h",
                           marker=dict(color=colours),
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, "Constituent signals — the dispersion is the uncertainty estimate",
             xtitle="position (−1 short … +1 long)", height=360)
    fig.update_xaxes(range=[-1.05, 1.05], tickformat="+.1f")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    P.line(fig, agg.index, agg, "aggregate trend", entity=cls, hover="%{y:+.2f}<extra></extra>")
    for lvl, colr in ((0.5, P.STATUS["good"]), (-0.5, P.STATUS["bad"])):
        fig.add_hline(y=lvl, line=dict(color=colr, width=1, dash="dot"), opacity=0.5)
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, f"{sym} aggregate trend position", ytitle="−1 … +1", height=360)
    fig.update_yaxes(range=[-1.05, 1.05], tickformat="+.1f")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 2. validation on corn
st.markdown("---")
st.markdown("#### Does the method recover real positioning? Validated on CBOT corn")
V = corn_validation()
if not V or not V.get("ok"):
    st.info("Run `python ingest/fetch_lse.py` with LSE_API_KEY set to fetch COT and CBOT corn.")
else:
    st.caption("Corn is the market where a positioning report exists, so it is where the method can "
               "be checked. Model sampled on the same Tuesday the COT positions are measured — this "
               "asks whether the model *describes* positioning, not whether it predicts anything.")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Correlation, levels", f"{V['corr_level']:+.2f}", f"Spearman {V['corr_level_spearman']:+.2f}")
    v2.metric("Correlation, weekly change", f"{V['corr_change']:+.2f}",
              help="Harder test than levels: does the model turn when real money turns?")
    v3.metric("Weeks", V["n"])
    v4.metric("Period", f"{pd.Timestamp(V['first']):%b %Y} – {pd.Timestamp(V['last']):%b %Y}")

    m = V["merged"]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        zt = (m.trend - m.trend.mean()) / m.trend.std()
        zc = (m.net_noncomm_pct_oi - m.net_noncomm_pct_oi.mean()) / m.net_noncomm_pct_oi.std()
        P.line(fig, m.date, zt, "trend model", entity="white", hover="%{y:+.2f}<extra>model</extra>")
        P.line(fig, m.date, zc, "CFTC non-commercial net", entity="yellow",
               hover="%{y:+.2f}<extra>COT</extra>")
        fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
        P.layout(fig, "CBOT corn: model vs reported positioning (both standardised)",
                 ytitle="standard deviations", height=340)
        st.plotly_chart(fig, width="stretch")
    with c2:
        per = pd.Series(V["per_signal"], name="corr with COT").sort_values(ascending=False)
        st.dataframe(per.to_frame().style.format("{:+.3f}"), width="stretch", height=310)
        st.caption("Slow signals track reported positioning far better than fast ones — real managed "
                   "money in corn is slow, which is what capacity constraints would predict. The "
                   "blend is kept equal-weight rather than tuned to this, so the port to SAFEX is "
                   "not fitted to corn.")

# ---------------------------------------------------------------- 3. does it port?
st.markdown("---")
st.markdown("#### Does it port to SAFEX? The honest answer")


@st.cache_data
def safex_footprint(symbol: str) -> dict:
    from scipy import stats as sps

    px = derived()["prices"]
    agg_ = T.aggregate(T.trend_panel(FV.roll_adjusted_index(derived()["cont"], symbol))).dropna()
    w = px[px.symbol == symbol].groupby("trade_date")[["volume", "open_interest"]].sum()
    w.index = pd.DatetimeIndex(w.index).as_unit("ns")
    j = pd.DataFrame({"trend": agg_}).join(w, how="inner").dropna()
    j["lv"] = np.log(j.volume.replace(0, np.nan))
    j["lv_ds"] = j.lv - j.groupby(j.index.month).lv.transform("mean")
    j["d_trend"] = j.trend.diff().abs()
    j["flip"] = (np.sign(j.trend) != np.sign(j.trend.shift())).astype(int)
    k = j.dropna(subset=["lv_ds", "d_trend"])
    flips, others = k[k.flip == 1].lv_ds, k[k.flip == 0].lv_ds
    t = sps.ttest_ind(flips.dropna(), others.dropna(), equal_var=False)
    return {"n": len(k), "n_flips": int(k.flip.sum()),
            "rho_vol": float(sps.spearmanr(k.d_trend, k.lv_ds).statistic),
            "rho_oi": float(sps.spearmanr(k.d_trend, np.log(k.open_interest)).statistic),
            "vol_uplift": float(np.exp(flips.mean()) / np.exp(others.mean()) - 1),
            "t": float(t.statistic), "p": float(t.pvalue)}


F_ = safex_footprint(sym)
f1, f2, f3, f4 = st.columns(4)
f1.metric("Turnover on signal-flip days", f"{F_['vol_uplift']:+.1%}",
          f"p = {F_['p']:.2f}", delta_color="off",
          help="Volume on days the aggregate signal changes sign, versus every other day, "
               "with month-of-year effects removed.")
f2.metric("|Δ trend| vs volume", f"{F_['rho_vol']:+.3f}", "Spearman", delta_color="off")
f3.metric("|Δ trend| vs open interest", f"{F_['rho_oi']:+.3f}", "Spearman", delta_color="off")
f4.metric("Sign flips observed", F_["n_flips"], f"over {F_['n']:,} days")

st.error(
    f"**No detectable trend-following footprint in {sym}.** Over {F_['n_flips']} sign flips in "
    f"{F_['n']:,} sessions, turnover on flip days differs from other days by {F_['vol_uplift']:+.1%} "
    f"(p = {F_['p']:.2f}), and the correlation between signal turnover and either volume or open "
    f"interest is near zero. If systematic trend money were a material share of this market, some "
    f"trace should appear here.", icon="🔎")

st.markdown(f"""
**What that does and does not mean.** SAFEX maize turns over about **R1.4bn a day in white and
R0.9bn in yellow** — roughly a thirtieth of CBOT corn. The flow is dominated by the physical trade:
co-ops, millers, exporters and the banks hedging them. So the most likely reading is that trend
followers are a small enough share to be invisible.

The tests are crude, though, and I would not overstate them. Volume and open interest net across
every participant, and a signal flip closes longs while opening shorts — open interest can stay
flat through exactly the trade we are looking for. A CTA share of a few percent would not show up.

**The test that would actually settle it** is the one you cannot run from here: JSE broker codes.
Positioning by broker, aggregated to a category split, is a genuine SAFEX COT proxy — and building
that is the first thing to do with desk data.

**Until then, treat the {sym} reading above as a hypothetical:** what a canonical trend follower
*would* hold, useful for asking whether your fundamental view is aligned with or against systematic
momentum. That is a legitimate risk overlay. It is not a flow forecast.
""")

with st.expander("Method"):
    st.markdown(f"""
- **Signals** — {len(T.CROSSOVERS)} EMA crossovers {', '.join(f'{s}/{l}' for s, l in T.CROSSOVERS)} and
  {len(T.MOMENTUM)} time-series momentum lookbacks ({', '.join(f'{n}d' for n in T.MOMENTUM)}).
- **Scaling** — crossovers are normalised by price volatility over the slow window, momentum by the
  volatility of a move of that length; both are then standardised against their own rolling history
  and squashed with `tanh` into [−1, +1]. Sign is direction, magnitude is conviction.
- **Blend** — equal weight. Deliberately *not* tuned to the corn COT fit, so the port to SAFEX
  inherits no corn-specific calibration.
- **Prices** — the roll-adjusted front-month index for SAFEX, CBOT daily settles for corn.
- **Validation** — model sampled on the COT Tuesday, so the comparison is contemporaneous. A
  predictive claim would need the Friday release date instead, which is stored alongside.
""")
