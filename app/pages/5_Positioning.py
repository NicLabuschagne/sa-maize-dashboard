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
from app.data.warehouse import load_macro
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
    cot = T.load_cot("ZC")
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
    out["price"] = px
    return out


@st.cache_data
def price_benchmark() -> dict:
    m = load_macro()
    m = m[m.series == "cbot_corn"].sort_values("date")
    if m.empty:
        return {}
    px = pd.Series(m.value.to_numpy(), index=pd.DatetimeIndex(m.date).as_unit("ns"))
    return T.price_benchmark(px, T.load_cot("ZC"))


@st.cache_data
def nowcast() -> dict:
    V_ = corn_validation()
    px = V_.get("price")
    if px is None:
        return {}
    agg_ = T.aggregate(T.trend_panel(px))
    cot = T.load_cot("ZC")
    nc = T.anchored_nowcast(agg_, cot)
    return {"nc": nc, "score": T.nowcast_scorecard(nc, cot)}


@st.cache_data
def cross_market() -> pd.DataFrame:
    m = load_macro()
    m = m[m.series == "cbot_corn"].sort_values("date")
    rows = {}
    if not m.empty:
        rows["CBOT corn"] = T.dynamics(pd.Series(m.value.to_numpy(),
                                                 index=pd.DatetimeIndex(m.date).as_unit("ns")))
    for s_ in ("WMAZ", "YMAZ"):
        rows[f"SAFEX {s_}"] = T.dynamics(FV.roll_adjusted_index(derived()["cont"], s_))
    return pd.DataFrame(rows).T


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

st.markdown("##### Position is the stock — flow is what actually gets traded")
fs = T.flow_state(agg)
g1, g2, g3, g4 = st.columns(4)
g1.metric("Implied flow, 1 week", f"{fs['flow'] * 100:+.0f}pp", fs["direction"], delta_color="off",
          help="Change in the modelled position over five sessions, in points of a full-size "
               "position. This is what a trend follower would have had to buy or sell.")
g2.metric("Implied flow, 1 month", f"{fs['flow_1m'] * 100:+.0f}pp")
g3.metric("Size of move vs history", f"{fs['pctile']:.0%}",
          help="Percentile of this week's absolute flow against every week in the sample.")
g4.metric("Position now", f"{fs['position'] * 100:+.0f}%")

ft = T.flow_table(agg)
px_w = idx.reindex(ft.index, method="ffill")
st.plotly_chart(P.price_position_flow(px_w, ft.position, ft.flow, cls,
                                      f"{sym} — price, modelled position and implied flow",
                                      price_label=f"{sym} front month", price_unit="R/t"),
                width="stretch")

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

    cpx = V.get("price")
    if cpx is not None:
        cagg = T.aggregate(T.trend_panel(cpx))
        cft = T.flow_table(cagg)
        st.plotly_chart(P.price_position_flow(cpx.reindex(cft.index, method="ffill"),
                                              cft.position, cft.flow, "aux",
                                              "CBOT corn — price, modelled position and implied flow",
                                              price_label="CBOT corn front", price_unit="$/bu"),
                        width="stretch")
        st.caption("Same three panels as SAFEX above, but here the middle panel can be checked "
                   "against the CFTC report — which is what the correlations above measure.")

    B = price_benchmark()
    if B.get("ok"):
        st.markdown("##### But a trend model is built from price — does it beat price alone?")
        st.caption("Tracking COT is not by itself evidence of anything: positioning follows price, "
                   "and so does any trend model. The question is whether the model adds information "
                   "beyond raw momentum.")
        b1, b2 = st.columns([2, 3])
        with b1:
            corr = pd.Series(B["corr"], name="corr with COT").sort_values(ascending=False)
            corr.index = corr.index.str.replace("trend", "TREND MODEL").str.replace("ret_", "price return ")
            st.dataframe(corr.to_frame().style.format("{:+.3f}"), width="stretch", height=250)
        with b2:
            r2 = pd.Series({"12-month price return alone": B["r2_ret12"],
                            "trend model alone": B["r2_trend"],
                            "all four return horizons": B["r2_rets"],
                            "trend model + all four returns": B["r2_both"]}, name="R² explaining COT")
            st.dataframe(r2.to_frame().style.format("{:.3f}"), width="stretch", height=180)
            st.markdown(
                f"Partial correlation of the trend model with COT, **after** controlling for all four "
                f"return horizons: **{B['partial_trend']:+.2f}**. It is not simply momentum relabelled.")
        st.info(
            f"**Where the model earns its keep is in the flow, not the direction.** On weekly *changes* "
            f"in positioning it scores **{B['dchg_trend']:+.2f}** against **{B['dchg_ret3m']:+.2f}** for a "
            f"raw 3-month return. The difference is the two features raw momentum lacks and real managed "
            f"futures have: position size **saturates** rather than scaling without limit, and it is "
            f"**volatility-scaled** rather than return-scaled. Those are what make it track the turn.",
            icon="📈")

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

st.markdown("##### The practical loop: nowcast daily, reconcile when the report lands")
N = nowcast()
if N and N.get("score", {}).get("ok"):
    sc, nc = N["score"], N["nc"]
    st.caption("The report gives a level as of Tuesday but is only public on Friday, so the official "
               "number is three to eight days stale at all times. Anchoring on the last *published* "
               "print and adding the model's implied flow since updates it daily without inventing "
               "the level — and it stays point-in-time, because the anchor only moves once the "
               "release date has passed.")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Anchored nowcast error", f"{sc['mae_anchored_sd']:.2f} sd",
              f"MAE {sc['mae_anchored']:.3f} of OI", delta_color="off",
              help="Mean absolute error against the level the next report actually published, "
                   "in standard deviations of reported positioning.")
    n2.metric("Model-only error", f"{sc['mae_model_only_sd']:.2f} sd",
              f"MAE {sc['mae_model_only']:.3f}", delta_color="off")
    n3.metric("Improvement from anchoring", f"{sc['improvement']:.0%}")
    n4.metric("Correlation with the print", f"{sc['corr_anchored']:+.3f}",
              f"model only {sc['corr_model_only']:+.3f}")

    e = sc["errors"]
    fig = go.Figure()
    P.line(fig, e.date, e.net_noncomm_pct_oi * 100, "reported (CFTC)", entity="yellow",
           hover="%{y:+.1f}%<extra>reported</extra>")
    P.line(fig, e.date, e.nowcast * 100, "anchored nowcast", entity="white",
           hover="%{y:+.1f}%<extra>anchored</extra>")
    P.line(fig, e.date, e.model_only * 100, "model only", entity="aux2", width=1.2,
           hover="%{y:+.1f}%<extra>model only</extra>")
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, "CBOT corn: what the model said on the Tuesday, against what the report published",
             ytitle="net non-commercial, % of open interest", height=360)
    st.plotly_chart(fig, width="stretch")

    st.success(
        f"**Anchoring cuts the error by {sc['improvement']:.0%}.** Run daily and reconciled every "
        f"Friday, the estimate lands within **{sc['mae_anchored_sd']:.2f} standard deviations** of "
        f"the eventual print. The model alone manages {sc['mae_model_only_sd']:.2f} sd. So the "
        f"sensible operating loop is exactly that: the model carries the estimate between reports, "
        f"and each release both corrects the level and scores the week.", icon="✅")
    st.warning(
        f"**And this is the error bar for SAFEX.** There is no report to anchor to, so the SAFEX "
        f"reading is permanently in model-only mode — the {sc['mae_model_only_sd']:.2f} sd column, "
        f"not the {sc['mae_anchored_sd']:.2f} sd one. Treat the level as indicative and the "
        f"*direction of flow* as the usable part. Broker-code data would supply the missing anchor.",
        icon="⚠️")

st.markdown("##### Do the price dynamics port, even if the flows cannot be seen?")
st.caption("If SAFEX price behaved nothing like CBOT, a trend model would mean something different "
           "there. It does not — the mechanism transfers cleanly even though the positioning behind "
           "it cannot be verified.")
X = cross_market()
disp_tbl = pd.DataFrame({
    "annualised vol": X.ann_vol, "return AC(1)": X.ac1, "signal flips / yr": X.flips_per_year,
    "mean |position|": X.mean_abs_trend, "% held with conviction": X.pct_conviction,
    "corr(trend, 12m return)": X.corr_trend_ret12, "trend Sharpe, gross": X.trend_sharpe_gross})
st.dataframe(disp_tbl.style.format({
    "annualised vol": "{:.1%}", "return AC(1)": "{:+.3f}", "signal flips / yr": "{:.1f}",
    "mean |position|": "{:.2f}", "% held with conviction": "{:.0%}",
    "corr(trend, 12m return)": "{:+.3f}", "trend Sharpe, gross": "{:+.2f}"}), width="stretch")
st.success(
    "**SAFEX is not a hostile trend market — if anything it trends more cleanly than CBOT corn.** "
    "Volatility is comparable, first-order return autocorrelation is *higher*, the signal flips "
    "*less* often, and the gross trend Sharpe is at least as good. So the zero footprint in the "
    "turnover tests is not because trend following fails here. It is because the money is not here, "
    "or is too small a share to see. That distinction matters: the overlay is describing a real "
    "feature of the price series, not an artefact of porting a model somewhere it does not belong.",
    icon="✅")
st.caption("Trend Sharpe is gross of costs and single-market — 0.2 to 0.4 is normal for one market; "
           "diversification across dozens is what makes managed futures work. It is a statement about "
           "the price series, not a strategy proposal.")

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
