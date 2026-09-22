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

st.caption("What a canonical trend follower would hold — a nowcast of positioning, not a forecast "
           "of it. Validated against the CFTC report on CBOT corn; SAFEX has no such report, so the "
           "level there is indicative and the flow is the usable part. Risk overlay only.")

D = derived()


@st.cache_data
def safex_trend(symbol: str) -> tuple[pd.Series, pd.DataFrame]:
    idx = FV.roll_adjusted_index(derived()["cont"], symbol)
    return idx, T.trend_panel(idx)


@st.cache_data
def corn_validation() -> dict:
    m = load_macro()
    m = m[m.series == "cbot_corn"].sort_values("date")
    if m.empty:
        return {}
    px = pd.Series(m.value.to_numpy(), index=pd.DatetimeIndex(m.date).as_unit("ns"))
    panel = T.trend_panel(px)
    cot = T.load_cot("ZC")
    if cot.empty:
        return {}
    out = T.validate_against_cot(T.aggregate(panel), cot)
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
    V_ = corn_validation()
    px = V_.get("price")
    return T.price_benchmark(px, T.load_cot("ZC")) if px is not None else {}


@st.cache_data
def nowcast() -> dict:
    V_ = corn_validation()
    px = V_.get("price")
    if px is None:
        return {}
    cot = T.load_cot("ZC")
    nc = T.anchored_nowcast(T.aggregate(T.trend_panel(px)), cot)
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

# ---------------------------------------------------------------- current reading
st.markdown("#### Current reading")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Aggregate trend", f"{now:+.2f}",
          "net long" if now > 0.1 else "net short" if now < -0.1 else "flat", delta_color="off")
k2.metric("Signals long / short", f"{int(d.n_long)} / {int(d.n_short)}",
          f"{int(d.n_flat)} flat of {int(d.n)}")
k3.metric("Agreement", f"{d.agreement:+.0%}",
          help="(long − short) ÷ total. Near ±100% means trend money is aligned.")
k4.metric("As of", f"{pd.Timestamp(agg.dropna().index[-1]):%d %b %Y}")

c1, c2 = st.columns(2)
with c1:
    colours = [P.STATUS["bad"] if v < 0 else P.STATUS["good"] for v in last.position]
    fig = go.Figure(go.Bar(x=last.position, y=last.signal, orientation="h",
                           marker=dict(color=colours),
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, "Constituent signals", xtitle="position (−1 short … +1 long)", height=340)
    fig.update_xaxes(range=[-1.05, 1.05], tickformat="+.1f")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    P.line(fig, agg.index, agg, "aggregate trend", entity=cls, hover="%{y:+.2f}<extra></extra>")
    for lvl, colr in ((0.5, P.STATUS["good"]), (-0.5, P.STATUS["bad"])):
        fig.add_hline(y=lvl, line=dict(color=colr, width=1, dash="dot"), opacity=0.5)
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, f"{sym} aggregate position, history", ytitle="−1 … +1", height=340)
    fig.update_yaxes(range=[-1.05, 1.05], tickformat="+.1f")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- flow
st.markdown("#### Position is the stock, flow is what gets traded")
fs = T.flow_state(agg)
g1, g2, g3, g4 = st.columns(4)
g1.metric("Implied flow, 1 week", f"{fs['flow'] * 100:+.0f}pp", fs["direction"], delta_color="off",
          help="Change in the modelled position over five sessions, in points of a full position.")
g2.metric("Implied flow, 1 month", f"{fs['flow_1m'] * 100:+.0f}pp")
g3.metric("Size of move vs history", f"{fs['pctile']:.0%}")
g4.metric("Position now", f"{fs['position'] * 100:+.0f}%")

ft = T.flow_table(agg)
_c = D["cont"]
front = _c[_c.symbol == sym].set_index(
    pd.DatetimeIndex(_c[_c.symbol == sym].trade_date).as_unit("ns"))["close_1"].sort_index()
st.plotly_chart(P.price_position_flow(front.reindex(ft.index, method="ffill"), ft.position, ft.flow,
                                      cls, f"{sym} — price, modelled position and implied flow",
                                      price_label=f"{sym} front month", price_unit="R/t"),
                width="stretch")

# ---------------------------------------------------------------- validation on corn
st.markdown("---")
st.markdown("#### Validation: CBOT corn, where a positioning report exists")
V = corn_validation()
if not V or not V.get("ok"):
    st.info("Run `python ingest/fetch_lse.py` with LSE_API_KEY set to fetch COT and CBOT corn.")
else:
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Correlation, levels", f"{V['corr_level']:+.2f}", f"Spearman {V['corr_level_spearman']:+.2f}")
    v2.metric("Correlation, weekly change", f"{V['corr_change']:+.2f}")
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
               hover="%{y:+.2f}<extra>CFTC</extra>")
        fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
        P.layout(fig, "Model vs reported positioning, both standardised", ytitle="std dev", height=330)
        st.plotly_chart(fig, width="stretch")
    with c2:
        per = pd.Series(V["per_signal"], name="corr with CFTC").sort_values(ascending=False)
        st.dataframe(per.to_frame().style.format("{:+.3f}"), width="stretch", height=300)

    cpx = V.get("price")
    if cpx is not None:
        cft = T.flow_table(T.aggregate(T.trend_panel(cpx)))
        st.plotly_chart(P.price_position_flow(cpx.reindex(cft.index, method="ffill"),
                                              cft.position, cft.flow, "aux",
                                              "CBOT corn — price, modelled position and implied flow",
                                              price_label="CBOT corn front", price_unit="$/bu"),
                        width="stretch")

    B = price_benchmark()
    if B.get("ok"):
        st.markdown("##### Is it just price?")
        b1, b2 = st.columns([2, 3])
        with b1:
            corr = pd.Series(B["corr"], name="corr with CFTC").sort_values(ascending=False)
            corr.index = corr.index.str.replace("trend", "TREND MODEL").str.replace("ret_", "price return ")
            st.dataframe(corr.to_frame().style.format("{:+.3f}"), width="stretch", height=240)
        with b2:
            r2 = pd.Series({"12-month price return alone": B["r2_ret12"],
                            "trend model alone": B["r2_trend"],
                            "all four return horizons": B["r2_rets"],
                            "trend model + all four returns": B["r2_both"]}, name="R² explaining CFTC")
            st.dataframe(r2.to_frame().style.format("{:.3f}"), width="stretch", height=175)
            st.markdown(
                f"Partial correlation with CFTC after controlling for all four return horizons: "
                f"**{B['partial_trend']:+.2f}**. On weekly *changes* the model scores "
                f"**{B['dchg_trend']:+.2f}** against **{B['dchg_ret3m']:+.2f}** for a raw 3-month "
                f"return — saturation and volatility scaling are what track the turn.")

# ---------------------------------------------------------------- nowcast loop
N = nowcast()
if N and N.get("score", {}).get("ok"):
    sc, nc = N["score"], N["nc"]
    st.markdown("##### Nowcast daily, reconcile when the report lands")
    st.caption("Positions are measured on Tuesday and published on Friday, so the official number is "
               "three to eight days stale. Anchoring on the last published print and adding model "
               "flow since updates it daily, and stays point-in-time.")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Anchored nowcast error", f"{sc['mae_anchored_sd']:.2f} sd", "vs the next print",
              delta_color="off")
    n2.metric("Model-only error", f"{sc['mae_model_only_sd']:.2f} sd", delta_color="off")
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
    P.layout(fig, "What the model said on the Tuesday, against what the report published",
             ytitle="net non-commercial, % of open interest", height=340)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"SAFEX has no report to anchor to, so it sits permanently in the model-only column — "
               f"{sc['mae_model_only_sd']:.2f} sd, not {sc['mae_anchored_sd']:.2f}. Broker-code data "
               f"would supply the missing anchor.")

# ---------------------------------------------------------------- does it port
st.markdown("---")
st.markdown("#### Does it port to SAFEX?")

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
            "vol_uplift": float(np.exp(flips.mean()) / np.exp(others.mean()) - 1),
            "p": float(t.pvalue)}


F_ = safex_footprint(sym)

st.markdown("**1. Is anyone trading it?** Turnover around signal flips, month-of-year effects removed.")
f1, f2, f3 = st.columns(3)
f1.metric("Turnover on flip days", f"{F_['vol_uplift']:+.1%}", f"p = {F_['p']:.2f}", delta_color="off")
f2.metric("|Δ trend| vs volume", f"{F_['rho_vol']:+.3f}", "Spearman", delta_color="off")
f3.metric("Sign flips observed", F_["n_flips"], f"over {F_['n']:,} days")
st.caption("No detectable footprint. SAFEX turns over ~R1.4bn a day against CBOT corn's thirty-fold "
           "larger book, and the flow is dominated by the physical trade. Volume and open interest "
           "net across all participants, so a small share would not show up either way.")

st.markdown("**2. Do the price dynamics behave the same way?**")
X = cross_market()
st.dataframe(pd.DataFrame({
    "annualised vol": X.ann_vol, "return AC(1)": X.ac1, "signal flips / yr": X.flips_per_year,
    "mean |position|": X.mean_abs_trend, "corr(trend, 12m return)": X.corr_trend_ret12,
    "trend Sharpe, gross": X.trend_sharpe_gross}).style.format({
        "annualised vol": "{:.1%}", "return AC(1)": "{:+.3f}", "signal flips / yr": "{:.1f}",
        "mean |position|": "{:.2f}", "corr(trend, 12m return)": "{:+.3f}",
        "trend Sharpe, gross": "{:+.2f}"}), width="stretch")
st.caption("Comparable volatility, higher return autocorrelation, fewer signal flips and a gross "
           "trend Sharpe at least as good. The mechanism transfers; only the flow behind it is "
           "unverifiable. Sharpe is gross and single-market — 0.2 to 0.4 is normal for one market.")

with st.expander("Method"):
    st.markdown(f"""
- **Signals** — the {len(T.CROSSOVERS) + len(T.MOMENTUM)} shown in the histogram above, and nothing
  else: EMA crossovers {', '.join(f'{s}/{l}' for s, l in T.CROSSOVERS)} and time-series momentum at
  {', '.join(f'{n}d' for n in T.MOMENTUM)}. The same set runs on SAFEX and on corn.
- **Scaling** — crossovers normalised by price volatility over the slow window, momentum by the
  volatility of a move of that length; both standardised against their own rolling history and
  squashed with `tanh` into [−1, +1]. Sign is direction, magnitude is conviction.
- **Blend** — equal weight, deliberately not tuned to the corn fit, so the SAFEX port inherits no
  corn-specific calibration.
- **Prices** — the model runs on the roll-adjusted front-month series; charts show the front-month
  close.
- **Validation** — sampled on the CFTC measurement Tuesday, so it is a nowcast of positioning, not
  a forecast of it.
""")
