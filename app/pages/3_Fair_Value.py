import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import fairvalue as FV
from app.data.fv_runner import run_models
from app.state import sidebar

st.set_page_config(page_title="Fair Value", layout="wide")
sel = sidebar()
cls, sym = sel["grain_class"], sel["symbol"]
st.title("Fair Value vs Point-in-Time S&D")
if cls == "total":
    st.info("Pick **white** or **yellow** in the sidebar — fair value needs a traded price series.")
    st.stop()

R = run_models(cls)
HZ = list(FV.HORIZONS)


def fmt_ic(df: pd.DataFrame) -> pd.DataFrame:
    """signal × horizon table of 'IC (p)' strings."""
    t = df.assign(cell=lambda d: d.apply(lambda r: f"{r.IC:+.2f} ({r.p:.2f})" if pd.notna(r.IC) else "—", axis=1))
    return t.pivot(index="signal", columns="horizon", values="cell").reindex(columns=HZ)


def scatter_fit(q: pd.DataFrame, fit: FV.FitResult, xlab: str, ylab: str, entity: str, log_y: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.exp(q.x) if xlab.startswith("months") else q.x, y=np.exp(q.y) if log_y else q.y,
                             mode="markers", name="history", marker=dict(size=7, color=P.color(entity), opacity=0.45),
                             text=q.latest_month.dt.strftime("%b %Y"),
                             hovertemplate="%{text}<br>x %{x:.2f}<br>y %{y:,.1f}<extra></extra>"))
    # full-sample fitted curve at the seasonal value of the latest month (in-sample, for the eye)
    xs = np.linspace(q.x.min(), q.x.max(), 60)
    last_season = FV.fourier(pd.Series([q.my_month.iloc[-1]]))[0]
    b = fit.coef_full
    ys = (last_season @ b.iloc[:5].to_numpy()) + b["x"] * xs
    fig.add_trace(go.Scatter(x=np.exp(xs) if xlab.startswith("months") else xs, y=np.exp(ys) if log_y else ys,
                             mode="lines", name=f"full-sample fit ({q.latest_month.iloc[-1]:%b} seasonal)",
                             line=dict(color=P.STATUS["navy"], width=2)))
    lx, ly = q.x.iloc[-1], q.y.iloc[-1]
    fig.add_trace(go.Scatter(x=[np.exp(lx) if xlab.startswith("months") else lx], y=[np.exp(ly) if log_y else ly],
                             mode="markers+text", name="latest", text=[q.latest_month.iloc[-1].strftime("%b %Y")],
                             textposition="top center", marker=dict(size=13, color=P.STATUS["bad"],
                                                                    line=dict(width=2, color="#ffffff"))))
    P.layout(fig, "", ytitle=ylab, xtitle=xlab, height=400)
    fig.update_layout(hovermode="closest")
    if xlab.startswith("months"):
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def z_history(q: pd.DataFrame, entity: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=q.latest_month, y=q.z, name="z", marker=dict(color=P.color(entity)),
                         hovertemplate="%{y:+.2f}<extra>z</extra>"))
    for lvl in (-1, 1):
        fig.add_hline(y=lvl, line=dict(color=P.muted(), width=1, dash="dot"))
    P.layout(fig, title, ytitle="z", height=300)
    fig.update_yaxes(tickformat=".1f")
    return fig


def model_block(key: str, xlab: str, ylab: str, entity: str, log_y: bool, fwd_label: str) -> None:
    fit = R[key]["fit"]
    q = fit.panel
    last = q.iloc[-1]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Current z", f"{last.z:+.2f}", help="Out-of-sample residual ÷ expanding residual std")
    if log_y:
        k2.metric("Fair value (real)", f"R{np.exp(last.fv):,.0f}", f"{np.exp(last.y) - np.exp(last.fv):+,.0f} actual − fv")
    else:
        k2.metric("Fair value", f"{last.fv:+.1f}%", f"{last.y - last.fv:+.1f}pp actual − fv")
    k3.metric("Slope on log cover", f"{fit.coef_full['x']:+.2f}", f"t = {fit.tstat_full['x']:.1f}")
    k4.metric("R² (full sample)", f"{fit.r2_full:.2f}")
    k5.metric("Residual half-life", f"{fit.half_life_months:.1f} mo")
    k6.metric("Obs (out-of-sample)", f"{q.z.notna().sum()}", f"since {q.latest_month[q.z.notna()].iloc[0]:%b %Y}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(scatter_fit(q, fit, xlab, ylab, entity, log_y), width="stretch")
    with c2:
        st.plotly_chart(z_history(q, entity, "Signal: out-of-sample residual z (positive = rich to fundamentals)"),
                        width="stretch")
        fig = go.Figure()
        P.line(fig, q.latest_month, np.exp(q.y) if log_y else q.y, "actual", entity=entity,
               hover="%{y:,.1f}<extra>actual</extra>")
        P.line(fig, q.latest_month, np.exp(q.fv) if log_y else q.fv, "fair value (expanding OOS)", entity="aux2",
               hover="%{y:,.1f}<extra>fair value</extra>")
        P.layout(fig, "", ytitle=ylab, height=220)
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"##### Does the signal predict? Spearman IC of signal at *t* vs {fwd_label}, by horizon")
    st.caption("Cells show IC (block-bootstrap p-value, block = 6 months). Negative IC = rich → lower forward outcome, "
               "i.e. the expected sign. Entry is the first close ≥ 1 day after the SAGIS release.")
    st.dataframe(fmt_ic(R[key]["ic"]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        h = st.selectbox("Tercile table horizon", HZ, index=1, key=f"h_{key}")
        t = R[key]["terciles"]
        t = t[t.horizon == h].set_index("bucket")[["n", "mean", "median", "hit_neg"]]
        st.dataframe(t.style.format({"mean": "{:+.3f}", "median": "{:+.3f}", "hit_neg": "{:.0%}"}),
                     width="stretch")
        st.caption("mean/median = forward outcome in the bucket; hit_neg = share of negative outcomes. "
                   "A working rich/cheap signal shows the *rich* bucket with the lowest mean and highest hit_neg.")
    with c2:
        if "stability" in R[key]:
            s = R[key]["stability"]
            s = s.assign(cell=s.apply(lambda r: f"{r.IC:+.2f} ({r.p:.2f}, n={int(r.n)})", axis=1))
            st.dataframe(s.pivot(index="sample", columns="horizon", values="cell").reindex(
                index=list(R[key]["stability"]["sample"].unique())), width="stretch")
            st.caption("Stability of the z-signal IC across sub-samples. If a result lives in one season, it isn't a result.")


def parity_block() -> None:
    if "D" not in R:
        st.info("Parity needs the 10:00 UTC snapshot table. Run `python ingest/fetch_lse.py` with LSE_API_KEY set.")
        return
    fit = R["D"]["fit"]
    q = fit.panel
    last = q.iloc[-1]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Basis z", f"{last.z:+.2f}")
    k2.metric("SAFEX vs parity", f"{np.exp(last.basis) - 1:+.1%}",
              f"R{last.close_1:,.0f} vs R{last.world_rand:,.0f}")
    k3.metric("Slope on log cover", f"{fit.coef_full['x']:+.3f}", f"t = {fit.tstat_full['x']:.1f}")
    k4.metric("R² (full sample)", f"{fit.r2_full:.2f}")
    k5.metric("Basis half-life", f"{fit.half_life_months:.1f} mo")
    k6.metric("Obs (out-of-sample)", f"{q.z.notna().sum()}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        P.line(fig, q.vintage_date, q.close_1, f"SAFEX {sym} front", entity=cls, hover="R%{y:,.0f}<extra>SAFEX</extra>")
        P.line(fig, q.vintage_date, q.world_rand, "CBOT corn × USD/ZAR at the SAFEX mark", entity="aux2",
               hover="R%{y:,.0f}<extra>parity</extra>")
        P.layout(fig, "SAFEX against world parity, both in R/t", ytitle="R/t", height=360)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        P.line(fig, q.vintage_date, (np.exp(q.basis) - 1) * 100, "basis (SAFEX / parity − 1)", entity=cls,
               hover="%{y:+.1f}%<extra>basis</extra>")
        P.line(fig, q.vintage_date, (np.exp(q.fv) - 1) * 100, "fair basis given cover (expanding OOS)", entity="aux2",
               hover="%{y:+.1f}%<extra>fair</extra>")
        fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
        P.layout(fig, "Basis and its stock-implied fair level", ytitle="%", height=360)
        st.plotly_chart(fig, width="stretch")

    st.plotly_chart(z_history(q, cls, "Basis z (positive = SAFEX rich to parity given local stocks)"),
                    width="stretch")

    st.markdown("##### The signal is relative, not directional")
    st.caption("Same signal, two outcomes. Rows show IC (block-bootstrap p). The basis predicts the *spread* between "
               "SAFEX and parity; it says little about where the outright price goes, because parity itself moves.")
    st.dataframe(fmt_ic(R["D"]["ic"]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        h = st.selectbox("Tercile table horizon", HZ, index=1, key="h_D")
        t = R["D"]["terciles"]
        t = t[t.horizon == h].set_index("bucket")[["n", "mean", "median", "hit_neg"]]
        st.dataframe(t.style.format({"mean": "{:+.3f}", "median": "{:+.3f}", "hit_neg": "{:.0%}"}),
                     width="stretch")
        st.caption("Outcome is the SAFEX-minus-parity return.")
    with c2:
        s = R["D"]["stability"]
        s = s.assign(cell=s.apply(lambda r: f"{r.IC:+.2f} ({r.p:.2f}, n={int(r.n)})", axis=1))
        st.dataframe(s.pivot(index="sample", columns="horizon", values="cell"), width="stretch")

    aw = R["A_world"]
    b10 = aw["ic_base"].query("horizon == '10d'").IC.iloc[0]
    w10 = aw["ic_world"].query("horizon == '10d'").IC.iloc[0]
    st.markdown("##### Tested and rejected: adding the world price to the outright model (A)")
    st.markdown(
        f"Putting log(real world parity) into Model A alongside cover raises R² from "
        f"**{aw['base'].r2_full:.2f} to {aw['with_world'].r2_full:.2f}** with a coefficient of "
        f"**{aw['with_world'].coef_full['lw']:+.2f}** (t = {aw['with_world'].tstat_full['lw']:.1f}) — a large gain in "
        f"explanatory power. It buys **no** forecasting power: the 10-day IC goes from **{b10:+.2f} to {w10:+.2f}**, "
        f"i.e. flat to worse depending on the class (the loss is clearer on yellow, where SAFEX tracks CBOT more "
        f"closely). Controlling for a contemporaneous near-martingale changes what the residual measures — from "
        f"*rich against a slow local fundamental*, which drifts back over about two weeks, to *out of line with "
        f"CBOT×ZAR today*, which physical trade arbitrages. Fit and forecast are different objectives, and R² picked "
        f"the wrong one. The world price is therefore kept out of Model A and used here, on the leg it actually prices.")
    st.caption("Timing robustness: repeating this with the previous CBOT settle instead of the 10:00 UTC print moves "
               "the 10-day IC by under 0.005 — the two world-price series correlate 0.9993 — so the conclusion does "
               "not rest on the snapshot convention. Getting the convention right still matters: the same-day CBOT "
               "settle is not knowable at the SAFEX mark, so using it would be look-ahead regardless of its effect.")


tabA, tabB, tabC, tabD = st.tabs(["A · Flat price vs cover", "B · Calendar spread vs cover", "C · White premium",
                                  "D · Import/export parity"])

with tabA:
    st.markdown(f"**log(real {cls} front price) = season + b · log(months of cover)** — price deflated by ZA CPI "
                f"as available on the release date; season = 2-harmonic Fourier on marketing-year month; "
                f"fitted on an expanding window ending the month before.")
    model_block("A", "months of cover (log axis)", f"real {cls} front, R/t (log axis)", cls, True,
                "roll-adjusted forward log return of the front month")

with tabB:
    st.markdown(f"**{cls} calendar spread (2nd − 1st main month, % annualised) = season + b · log(months of cover)** — "
                f"no deflation needed. Forward outcome is the *change* in the spread.")
    model_block("B", "months of cover (log axis)", "2nd − 1st spread, % ann.", cls, False,
                "forward change in the annualised spread (pp)")

with tabC:
    st.markdown("**White premium (% of yellow, same front contract) = season + b · [log(white cover) − log(yellow cover)]**")
    model_block("C", "log(white cover) − log(yellow cover)", "white premium, % of yellow", "total", False,
                "forward change in the premium (pp)")
    fm = R["C"]["fit_mix"]
    icm = R["C"]["ic_mix"].set_index("horizon")
    ic0 = R["C"]["ic"].set_index("horizon")
    st.markdown("##### Tested and rejected: does the human-vs-feed demand mix drive the premium?")
    st.markdown(
        f"Adding log(trailing-12m human consumption ÷ animal feed) as a regressor: coefficient "
        f"**{fm.coef_full['demand_mix']:+.1f}** (t = {fm.tstat_full['demand_mix']:.1f}), R² "
        f"{R['C']['fit'].r2_full:.3f} → {fm.r2_full:.3f}. The sign is *negative* — a higher human/feed ratio goes with a "
        f"**lower** white premium — which is the demand response to price (feed users substitute into yellow when white is "
        f"dear), not a driver of it. And it adds nothing forward: IC at 1m {ic0.loc['1m','IC']:+.2f} → "
        f"{icm.loc['1m','IC']:+.2f}, at 3m {ic0.loc['3m','IC']:+.2f} → {icm.loc['3m','IC']:+.2f}. "
        f"Dropped from the model; the premium's own mean reversion is the usable observation.")

with st.expander("Method notes"):
    st.markdown("""
- **Point-in-time**: cover uses only the SAGIS release available on each date; CPI is lagged 45 days to publication;
  the fair value at *t* is fitted on data through *t − 1* (first fit after 48 months), so every residual shown is out-of-sample.
- **Seasonality**: 2-harmonic Fourier terms on marketing-year month (May = 1). Four parameters instead of eleven dummies.
- **Forward returns**: roll-adjusted log returns of the front main-month contract, entered at the first close at least one
  day after the release, so the release-day move is excluded.
- **Inference**: Spearman rank IC; p-values from a circular block bootstrap (block 6 months, 2 000 draws) because
  overlapping forward windows make the ordinary t-test invalid.
- **Benchmarks**: raw −log(cover) and the expanding seasonal-mean return show whether the residual adds anything beyond
  "low stocks = high price" and the harvest cycle.
- **Known weakness**: Feb–May, when the Crop Estimates Committee's forecast drives price and realised stocks are stale.
  The "ex Feb–May" row in the stability table isolates it. Adding CEC vintages is the v2 fix.
""")

with tabD:
    st.markdown("**log(SAFEX / world parity) = season + b · log(months of cover)**, where world parity is "
                "CBOT corn × USD/ZAR converted to R/t. SAFEX marks at 12:00 SAST = 10:00 UTC and South Africa keeps no "
                "DST, so both legs are taken from the 10:00 UTC bar — CBOT settles at 19:20/20:20 UTC, *after* the "
                "SAFEX mark, so the same-day settle is not information a SAFEX trader has. The 10:00 UTC corn print is "
                "the overnight Globex session and always carries volume.")
    parity_block()
