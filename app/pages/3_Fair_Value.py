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
cls = sel["grain_class"]
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
        st.plotly_chart(scatter_fit(q, fit, xlab, ylab, entity, log_y), use_container_width=True)
    with c2:
        st.plotly_chart(z_history(q, entity, "Signal: out-of-sample residual z (positive = rich to fundamentals)"),
                        use_container_width=True)
        fig = go.Figure()
        P.line(fig, q.latest_month, np.exp(q.y) if log_y else q.y, "actual", entity=entity,
               hover="%{y:,.1f}<extra>actual</extra>")
        P.line(fig, q.latest_month, np.exp(q.fv) if log_y else q.fv, "fair value (expanding OOS)", entity="aux2",
               hover="%{y:,.1f}<extra>fair value</extra>")
        P.layout(fig, "", ytitle=ylab, height=220)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### Does the signal predict? Spearman IC of signal at *t* vs {fwd_label}, by horizon")
    st.caption("Cells show IC (block-bootstrap p-value, block = 6 months). Negative IC = rich → lower forward outcome, "
               "i.e. the expected sign. Entry is the first close ≥ 1 day after the SAGIS release.")
    st.dataframe(fmt_ic(R[key]["ic"]), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        h = st.selectbox("Tercile table horizon", HZ, index=1, key=f"h_{key}")
        t = R[key]["terciles"]
        t = t[t.horizon == h].set_index("bucket")[["n", "mean", "median", "hit_neg"]]
        st.dataframe(t.style.format({"mean": "{:+.3f}", "median": "{:+.3f}", "hit_neg": "{:.0%}"}),
                     use_container_width=True)
        st.caption("mean/median = forward outcome in the bucket; hit_neg = share of negative outcomes. "
                   "A working rich/cheap signal shows the *rich* bucket with the lowest mean and highest hit_neg.")
    with c2:
        if "stability" in R[key]:
            s = R[key]["stability"]
            s = s.assign(cell=s.apply(lambda r: f"{r.IC:+.2f} ({r.p:.2f}, n={int(r.n)})", axis=1))
            st.dataframe(s.pivot(index="sample", columns="horizon", values="cell").reindex(
                index=list(R[key]["stability"]["sample"].unique())), use_container_width=True)
            st.caption("Stability of the z-signal IC across sub-samples. If a result lives in one season, it isn't a result.")


tabA, tabB, tabC = st.tabs(["A · Flat price vs cover", "B · Calendar spread vs cover", "C · White premium"])

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
