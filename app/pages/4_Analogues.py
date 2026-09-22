import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import analogues as AN
from app.data import fairvalue as FV
from app.data.fv_runner import run_models
from app.state import derived, sidebar

st.set_page_config(page_title="Analogues", layout="wide")
sel = sidebar()
cls, sym = sel["grain_class"], sel["symbol"]
st.title("Analogues — what happened after similar states")
if cls == "total":
    st.info("Pick **white** or **yellow** in the sidebar.")
    st.stop()

D = derived()
R = run_models(cls)
MY_MONTHS = {i + 1: m for i, m in enumerate(["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"])}


@st.cache_data
def state_panel(grain_class: str) -> pd.DataFrame:
    """One row per SAGIS release with every conditioning variable, all as known on that date."""
    a = run_models(grain_class)["A"]["fit"].panel[["vintage_date", "latest_month", "marketing_year", "my_month",
                                                    "months_cover", "close_1", "z"]].copy()
    b = run_models(grain_class)["B"]["fit"].panel[["vintage_date", "z"]].rename(columns={"z": "spread_z"})
    a = a.merge(b, on="vintage_date", how="left")
    a["months_cover_pct_same_month"] = AN.pct_rank_same_month(a)
    a["months_cover_yoy"] = AN.yoy_log_change(a)
    return a


p = state_panel(cls)
idx = FV.roll_adjusted_index(D["cont"], sym)

# --- event builder -------------------------------------------------------------------------
st.markdown("#### 1. Define the state")
EVENTS = {
    "z": ("Fair-value z (Model A residual)", -3.0, 3.0, 1.0, 0.25),
    "spread_z": ("Calendar-spread z (Model B residual)", -3.0, 3.0, 1.0, 0.25),
    "cover_pct": ("Cover percentile vs prior years, same MY month (0–1)", 0.0, 1.0, 0.25, 0.05),
    "cover_yoy": ("Cover, log change vs same month last season", -1.5, 1.5, -0.3, 0.05),
}
c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
event = c1.selectbox("Variable", list(EVENTS), format_func=lambda k: EVENTS[k][0])
direction = c2.radio("Condition", ["high", "low"], format_func=lambda d: "≥ threshold" if d == "high" else "≤ threshold",
                     horizontal=True)
lo, hi, default, step = EVENTS[event][1:]
if event == "cover_pct" and direction == "high":
    default = 0.75
if event == "cover_yoy" and direction == "high":
    default = 0.3
threshold = c3.slider("Threshold", lo, hi, default, step)
months = c4.multiselect("Restrict to marketing-year months (optional)", list(MY_MONTHS), format_func=MY_MONTHS.get)

mask = AN.build_mask(p, event, threshold, direction, months)
ep = AN.episodes(mask)
matched = p[mask].copy()
matched["episode"] = ep[mask].astype(int)
n_ep = int(matched.episode.nunique()) if len(matched) else 0
current = p.iloc[-1]
cur_val = current[{"z": "z", "spread_z": "spread_z", "cover_pct": "months_cover_pct_same_month",
                   "cover_yoy": "months_cover_yoy"}[event]]
in_state = bool(mask.iloc[-1])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matched releases", len(matched))
k2.metric("Independent episodes", n_ep, help="Consecutive matched releases count as one episode")
k3.metric("Latest release value", f"{cur_val:+.2f}" if pd.notna(cur_val) else "—",
          f"{current.latest_month:%b %Y} data")
k4.metric("Latest release in this state?", "YES" if in_state else "no")

if len(matched) < 5:
    st.warning("Fewer than 5 matches — widen the condition.")
    st.stop()

# --- paths --------------------------------------------------------------------------------
st.markdown("#### 2. Forward price paths from the matched releases")
n_days = st.slider("Path length (trading days)", 10, 120, AN.PATH_DAYS, 5)
paths = AN.forward_paths(idx, matched.vintage_date, n_days)
paths_all = AN.forward_paths(idx, p.vintage_date, n_days)
fz = AN.fan(paths) * 100
fig = go.Figure()
for i, (d, row) in enumerate(paths.iterrows()):
    if i >= 80:
        break
    fig.add_trace(go.Scatter(x=row.index, y=row.to_numpy() * 100, mode="lines", showlegend=False,
                             line=dict(color=P.muted(), width=1), opacity=0.35,
                             hovertemplate="day %{x}: %{y:+.1f}%<extra>" + pd.Timestamp(d).strftime("%b %Y") + "</extra>"))
col = P.color(cls)
fig.add_trace(go.Scatter(x=fz.index, y=fz.q90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fz.index, y=fz.q10, mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(0,84,204,0.12)", name="10–90%", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fz.index, y=fz.q75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fz.index, y=fz.q25, mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(0,84,204,0.22)", name="25–75%", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fz.index, y=fz.q50, mode="lines", name="median (conditional)", line=dict(color=col, width=3),
                         hovertemplate="day %{x}: %{y:+.1f}%<extra>median</extra>"))
fu = AN.fan(paths_all) * 100
fig.add_trace(go.Scatter(x=fu.index, y=fu.q50, mode="lines", name="median (all releases)",
                         line=dict(color=P.STATUS["navy"], width=2, dash="dash"),
                         hovertemplate="day %{x}: %{y:+.1f}%<extra>unconditional median</extra>"))
fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
P.layout(fig, f"{sym} front month, cumulative log return from the first close ≥ 1 day after each matched release",
         ytitle="%", xtitle="trading days after entry", height=430)
fig.update_yaxes(tickformat="+.0f")
st.plotly_chart(fig, width="stretch")

# --- distribution at horizon ---------------------------------------------------------------
st.markdown("#### 3. Distribution at a horizon: conditional vs all releases")
h_name = st.select_slider("Horizon", list(AN.HORIZON_CHOICES), value="10d")
h = AN.HORIZON_CHOICES[h_name]
if h > n_days:
    st.warning("Horizon exceeds path length — extend the path length above.")
    st.stop()
cond, uncond = paths[h] * 100, paths_all[h] * 100
S = AN.horizon_stats(cond, uncond, n_ep)

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=uncond.dropna(), histnorm="probability density", name=f"all releases (n={S['n_uncond']})",
                               marker=dict(color=P.muted()), opacity=0.6, nbinsx=30))
    fig.add_trace(go.Histogram(x=cond.dropna(), histnorm="probability density", name=f"conditional (n={S['n_obs']})",
                               marker=dict(color=col), opacity=0.7, nbinsx=30))
    fig.add_vline(x=S["cond_median"], line=dict(color=col, width=2))
    fig.add_vline(x=S["uncond_median"], line=dict(color=P.STATUS["navy"], width=2, dash="dash"))
    fig.update_layout(barmode="overlay")
    P.layout(fig, f"{h_name} forward log return (%), density", ytitle="density", xtitle="%", height=360)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
with c2:
    tbl = pd.DataFrame({
        "conditional": [S["n_obs"], S["cond_mean"], S["cond_median"], S["cond_p_neg"], S["cond_p10"], S["cond_p90"]],
        "all releases": [S["n_uncond"], S["uncond_mean"], S["uncond_median"], S["uncond_p_neg"], S["uncond_p10"], S["uncond_p90"]],
    }, index=["n", "mean %", "median %", "P(return < 0)", "10th pct %", "90th pct %"])
    st.dataframe(tbl.style.format(lambda v: f"{v:.0f}" if isinstance(v, (int, np.integer)) or v == int(v) and abs(v) > 20
                                  else f"{v:+.2f}"), width="stretch")
    st.markdown(f"**Episodes:** {S['n_episodes']} independent · **KS test** D = {S['ks_stat']:.2f}, p = {S['ks_p']:.2f}")
    st.caption("KS compares the two samples' shapes. Overlapping windows inflate n_obs, so read p against the episode "
               "count, not the observation count. A shift in median with P(return<0) moving away from ~50% is the "
               "practical read.")

with st.expander("Matched releases"):
    m = matched[["vintage_date", "latest_month", "marketing_year", "episode", "months_cover", "z", "spread_z",
                 "months_cover_pct_same_month", "months_cover_yoy", "close_1"]].copy()
    m["fwd_h_%"] = cond.to_numpy()
    st.dataframe(m.style.format({"months_cover": "{:.2f}", "z": "{:+.2f}", "spread_z": "{:+.2f}",
                                 "months_cover_pct_same_month": "{:.2f}", "months_cover_yoy": "{:+.2f}",
                                 "close_1": "{:,.0f}", "fwd_h_%": "{:+.2f}"}), width="stretch", height=400)

with st.expander("Method notes"):
    st.markdown("""
- Every conditioning variable is **as known on the release date**: z-scores are the out-of-sample residuals from the
  Fair Value page; the cover percentile ranks the release against *prior* seasons' same-month values (needs ≥ 5 prior
  years); the YoY change compares with the same month one season earlier as published then.
- Paths enter at the first close at least one day after the release and use the roll-adjusted front-month index,
  so no roll jumps.
- The **unconditional** comparison uses the same release calendar (every SAGIS release), not every trading day, so the
  two distributions share the same entry timing.
- Consecutive matched releases are one **episode**. With ~17 seasons of price history, the episode count is the honest
  sample size.
""")
