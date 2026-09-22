import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import analogues as AN
from app.data import backtest as BT
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
MY_MONTHS = {i + 1: m for i, m in enumerate(
    ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"])}


@st.cache_data
def state_panel(grain_class: str) -> pd.DataFrame:
    """One row per release with every conditioning variable, all known at that date."""
    R = run_models(grain_class)
    a = R["A"]["fit"].panel[["vintage_date", "latest_month", "marketing_year", "my_month",
                             "months_cover", "close_1", "z"]].copy()
    a = a.merge(R["B"]["fit"].panel[["vintage_date", "z"]].rename(columns={"z": "spread_z"}),
                on="vintage_date", how="left")
    if "D" in R:
        a = a.merge(R["D"]["fit"].panel[["vintage_date", "z"]].rename(columns={"z": "basis_z"}),
                    on="vintage_date", how="left")
    else:
        a["basis_z"] = np.nan
    a["months_cover_pct_same_month"] = AN.pct_rank_same_month(a)
    a["months_cover_yoy"] = AN.yoy_log_change(a)
    a["months_cover_surprise"] = AN.seasonal_surprise(a)
    return a


p = state_panel(cls)
idx = FV.roll_adjusted_index(D["cont"], sym)

# ------------------------------------------------------------------ 1. compound state builder
st.markdown("#### 1. Define the state")
st.caption("Conditions are ANDed. **Valuation** variables ask whether the price is stretched; "
           "**divergence** variables ask whether the balance sheet itself is away from normal. "
           "A desk model would add a third kind here — its own forecast against the official "
           "estimate — which needs the crop-estimate layer we have not built.")

DEFAULTS = {"z": (-3.0, 3.0, 1.0, 0.25), "spread_z": (-3.0, 3.0, 1.0, 0.25),
            "basis_z": (-3.0, 3.0, 1.0, 0.25),
            "months_cover_pct_same_month": (0.0, 1.0, 0.25, 0.05),
            "months_cover_yoy": (-1.5, 1.5, -0.3, 0.05),
            "months_cover_surprise": (-1.2, 1.2, -0.2, 0.05)}
OPTS = list(AN.COLUMNS)
conditions: list[tuple[str, str, float]] = []

for i in range(3):
    c1, c2, c3 = st.columns([3, 1.2, 1.4])
    label = "Condition" if i == 0 else f"AND condition {i + 1} (optional)"
    choices = OPTS if i == 0 else ["—"] + OPTS
    col = c1.selectbox(label, choices, index=0 if i == 0 else 0, key=f"var{i}",
                       format_func=lambda k: "—" if k == "—" else
                       f"{AN.COLUMNS[k][0]}  ·  {AN.COLUMNS[k][1]}")
    if col == "—":
        c2.empty(); c3.empty()
        continue
    direction = c2.radio("Test", ["high", "low"], key=f"dir{i}", horizontal=True,
                         format_func=lambda d: "≥" if d == "high" else "≤")
    lo, hi, dflt, step = DEFAULTS[col]
    thr = c3.slider("Threshold", lo, hi, dflt, step, key=f"thr{i}")
    conditions.append((col, direction, float(thr)))

months = st.multiselect("Restrict to marketing-year months (optional)", list(MY_MONTHS),
                        format_func=MY_MONTHS.get)

mask = AN.build_mask_multi(p, conditions, months)
matched = p[mask].copy()
matched["episode"] = AN.episodes(mask)[mask].astype(int)
n_ep = int(matched.episode.nunique()) if len(matched) else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matched releases", len(matched))
k2.metric("Independent episodes", n_ep, help="Consecutive matched releases are one episode")
k3.metric("Conditions", len(conditions))
k4.metric("Latest release in state?", "YES" if len(mask) and bool(mask.iloc[-1]) else "no")

if len(matched) < 5:
    st.warning("Fewer than 5 matches — loosen a threshold or drop a condition.")
    st.stop()

# ------------------------------------------------------------------ 2. paths
st.markdown("#### 2. Forward price paths from the matched releases")
n_days = st.slider("Path length (trading days)", 10, 120, AN.PATH_DAYS, 5)
paths = AN.forward_paths(idx, matched.vintage_date, n_days)
paths_all = AN.forward_paths(idx, p.vintage_date, n_days)
fz, fu = AN.fan(paths) * 100, AN.fan(paths_all) * 100
col = P.color(cls)

fig = go.Figure()
for i, (d, row) in enumerate(paths.iterrows()):
    if i >= 80:
        break
    fig.add_trace(go.Scatter(x=row.index, y=row.to_numpy() * 100, mode="lines", showlegend=False,
                             line=dict(color=P.muted(), width=1), opacity=0.35,
                             hovertemplate="day %{x}: %{y:+.1f}%<extra>"
                                           + pd.Timestamp(d).strftime("%b %Y") + "</extra>"))
for hi, lo_, shade, name in ((fz.q90, fz.q10, "rgba(0,84,204,0.12)", "10–90%"),
                             (fz.q75, fz.q25, "rgba(0,84,204,0.22)", "25–75%")):
    fig.add_trace(go.Scatter(x=fz.index, y=hi, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fz.index, y=lo_, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=shade, name=name, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fz.index, y=fz.q50, mode="lines", name="median (conditional)",
                         line=dict(color=col, width=3),
                         hovertemplate="day %{x}: %{y:+.1f}%<extra>median</extra>"))
fig.add_trace(go.Scatter(x=fu.index, y=fu.q50, mode="lines", name="median (all releases)",
                         line=dict(color=P.STATUS["navy"], width=2, dash="dash"),
                         hovertemplate="day %{x}: %{y:+.1f}%<extra>unconditional</extra>"))
fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
P.layout(fig, f"{sym} front month, cumulative log return from the first close ≥ 1 day after each match",
         ytitle="%", xtitle="trading days after entry", height=420)
fig.update_yaxes(tickformat="+.0f")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 3. horizon distribution
st.markdown("#### 3. Distribution at a horizon: conditional vs all releases")
h_name = st.select_slider("Horizon", list(AN.HORIZON_CHOICES), value="10d")
h = AN.HORIZON_CHOICES[h_name]
if h > n_days:
    st.warning("Horizon exceeds path length — extend it above.")
    st.stop()
cond, uncond = paths[h] * 100, paths_all[h] * 100
S = AN.horizon_stats(cond, uncond, n_ep)

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=uncond.dropna(), histnorm="probability density",
                               name=f"all releases (n={S['n_uncond']})",
                               marker=dict(color=P.muted()), opacity=0.6, nbinsx=30))
    fig.add_trace(go.Histogram(x=cond.dropna(), histnorm="probability density",
                               name=f"conditional (n={S['n_obs']})",
                               marker=dict(color=col), opacity=0.7, nbinsx=30))
    fig.add_vline(x=S["cond_median"], line=dict(color=col, width=2))
    fig.add_vline(x=S["uncond_median"], line=dict(color=P.STATUS["navy"], width=2, dash="dash"))
    fig.update_layout(barmode="overlay", hovermode="closest")
    P.layout(fig, f"{h_name} forward log return (%)", ytitle="density", xtitle="%", height=340)
    st.plotly_chart(fig, width="stretch")
with c2:
    tbl = pd.DataFrame({
        "conditional": [S["n_obs"], S["cond_mean"], S["cond_median"], S["cond_p_neg"],
                        S["cond_p10"], S["cond_p90"]],
        "all releases": [S["n_uncond"], S["uncond_mean"], S["uncond_median"], S["uncond_p_neg"],
                         S["uncond_p10"], S["uncond_p90"]],
    }, index=["n", "mean %", "median %", "P(return < 0)", "10th pct %", "90th pct %"])
    st.dataframe(tbl.style.format("{:+.2f}"), width="stretch")
    st.markdown(f"**Episodes:** {S['n_episodes']} · **KS** D={S['ks_stat']:.2f}, p={S['ks_p']:.2f}")

# ------------------------------------------------------------------ 4. backtest
st.markdown("---")
st.markdown("#### 4. Backtest this state")
st.caption("Fade the dislocation: short when the valuation signal is rich, long when cheap. "
           "Trades may overlap; the portfolio holds every open trade at equal weight.")

b1, b2, b3, b4 = st.columns(4)
bt_h = b1.slider("Holding period (trading days)", 3, 60, 10, 1)
exit_rule = b2.selectbox("Exit", ["Fixed horizon", "Stop / target"])
stop = target = None
if exit_rule == "Stop / target":
    stop = b3.slider("Stop (%)", 1.0, 20.0, 6.0, 0.5) / 100
    target = b4.slider("Target (%)", 1.0, 30.0, 8.0, 0.5) / 100
else:
    b3.empty(); b4.empty()

c1, c2, c3, c4 = st.columns(4)
half_spread = c1.number_input("Half-spread (R/t)", 0.0, 50.0, 3.0, 0.5)
brokerage = c2.number_input("Brokerage (R/t)", 0.0, 10.0, 0.30, 0.05)
slippage = c3.number_input("Slippage (R/t)", 0.0, 50.0, 2.0, 0.5)
size_by_z = c4.checkbox("Size by |z|", value=False, help="Scale position with conviction, capped at 3σ")
costs = BT.CostModel(half_spread, brokerage, slippage)

_mys = sorted(p.marketing_year.unique())
holdout_from = st.select_slider(
    "Hold out from", options=_mys + ["(no holdout)"], value=_mys[-2],
    help="Everything from this marketing year onward is withheld from the equity curve below and "
         "scored once, separately. Move it right to bring recent seasons into the backtest; pick "
         "'(no holdout)' to use the whole sample.")

ent = matched[["vintage_date", "z", "close_1"]].rename(columns={"close_1": "front_close"}).copy()
ent["marketing_year"] = matched["marketing_year"].to_numpy()
if holdout_from == "(no holdout)":
    is_ent, oos_ent = ent, ent.iloc[0:0]
else:
    is_ent = ent[ent.marketing_year < holdout_from]
    oos_ent = ent[ent.marketing_year >= holdout_from]

res = BT.run_backtest(idx, is_ent, horizon=bt_h, costs=costs, stop=stop, target=target,
                      size_by_z=size_by_z)

if len(is_ent):
    _lo, _hi = pd.to_datetime(is_ent.vintage_date).agg(["min", "max"])
    st.caption(
        f"Backtesting **{len(is_ent)} of {len(ent)}** matched releases — "
        f"{_lo:%b %Y} to {_hi:%b %Y}"
        + (" — no holdout set." if holdout_from == "(no holdout)"
           else f", with {len(oos_ent)} held out from {holdout_from}." if len(oos_ent)
           else f". No matched release falls in the {holdout_from}+ holdout, so there is nothing "
                f"to score there — move the slider left to reserve some.")
        + f" The equity curve ends where the holdout begins. The final release also needs "
          f"{bt_h} trading days of price history after it to complete a trade, so the newest one or "
          f"two releases may not produce a trade yet.")
else:
    st.warning("Every matched release falls inside the holdout — move the slider right, or pick "
               "'(no holdout)', to see an equity curve.", icon="⚠️")

# --- trial counter: every distinct configuration this session is a trial ------------------
cfg = (cls, tuple(conditions), tuple(months), bt_h, exit_rule, stop, target, size_by_z,
       half_spread, brokerage, slippage, holdout_from)
trials = st.session_state.setdefault("bt_trials", {})
trials.setdefault(cfg, res.stats.get("sharpe_daily", np.nan))
n_trials = len(trials)
sharpes_d = np.array([v for v in trials.values() if v == v])
sharpe_std_d = float(np.std(sharpes_d)) if len(sharpes_d) > 1 else 0.0

s = res.stats
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Trades", s["n_trades"], f"{s['n_episodes']} episodes")
m2.metric("Hit rate", "—" if s["hit_rate"] != s["hit_rate"] else f"{s['hit_rate']:.0%}")
m3.metric("Avg net / trade", "—" if s["avg_net"] != s["avg_net"] else f"{s['avg_net'] * 100:+.2f}%")
m4.metric("Total (in-sample)", f"{s['total_ret'] * 100:+.0f}%")
m5.metric("Sharpe (ann.)", "—" if s["sharpe"] != s["sharpe"] else f"{s['sharpe']:+.2f}")
m6.metric("Max drawdown", f"{s['max_dd'] * 100:.0f}%")

if s["n_trades"]:
    _lo_i, _hi_i = res.trades.entry_date.min(), res.trades.exit_date.max()
    eq = res.equity.loc[_lo_i:_hi_i]
    fig = go.Figure()
    P.line(fig, eq.index, (eq - 1) * 100, "equity (in-sample)", entity=cls,
           hover="%{y:+.1f}%<extra></extra>")
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, f"Cumulative return, equal-weight across open trades, net of costs "
                  f"({eq.index.min():%b %Y} – {eq.index.max():%b %Y})", ytitle="%", height=300)
    st.plotly_chart(fig, width="stretch")

# --- the honest part ----------------------------------------------------------------------
st.markdown("##### Is this real, or did we just search until something looked good?")
hc = BT.sharpe_haircut(s["sharpe"] if s["sharpe"] == s["sharpe"] else 0.0, n_trials,
                       sharpe_std_d * np.sqrt(BT.TRADING_DAYS))
dsr = BT.deflated_sharpe(s["sharpe_daily"], s["n_obs"], s["skew"], s["kurtosis"],
                         n_trials, sharpe_std_d)

o1, o2, o3, o4 = st.columns(4)
o1.metric("Configurations tried", n_trials, help="Every distinct setting you have run this session")
o2.metric("E[max Sharpe] under null", f"{hc['expected_max']:+.2f}",
          help="What the best of this many random trials would score on noise alone")
o3.metric("Haircut Sharpe", f"{hc['haircut']:+.2f}",
          delta="evidence" if hc["haircut"] > 0 else "no evidence",
          delta_color="normal" if hc["haircut"] > 0 else "inverse")
o4.metric("Deflated Sharpe", "—" if dsr != dsr else f"{dsr:.2f}",
          delta="passes 0.95" if (dsr == dsr and dsr >= 0.95) else "below 0.95",
          delta_color="normal" if (dsr == dsr and dsr >= 0.95) else "inverse")

flags = []
if s["n_episodes"] and s["n_episodes"] < 20:
    flags.append(f"Only **{s['n_episodes']} independent episodes** — overlapping monthly entries make "
                 f"{s['n_trades']} trades look like more evidence than it is.")
if s["sharpe"] == s["sharpe"] and s["sharpe"] > 2:
    flags.append("**Sharpe above 2 on daily data** is a standard red flag for overfitting or leakage.")
if hc["haircut"] <= 0:
    flags.append("**Haircut Sharpe is negative** — after correcting for the number of configurations "
                 "tried, there is no statistical evidence of alpha here.")
if s["cost_share"] == s["cost_share"] and s["cost_share"] > 0.3:
    flags.append(f"Costs eat **{s['cost_share']:.0%}** of the average gross move — the edge is mostly "
                 f"a cost assumption.")
if n_trials > 10:
    flags.append(f"You have tried **{n_trials} configurations**. Treat the best one as a hypothesis "
                 f"to test on new data, not as a result.")
for f in flags:
    st.warning(f, icon="⚠️")
if not flags:
    st.info("No automatic red flags — which is not the same as a validated strategy.", icon="ℹ️")

# --- holdout, scored once ------------------------------------------------------------------
with st.expander(f"Holdout — {holdout_from} onward ({len(oos_ent)} matched releases). Open once, at the end."):
    st.caption("A holdout is only worth having if you look at it once, after you have stopped "
               "changing the configuration. Looking repeatedly turns it into another training set.")
    if st.button("Score the holdout", type="primary"):
        if len(oos_ent) < 3:
            st.warning("Too few matched releases in the holdout to say anything.")
        else:
            oos = BT.run_backtest(idx, oos_ent, horizon=bt_h, costs=costs, stop=stop,
                                  target=target, size_by_z=size_by_z)
            os_ = oos.stats
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Trades", os_["n_trades"], f"{os_['n_episodes']} episodes")
            h2.metric("Hit rate", "—" if os_["hit_rate"] != os_["hit_rate"] else f"{os_['hit_rate']:.0%}")
            h3.metric("Total", f"{os_['total_ret'] * 100:+.0f}%")
            h4.metric("Sharpe (ann.)", "—" if os_["sharpe"] != os_["sharpe"] else f"{os_['sharpe']:+.2f}")
            gap = (s["sharpe"] or 0) - (os_["sharpe"] or 0)
            st.caption(f"In-sample Sharpe {s['sharpe']:+.2f} vs holdout {os_['sharpe']:+.2f} "
                       f"(gap {gap:+.2f}). A large positive gap is what overfitting looks like; "
                       f"a holdout that matches in-sample too closely suggests leakage instead.")

with st.expander("Trade log"):
    if s["n_trades"]:
        cols = ["entry_date", "exit_date", "held_days", "direction", "z", "entry_px",
                "gross_ret", "cost", "net_ret"]
        st.dataframe(res.trades[cols].style.format({
            "z": "{:+.2f}", "entry_px": "{:,.0f}", "gross_ret": "{:+.3f}",
            "cost": "{:.4f}", "net_ret": "{:+.3f}"}), width="stretch", height=380)

with st.expander("Method notes"):
    st.markdown(f"""
- **State** — every conditioning variable is as known on the release date. Valuation variables are the
  out-of-sample model residuals from the Fair Value page; divergence variables compare the published
  balance sheet with its own seasonal norm using prior seasons only.
- **Entry** — the first close at least one day after the release, so the release-day reaction is excluded.
  Direction fades the valuation signal.
- **Costs** — round trip is 2 × (half-spread + brokerage + slippage) = **R{costs.round_trip_r_t:.2f}/t**,
  about **{costs.log_drag(4000) * 1e4:.0f} bps** at R4,000/t. SAFEX marks once a day on a thin book,
  so this assumption decides the answer more than most model choices do.
- **Portfolio** — overlapping trades are held at equal weight; the Sharpe comes from the daily
  portfolio series, annualised by √252.
- **Trials** — every distinct configuration run in this session counts. The haircut subtracts
  E[max Sharpe] over that many draws from a zero-mean null (Bailey & López de Prado); the Deflated
  Sharpe converts the result into P(true Sharpe > 0) given trials, sample length, skew and kurtosis.
- **What this cannot tell you** — with ~200 monthly releases most configurations here yield 10–30
  independent episodes. That is enough to reject a strategy and rarely enough to confirm one.
""")
