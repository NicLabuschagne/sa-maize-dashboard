import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from app import plots as P
from app.data import backtest as BT
from app.data import release_fade as RF
from app.data.warehouse import load_macro_snap, load_signals
from app.state import derived, sidebar

# Configurations already tried in research/ to arrive at the defaults (thresholds, holds, legs,
# entry timing). Counted so the deflated Sharpe starts honest rather than at one trial.
RESEARCH_TRIALS = 12
CHECK_FROM = pd.Timestamp("2013-01-01")

st.set_page_config(page_title="Release Fade", layout="wide")
sidebar()
st.title("Release Fade — portfolio backtest")
st.caption("At each SAGIS release, fade every selected leg whose fair-value z is past the threshold; "
           "hold a fixed number of sessions. Costs per SAFEX leg, plus the CBOT/FX hedge on parity.")


@st.cache_data(show_spinner="Building instruments…")
def _inst() -> dict:
    D = derived()
    return RF.instruments(D["cont"], D["prices"], load_macro_snap())


@st.cache_data(show_spinner=False)
def _run(legs: tuple, start, end, threshold: float, hold: int, open_entry: bool, cost_mult: float,
         stop_pct: float | None) -> dict:
    return RF.run_portfolio(load_signals(), _inst(), dict(legs), start, end, threshold, hold,
                            open_entry, cost_mult, stop_pct)


inst = _inst()
last = max(s.index.max() for s in inst["idx"].values()).date()

# --- controls -------------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
legs = c1.multiselect("Legs", list(RF.LEGS), default=list(RF.DEFAULT_LEGS))
threshold = c2.slider("Enter at |z| ≥", 0.5, 2.5, 1.0, 0.25)
hold = c3.slider("Hold (sessions)", 3, 30, 10, 1)
entry = c4.radio("Outright entry", ["Next open", "Next 12:00 mark"], index=0,
                 help="SAGIS publishes ~14:30, after the 12:00 mark, so the next open is tradeable. "
                      "Parity, spreads and the cross always enter at the mark.")

d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
start = d1.date_input("From", pd.Timestamp("2021-09-01").date(), min_value=CHECK_FROM.date(), max_value=last)
end = d2.date_input("To", last, min_value=CHECK_FROM.date(), max_value=last)
cost_mult = d3.select_slider("Costs", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.0,
                             format_func=lambda x: f"{x:g}×")
stop_opt = d4.selectbox("Hard stop", ["None (time stop only)", "−5%", "−8%", "−10%", "−15%"], index=0)
stop_pct = None if stop_opt.startswith("None") else float(stop_opt.strip("−%"))

if not legs:
    st.info("Pick at least one leg.")
    st.stop()

weights = {k: 1.0 for k in legs}
if len(legs) > 1:
    wcols = st.columns(len(legs))
    for col, k in zip(wcols, legs):
        weights[k] = col.number_input(f"Weight · {k}", 0.0, 10.0, 1.0, 0.25, key=f"w_{k}")
    if sum(weights.values()) == 0:
        st.info("Weights sum to zero.")
        st.stop()

args = dict(legs=tuple(weights.items()), threshold=threshold, hold=hold,
            open_entry=entry == "Next open", cost_mult=cost_mult, stop_pct=stop_pct)
res = _run(start=pd.Timestamp(start), end=pd.Timestamp(end), **args)
port, trades = res["portfolio"], res["trades"]
m = RF.metrics(port, trades)

# --- headline ---------------------------------------------------------------------------------
fmt = lambda v, f: "—" if v is None or v != v else format(v, f)  # noqa: E731
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Sharpe", fmt(m.get("Sharpe"), ".2f"))
k2.metric("Sortino", fmt(m.get("Sortino"), ".2f"))
k3.metric("Calmar", fmt(m.get("Calmar"), ".2f"))
k4.metric("Max drawdown", fmt(m.get("Max DD %"), ".1f") + "%")
k5.metric("CAGR", fmt(m.get("CAGR %"), ".1f") + "%")
k6.metric("Trades · hit", f"{m.get('Trades', 0)} · {fmt(m.get('Hit %'), '.0f')}%")

if len(trades):
    st.plotly_chart(P.equity_drawdown(RF.curves(res), list(res["legs"].columns),
                                      f"{pd.Timestamp(start):%b %Y} – {pd.Timestamp(end):%b %Y}"),
                    width="stretch")
else:
    st.info("No release crossed the threshold in this window.")

# --- per leg ------------------------------------------------------------------------------------
rows = {"Portfolio": m}
for k in res["legs"].columns:
    rows[k] = RF.metrics(res["legs"][k], trades[trades.leg == k] if len(trades) else None)
cols = ["Sharpe", "Sortino", "Calmar", "Max DD %", "CAGR %", "Vol %", "Trades", "Hit %",
        "Avg win %", "Avg loss %", "Profit factor", "Worst trade %", "Worst open loss %",
        "Longest DD (sessions)", "In market %"]
tbl = pd.DataFrame(rows).T.reindex(columns=cols)
INT = {"Trades": "{:.0f}", "Longest DD (sessions)": "{:.0f}", "Hit %": "{:.0f}", "In market %": "{:.0f}"}
st.dataframe(tbl.style.format("{:.2f}", na_rep="—").format(INT, na_rep="—"), width="stretch")

if len(legs) > 1 and len(trades):
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.markdown("**Daily P&L correlation**")
        st.dataframe(res["legs"].corr().style.format("{:+.2f}"), width="stretch")

# --- same settings, before the window --------------------------------------------------------
before_end = pd.Timestamp(start) - pd.Timedelta(days=1)
if before_end > CHECK_FROM + pd.Timedelta(days=365):
    pre = _run(start=CHECK_FROM, end=before_end, **args)
    pm = RF.metrics(pre["portfolio"], pre["trades"])
    with (cc2 if len(legs) > 1 and len(trades) else st.container()):
        st.markdown(f"**Same settings, {CHECK_FROM:%Y} – {before_end:%b %Y}**")
        pre_rows = {"Portfolio": pm, **{k: RF.metrics(pre["legs"][k], pre["trades"][pre["trades"].leg == k]
                                                       if len(pre["trades"]) else None)
                                        for k in pre["legs"].columns}}
        st.dataframe(pd.DataFrame(pre_rows).T.reindex(columns=["Sharpe", "Max DD %", "CAGR %", "Trades", "Hit %"])
                     .style.format("{:.2f}", na_rep="—").format(INT, na_rep="—"), width="stretch")

# --- selection honesty -------------------------------------------------------------------------
cfg = (tuple(sorted(weights.items())), threshold, hold, args["open_entry"], cost_mult, stop_pct,
       str(start), str(end))
trials = st.session_state.setdefault("rf_trials", {})
sr_d = float(port.mean() / port.std()) if port.std() > 0 else np.nan
trials.setdefault(cfg, sr_d)
n_trials = RESEARCH_TRIALS + len(trials)
srs = np.array([v for v in trials.values() if v == v])
n_obs = int(len(port))
sd_trials = float(np.std(srs)) if len(srs) > 2 else 1 / np.sqrt(max(n_obs, 2))
dsr = BT.deflated_sharpe(sr_d, n_obs, float(port.skew()), float(port.kurtosis() + 3), n_trials, sd_trials)
hc = BT.sharpe_haircut(m.get("Sharpe", 0.0) or 0.0, n_trials, sd_trials * np.sqrt(BT.TRADING_DAYS))
o1, o2, o3 = st.columns(3)
o1.metric("Configurations counted", n_trials,
          help=f"{RESEARCH_TRIALS} from the research that chose the defaults, plus every distinct "
               f"setting run on this page this session")
o2.metric("Haircut Sharpe", fmt(hc["haircut"], "+.2f"),
          help="Observed Sharpe less the best Sharpe this many configurations would show on noise")
o3.metric("Deflated Sharpe", fmt(dsr, ".2f"), help="P(true Sharpe > 0) given trials, length, skew, kurtosis. "
                                                  "0.95 is the usual bar.")

with st.expander("Trade log"):
    if len(trades):
        show = trades.assign(release=trades.release.dt.date, entry=trades.entry.dt.date,
                             exit=trades.exit.dt.date).sort_values(["entry", "leg"])
        st.dataframe(show[["leg", "release", "entry", "exit", "side", "z", "sessions", "exit_on",
                           "net_pct", "worst_pct"]].style.format(
            {"z": "{:+.2f}", "net_pct": "{:+.2f}", "worst_pct": "{:+.2f}"}),
            width="stretch", hide_index=True)
