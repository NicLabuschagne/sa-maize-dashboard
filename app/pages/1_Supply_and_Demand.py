import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import features as F
from app.data import weekly as W
from app.state import derived, fmt_t, sidebar

st.set_page_config(page_title="S&D Explorer", layout="wide")
sel = sidebar()
D = derived()
cls = sel["grain_class"]
bs, sd = D["bs"], D["sd"][D["sd"].grain_class == cls]

st.title("Supply & Demand Explorer")
st.caption("Every number is as published in the SAGIS release of that date (point-in-time). "
           "Marketing year runs May–April.")

ATTRS = {"closing_stock": "Unutilised (closing) stock", "deliveries": "Producer deliveries",
         "utilisation": "Utilisation (processed + withdrawn + released)", "exports": "Exports",
         "imports": "Imports", "human_consumption": "Human consumption", "animal_feed": "Animal feed",
         "months_cover": "Months of cover", "stocks_to_use": "Stocks-to-use (÷ trailing-12m disappearance)"}
attr = st.selectbox("Series", list(ATTRS), format_func=ATTRS.get, index=0)

# --- 1. monthly series as published -------------------------------------------------------
fig = go.Figure()
is_ratio = attr in ("months_cover", "stocks_to_use")
P.line(fig, sd.latest_month, sd[attr], f"{cls} {ATTRS[attr]}", entity=cls,
       hover=("%{y:.2f}" if is_ratio else "%{y:,.0f} t") + "<extra></extra>")
P.layout(fig, f"{ATTRS[attr]} — monthly, as first published", ytitle="" if is_ratio else "t")
st.plotly_chart(fig, width="stretch")

# --- 2. season-to-date overlay -----------------------------------------------------------
if not is_ratio:
    piv = F.ytd_by_season(bs, attr, cls)
    n = st.slider("Seasons to show", 3, len(piv.columns), min(10, len(piv.columns)))
    st.plotly_chart(P.season_overlay(piv.iloc[:, -n:], f"{ATTRS[attr]} — season-to-date by marketing year",
                                     entity=cls), width="stretch")

# --- 2b. weekly pace ---------------------------------------------------------------------
wk = W.load_weekly()
if len(wk):
    st.markdown("#### Weekly pace")
    st.caption("SAGIS weekly: deliveries ~5 days after the Friday week-end, trade ~12. "
               "Past seasons as finalised.")
    flow = st.radio("Flow", list(W.FLOWS), format_func=W.FLOWS.get, horizontal=True, key="wk_flow")
    cum = W.season_to_date(wk, flow, cls)
    if not cum.empty:
        pc = W.pace(cum)
        latest = wk[(wk.flow == flow) & (wk.grain_class == cls) & (wk.season == pc["season"])
                    & (wk.week == pc["week"])]
        pct = lambda v: "—" if v != v else f"{v:+.0f}%"  # noqa: E731
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"To week {pc['week']}", fmt_t(pc["to_date"]))
        m2.metric("vs last season", pct(pc["vs_last_pct"]))
        m3.metric("vs 5-season avg", pct(pc["vs_avg_pct"]))
        if len(latest):
            m4.metric(f"Wk to {latest.week_end.iloc[0]:%d %b}", fmt_t(latest.tons_week.iloc[0]))
        n_w = st.slider("Seasons to show", 3, len(cum.columns), min(10, len(cum.columns)), key="wk_n")
        st.plotly_chart(P.season_overlay(cum.iloc[:, -n_w:], f"{W.FLOWS[flow]} — season-to-date by week",
                                         entity=cls, weekly=True), width="stretch")

# --- 3. revisions ------------------------------------------------------------------------
if attr not in ("months_cover", "stocks_to_use"):
    rev = F.sd_revisions(bs, attr, cls).dropna(subset=["revised"])
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure(go.Bar(x=rev.month, y=rev.revision_pct, name="revision %",
                               marker=dict(color=P.color(cls)), hovertemplate="%{y:+.2f}%<extra></extra>"))
        P.layout(fig, "Revision to the preliminary figure in the following release (%)", ytitle="%", height=300)
        fig.update_yaxes(tickformat=".1f")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Revision stats**")
        st.dataframe(rev.revision_pct.describe().round(2).rename("pct").to_frame(), width="stretch")

# --- 4. release table ---------------------------------------------------------------------
st.markdown("#### Release table")
vintages = sorted(bs.vintage_date.unique())
v = st.select_slider("Release", options=vintages, value=vintages[-1], format_func=lambda d: d.strftime("%Y-%m-%d"))
rel = bs[(bs.vintage_date == v) & (~bs.is_final)]
tbl = rel.pivot_table(index="attribute", columns=["period_type", "grain_class"], values="value_t")
order = [a for a in ("opening_stock", "acquisition", "deliveries", "imports", "utilisation", "processed_local",
                     "human_consumption", "animal_feed", "gristing", "biofuel", "withdrawn_producers",
                     "released_end_consumer", "exports", "exports_products", "exports_whole", "sundries",
                     "closing_stock", "stock_storers_traders", "stock_processors") if a in tbl.index]
st.dataframe(tbl.loc[order].style.format("{:,.0f}"), width="stretch", height=600)
