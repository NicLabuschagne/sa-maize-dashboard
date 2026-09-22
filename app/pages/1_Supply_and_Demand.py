import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import features as F
from app.state import derived, sidebar

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
