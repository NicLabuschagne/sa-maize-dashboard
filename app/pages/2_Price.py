import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.state import derived, sidebar

st.set_page_config(page_title="Price", layout="wide")
sel = sidebar()
D = derived()
cont, prices, sd = D["cont"], D["prices"], D["sd"]

st.title("SAFEX Maize Prices")

c1, c2 = st.columns([3, 1])
with c2:
    start = st.date_input("From", value=pd.Timestamp("2015-01-01").date(), min_value=pd.Timestamp("2009-05-01").date())
    show_vint = st.checkbox("Mark SAGIS release dates", value=False)
    show_my = st.checkbox("Mark marketing-year starts (1 May)", value=True)
cont = cont[cont.trade_date >= pd.Timestamp(start)]

# --- front month, both classes ---------------------------------------------------------------
fig = go.Figure()
for sym in ("WMAZ", "YMAZ"):
    c = cont[cont.symbol == sym]
    P.line(fig, c.trade_date, c.close_1, f"{sym} front", entity=sym, hover="R%{y:,.0f}<extra>" + sym + "</extra>")
if show_my:
    P.add_vlines(fig, pd.Series([pd.Timestamp(y, 5, 1) for y in range(start.year, 2030)]), "MY start",
                 xmax=cont.trade_date.max())
if show_vint:
    v = sd[(sd.grain_class == "total") & (sd.vintage_date >= pd.Timestamp(start))].vintage_date
    P.add_vlines(fig, v)
P.layout(fig, "Front-month close, main delivery months (Mar/May/Jul/Sep/Dec), rolled 7 days before expiry", ytitle="R/t")
fig.update_xaxes(range=[pd.Timestamp(start), cont.trade_date.max()])
with c1:
    st.plotly_chart(fig, width="stretch")

# --- spreads ---------------------------------------------------------------------------------
l, r = st.columns(2)
with l:
    fig = go.Figure()
    for sym in ("WMAZ", "YMAZ"):
        c = cont[cont.symbol == sym]
        P.line(fig, c.trade_date, c.spread_2_1, f"{sym} 2nd − 1st", entity=sym,
               hover="R%{y:+,.0f}<extra>" + sym + "</extra>")
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, "Calendar spread: 2nd main month − front (R/t). Positive = carry, negative = inversion", ytitle="R/t")
    st.plotly_chart(fig, width="stretch")
with r:
    wy = D["wy"][D["wy"].trade_date >= pd.Timestamp(start)]
    fig = go.Figure()
    P.line(fig, wy.trade_date, wy.wy_spread, "White − Yellow (front)", entity="total",
           hover="R%{y:+,.0f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=P.muted(), width=1))
    P.layout(fig, "White premium over yellow, same front contract (R/t)", ytitle="R/t")
    st.plotly_chart(fig, width="stretch")

# --- term structure on a chosen date ----------------------------------------------------------
st.markdown("#### Forward curve on a date")
dates = sorted(prices.trade_date.unique())
d = st.select_slider("Trade date", options=dates, value=dates[-1], format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
curve = prices[(prices.trade_date == d) & (prices.open_interest > 0)].sort_values("expiry_date")
fig = go.Figure()
for sym in ("WMAZ", "YMAZ"):
    c = curve[curve.symbol == sym]
    fig.add_trace(go.Scatter(x=c.expiry, y=c.close, name=sym, mode="lines+markers",
                             line=dict(color=P.color(sym), width=2), marker=dict(size=8),
                             customdata=c[["open_interest", "volume"]],
                             hovertemplate="R%{y:,.0f}  OI %{customdata[0]:,}  vol %{customdata[1]:,}<extra>" + sym + "</extra>"))
P.layout(fig, f"Settlement by contract, {pd.Timestamp(d):%d %b %Y} (contracts with open interest)", ytitle="R/t", height=320)
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")
