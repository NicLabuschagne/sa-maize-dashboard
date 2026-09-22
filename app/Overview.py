import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data import features as F
from app.state import derived, fmt_t, sidebar
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")
sel = sidebar()
D = derived()
cls, sym = sel["grain_class"], sel["symbol"]

st.title(f"{sel['country']} {sel['commodity']} — Overview")

sd = D["sd"][D["sd"].grain_class == cls].dropna(subset=["months_cover"])
last = sd.iloc[-1]
prev = sd.iloc[-2]
cont = D["cont"][D["cont"].symbol == sym] if sym else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest SAGIS release", last.vintage_date.strftime("%d %b %Y"), f"data to {last.latest_month:%b %Y}")
c2.metric("Unutilised stock", fmt_t(last.closing_stock), fmt_t(last.closing_stock - prev.closing_stock))
c3.metric("Months of cover", f"{last.months_cover:.1f}", f"{last.months_cover - prev.months_cover:+.1f}")
if cont is not None and len(cont):
    px = cont.iloc[-1]
    c4.metric(f"{sym} front ({px.expiry_1})", f"R{px.close_1:,.0f}", f"{px.close_1 - cont.iloc[-2].close_1:+,.0f}")
    c5.metric("Calendar spread (2nd − 1st)", f"R{px.spread_2_1:+,.0f}", f"{px.spread_2_1_pct_ann:+.1f}% ann.")
else:
    c4.metric("Front month", "—"); c5.metric("Calendar spread", "—")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    P.line(fig, sd.latest_month, sd.months_cover, f"{cls} months of cover", entity=cls,
           hover="%{y:.1f} months<extra></extra>")
    P.layout(fig, "Months of cover (unutilised stock ÷ trailing-12m disappearance)", ytitle="months")
    st.plotly_chart(fig, width="stretch")
with right:
    if cont is not None:
        fig = go.Figure()
        P.line(fig, cont.trade_date, cont.close_1, f"{sym} front month", entity=sym,
               hover="R%{y:,.0f}<extra></extra>")
        P.layout(fig, f"{sym} front-month close (main months, rolled {F.ROLL_DAYS}d before expiry)", ytitle="R/t")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Select white or yellow for a price series.")

with st.expander("Latest release — balance sheet (latest month, tons)"):
    bs = D["bs"]
    latest = bs[(bs.vintage_date == last.vintage_date) & (bs.period_type.isin(["latest_month", "ytd", "ytd_prior"]))
                & (~bs.is_final)]
    tbl = latest.pivot_table(index="attribute", columns=["period_type", "grain_class"], values="value_t")
    st.dataframe(tbl.style.format("{:,.0f}"), width="stretch")
