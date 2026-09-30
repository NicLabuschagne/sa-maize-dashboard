import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import plots as P
from app.data.band_runner import run_band_model
from app.state import sidebar

st.set_page_config(page_title="Band Fair Value", layout="wide")
sel = sidebar()
cls, sym = sel["grain_class"], sel["symbol"]
st.title("Band Fair Value")
st.caption("Where SAFEX should sit between export and import parity, given stocks-to-use. "
           "The stocks number is nowcast from each weekly SAGIS release between monthly balance sheets.")
if cls == "total":
    st.info("Pick **white** or **yellow** in the sidebar: the band needs a traded price series.")
    st.stop()

R = run_band_model(cls)
daily = R["daily"]
live = daily.dropna(subset=["fair_value"])
last = live.iloc[-1]

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"SAFEX {sym} (90-day)", f"R{last.safex:,.0f}")
k2.metric("Fair value", f"R{last.fair_value:,.0f}")
k3.metric("SAFEX vs fair value", f"{last.gap_rand:+,.0f} R/t", "above fair" if last.gap_rand > 0 else "below fair",
          delta_color="off")
k4.metric("Stocks-to-use (nowcast)", f"{last.stu_domestic_nowcast:.2f}")
k5, k6, k7, k8 = st.columns(4)
k5.metric("Band position", f"{last.position:.2f}")
k6.metric("Fair position", f"{last.fair_position:.2f}")
k7.metric("Export edge", f"R{last.export_edge:,.0f}")
k8.metric("Import edge", f"R{last.import_edge:,.0f}")
weekly_note = f", plus weekly trade to the week ending {last.last_week_published:%d %b}" if pd.notna(last.last_week_published) else ""
st.caption(f"As of {last.date:%d %b %Y}. Position: 0 = export parity, 1 = import parity. Stocks: SAGIS release of "
           f"{last.vintage_date:%d %b %Y} (month {last.latest_month:%b %Y}, ratio {last.stu_domestic:.2f})"
           f"{weekly_note} ({int(last.nowcast_weeks)} weeks). World price: CBOT × USD/ZAR at 10:00 UTC.")

span = st.radio("Period", ["Since 2020", "Last 5 years", "All out-of-sample"], index=0, horizontal=True)
start = {"Since 2020": pd.Timestamp("2020-01-01"), "Last 5 years": last.date - pd.DateOffset(years=5),
         "All out-of-sample": live.date.min()}[span]
view = live[live.date >= start]
st.plotly_chart(P.band_fair_value(view, cls, "SAFEX inside the parity band, with the out-of-sample fair value"),
                width="stretch")
st.plotly_chart(P.band_position_chart(view, cls, "Band position against the fair position"), width="stretch")

c1, c2 = st.columns([3, 2])
with c1:
    recent = daily[daily.date >= last.date - pd.DateOffset(years=3)]
    fig = go.Figure()
    P.line(fig, recent.date, recent.stu_domestic, "monthly release", entity="aux2", width=1.5,
           hover="%{y:.2f}<extra>monthly</extra>")
    P.line(fig, recent.date, recent.stu_domestic_nowcast, "with weekly nowcast", entity=cls, width=2,
           hover="%{y:.2f}<extra>nowcast</extra>")
    P.layout(fig, "Stocks-to-use: monthly release vs weekly nowcast (closing stock ÷ 12m domestic use)",
             ytitle="stocks-to-use", height=340)
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("**Out-of-sample fit** (weekly; R² against the historical average position)")
    scores = pd.concat([R["scores"], R["scores_by_year"]], ignore_index=True)
    st.dataframe(scores.rename(columns={"window": "Period", "weeks": "Weeks", "oos_r2": "R²",
                                        "mean_abs_gap_rand": "Mean |gap| R/t"}),
                 hide_index=True, width="stretch",
                 column_config={"R²": st.column_config.NumberColumn(format="%.2f"),
                                "Mean |gap| R/t": st.column_config.NumberColumn(format="%.0f")})

with st.expander("Method and how far to trust it"):
    st.markdown("""
- **Band.** The export edge is the world price × the 5th percentile of the SAFEX/world basis on all *earlier*
  days. The import edge adds the SAGIS cost width (freight, port and rail, where the Gulf FOB cancels) scaled by
  how far above the floor SAFEX has historically reached. It is learnt from the market because no trusted paper
  parity is available.
- **Price.** 90-day constant maturity, interpolated between the SAFEX contracts either side of 90 days, so the
  March→May old/new-crop roll does not jump.
- **Stocks-to-use.** The latest SAGIS closing stock plus weekly deliveries and imports, minus exports, minus
  pro-rated use, divided by trailing-12-month *domestic* use. Exports are left out of the denominator because
  they rise when price is at export parity.
- **Model.** Position = constant + season + slope × stocks-to-use, refit every week on earlier weeks only. The
  fair value shown on any past day is what the model said at the time.
- **Trust.** It explains about a fifth of where price sits in the band (R² ≈ 0.2 out of sample, including
  2023 onwards). **The gap to fair value has not predicted 5–40-day moves**, and a desk-style export-parity
  trade on a market-learnt floor lost money. Use it to see where fundamentals put the market, not as a trade
  trigger. It is weakest at harvest (May–Jul), when the market prices the new crop before stocks show it.
  Full study: `research/band_position/REPORT.md`.
""")
