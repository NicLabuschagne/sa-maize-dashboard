import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.data.loaders import load_demo
from app.plots import line
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

df = load_demo()

c1, c2, c3 = st.columns(3)
c1.metric("Last", f"{df['price'].iloc[-1]:.2f}", f"{df['ret'].iloc[-1]:+.2f}")
c2.metric("Rows", len(df))
c3.metric("Std (chg)", f"{df['ret'].std():.2f}")

st.plotly_chart(line(df, "date", "price", "Demo series"), use_container_width=True)

with st.expander("Raw data"):
    st.dataframe(df.tail(50), use_container_width=True)
