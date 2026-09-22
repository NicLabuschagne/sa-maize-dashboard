"""Data loaders. Replace `load_demo` with real sources as the project grows."""
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_demo(n: int = 250, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily series so the skeleton renders before real data is wired in."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    px = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({"date": idx, "price": px, "ret": np.r_[np.nan, np.diff(px)]})
