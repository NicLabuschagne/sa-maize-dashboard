"""Trend-model positioning: what a canonical trend follower would hold in a market.

This is deliberately *not* called CTA positioning. For CBOT we can check the output against
CFTC Commitments of Traders and quote a measured correlation. For SAFEX no such report exists -
the JSE publishes open interest but not a breakdown by trader category - so the same model run
on SAFEX is an unvalidated port, and the page says so.

"Trend following" is not one strategy: it spans moving-average crossovers and time-series
momentum across lookbacks from a fortnight to a year, variously vol-scaled. So the output here
is a *panel* of signals, and the dispersion across them is part of the answer. When every signal
agrees, trend money is aligned and a reversal has fuel behind it; when they disagree there is no
consensus to squeeze.

Each signal returns a position in [-1, +1]: sign is direction, magnitude is conviction.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH

CROSSOVERS: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96), (64, 192))
MOMENTUM: tuple[int, ...] = (21, 63, 126, 252)
VOL_WINDOW = 252
MIN_PERIODS = 60


def load_cot(symbol: str | None = None) -> pd.DataFrame:
    """CFTC Commitments of Traders. Empty frame if it has not been fetched."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["symbol", "date", "net_noncomm_pct_oi"])
    try:
        q = "SELECT * FROM cot" + (f" WHERE symbol = '{symbol}'" if symbol else "") + " ORDER BY date"
        return con.execute(q).df()
    except Exception:  # noqa: BLE001 - table absent
        return pd.DataFrame(columns=["symbol", "date", "net_noncomm_pct_oi"])
    finally:
        con.close()


def _squash(x: pd.Series) -> pd.Series:
    """Standardise by the signal's own history, then squash into (-1, 1)."""
    sd = x.rolling(VOL_WINDOW, min_periods=MIN_PERIODS).std()
    return np.tanh(x / sd.replace(0, np.nan))


def crossover_signal(px: pd.Series, short: int, long_: int) -> pd.Series:
    """EMA(short) - EMA(long), normalised by price volatility over the slow window."""
    raw = px.ewm(span=short, adjust=False).mean() - px.ewm(span=long_, adjust=False).mean()
    return _squash(raw / px.rolling(long_, min_periods=long_ // 2).std().replace(0, np.nan))


def momentum_signal(px: pd.Series, n: int) -> pd.Series:
    """Time-series momentum over n days, scaled by the volatility of an n-day move."""
    lp = np.log(px)
    ret = lp - lp.shift(n)
    vol = lp.diff().rolling(n, min_periods=n // 2).std() * np.sqrt(n)
    return np.tanh(ret / vol.replace(0, np.nan))


def trend_panel(px: pd.Series) -> pd.DataFrame:
    """One column per constituent signal, indexed like `px`."""
    px = px.dropna().sort_index()
    out = {f"MA {s}/{l}": crossover_signal(px, s, l) for s, l in CROSSOVERS}
    out.update({f"Mom {n}d": momentum_signal(px, n) for n in MOMENTUM})
    return pd.DataFrame(out)


def aggregate(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight blend across signals, in [-1, +1]. Positive = net long."""
    return panel.mean(axis=1, skipna=True).rename("trend")


def dispersion(panel: pd.DataFrame, flat_band: float = 0.1) -> pd.DataFrame:
    """How many constituent signals are long, short or flat on each date."""
    long_ = (panel > flat_band).sum(axis=1)
    short = (panel < -flat_band).sum(axis=1)
    n = panel.notna().sum(axis=1)
    return pd.DataFrame({"n_long": long_, "n_short": short, "n_flat": n - long_ - short,
                         "n": n, "agreement": (long_ - short) / n.replace(0, np.nan)})


def latest_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Current reading of every signal, for the histogram."""
    last = panel.dropna(how="all").iloc[-1]
    return pd.DataFrame({"signal": last.index, "position": last.to_numpy()})


def validate_against_cot(trend: pd.Series, cot: pd.DataFrame,
                         col: str = "net_noncomm_pct_oi") -> dict:
    """Does the model recover real positioning where a report exists?

    COT positions are as of Tuesday, so the model is sampled on the same Tuesday - this measures
    whether the method *describes* positioning, not whether it predicts anything.
    """
    c = cot.dropna(subset=[col]).copy()
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    t = trend.dropna()
    t.index = pd.DatetimeIndex(t.index).as_unit("ns")
    m = pd.merge_asof(c.sort_values("date"),
                      pd.DataFrame({"tdate": t.index, "trend": t.to_numpy()}).sort_values("tdate"),
                      left_on="date", right_on="tdate", direction="backward",
                      tolerance=pd.Timedelta("5D")).dropna(subset=["trend"])
    if len(m) < 30:
        return {"ok": False, "n": len(m)}
    lvl = m["trend"].corr(m[col])
    lvl_s = m["trend"].corr(m[col], method="spearman")
    d = m[["trend", col]].diff().dropna()
    chg = d["trend"].corr(d[col]) if len(d) > 30 else np.nan
    return {"ok": True, "n": int(len(m)), "corr_level": float(lvl),
            "corr_level_spearman": float(lvl_s), "corr_change": float(chg),
            "first": m["date"].min(), "last": m["date"].max(), "merged": m}
