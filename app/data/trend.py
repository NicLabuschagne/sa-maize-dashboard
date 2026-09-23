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
        if symbol:
            return con.execute("SELECT * FROM cot WHERE symbol = ? ORDER BY date", [symbol]).df()
        return con.execute("SELECT * FROM cot ORDER BY date").df()
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


# ----------------------------------------------------------------------------- is it just price?
RETURN_HORIZONS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(1 - (r @ r) / ((y - y.mean()) ** 2).sum())


def price_benchmark(px: pd.Series, cot: pd.DataFrame, col: str = "net_noncomm_pct_oi") -> dict:
    """Does the trend model beat raw price momentum at explaining reported positioning?

    A trend model is built from price, so tracking COT is not by itself evidence that it captures
    anything about positioning. The test is whether it beats price returns, and whether it retains
    explanatory power once several return horizons are already in the regression.
    """
    lp = np.log(px.dropna())
    feat = pd.DataFrame({"trend": aggregate(trend_panel(px)).to_numpy()}, index=px.dropna().index)
    for lab, n in RETURN_HORIZONS.items():
        feat[lab] = (lp - lp.shift(n)).to_numpy()
    feat = feat.reset_index(names="d")
    c = cot.dropna(subset=[col]).copy()
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    j = pd.merge_asof(c.sort_values("date"), feat.sort_values("d"), left_on="date", right_on="d",
                      direction="backward", tolerance=pd.Timedelta("5D")).dropna(
                          subset=["trend", "ret_12m", col])
    if len(j) < 50:
        return {"ok": False, "n": len(j)}
    y = j[col].to_numpy()
    rets = list(RETURN_HORIZONS)

    def design(cols: list[str]) -> np.ndarray:
        return np.column_stack([np.ones(len(j))] + [j[c_].to_numpy() for c_ in cols])

    Xr = design(rets)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bt, *_ = np.linalg.lstsq(Xr, j.trend.to_numpy(), rcond=None)
    partial = float(np.corrcoef(j.trend.to_numpy() - Xr @ bt, y - Xr @ br)[0, 1])
    d = j[["trend", "ret_3m", col]].diff().dropna()
    return {
        "ok": True, "n": int(len(j)),
        "corr": {c_: float(np.corrcoef(j[c_], y)[0, 1]) for c_ in ["trend"] + rets},
        "r2_trend": _r2(y, design(["trend"])),
        "r2_ret12": _r2(y, design(["ret_12m"])),
        "r2_rets": _r2(y, Xr),
        "r2_both": _r2(y, design(["trend"] + rets)),
        "partial_trend": partial,
        "dchg_trend": float(d.trend.corr(d[col])),
        "dchg_ret3m": float(d.ret_3m.corr(d[col])),
    }


def dynamics(px: pd.Series) -> dict:
    """How the trend model behaves against price in one market - the basis for asking whether
    the mechanism ports to a market where positioning cannot be observed."""
    px = px.dropna()
    lp = np.log(px)
    r = lp.diff()
    agg = aggregate(trend_panel(px)).dropna()
    years = (px.index[-1] - px.index[0]).days / 365.25
    flips = int((np.sign(agg) != np.sign(agg.shift())).sum())
    r12 = (lp - lp.shift(252)).reindex(agg.index)
    pnl = (agg.shift(1) * r.reindex(agg.index)).dropna()     # gross of costs, by construction
    return {
        "ann_vol": float(r.std() * np.sqrt(252)),
        "ac1": float(r.autocorr(1)), "ac5": float(r.autocorr(5)),
        "flips_per_year": float(flips / years),
        "mean_abs_trend": float(agg.abs().mean()),
        "pct_conviction": float((agg.abs() > 0.5).mean()),
        "corr_trend_ret12": float(r12.corr(agg)),
        "trend_sharpe_gross": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() else np.nan,
        "years": float(years),
    }


# ----------------------------------------------------------------------------- flow
FLOW_WINDOW = 5          # trading days: one week of implied buying or selling


def flow(agg: pd.Series, window: int = FLOW_WINDOW) -> pd.Series:
    """Change in the modelled position - what a trend follower would have had to trade.

    Position is the stock, flow is the flow. A market moves on the buying and selling, so this
    is the series to put under a price chart. Units are position points: +0.20 means the model
    added a fifth of a full-size position over the window.
    """
    return (agg - agg.shift(window)).rename("flow")


def flow_table(agg: pd.Series, window: int = FLOW_WINDOW) -> pd.DataFrame:
    """Position and flow together, weekly, for plotting under a price series."""
    f = flow(agg, window)
    out = pd.DataFrame({"position": agg, "flow": f}).dropna()
    return out.resample("W-FRI").last().dropna()


def flow_state(agg: pd.Series, window: int = FLOW_WINDOW) -> dict:
    """Current position and flow, with the flow percentile against its own history."""
    a = agg.dropna()
    f = flow(a, window).dropna()
    if f.empty:
        return {}
    cur = float(f.iloc[-1])
    return {
        "position": float(a.iloc[-1]),
        "flow": cur,
        "direction": "buying" if cur > 0.02 else "selling" if cur < -0.02 else "flat",
        "pctile": float((f.abs() <= abs(cur)).mean()),
        "flow_1m": float((a.iloc[-1] - a.iloc[-22]) if len(a) > 22 else np.nan),
        "as_of": a.index[-1],
    }


# ----------------------------------------------------------------------------- anchored nowcast
MIN_REPORTS = 26          # half a year of released reports before the map is fitted
_NOWCAST_COLS = ["model", "model_only", "naive", "anchor_level", "anchor_date",
                 "days_since_report", "nowcast"]


def _prepare_cot(cot: pd.DataFrame, col: str) -> pd.DataFrame:
    c = cot.dropna(subset=[col]).copy()
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    # CFTC publishes Tuesday's positions on the Friday. A few rows carry no release date;
    # assume the standard three-day lag rather than discarding them.
    rd = pd.to_datetime(c["release_date"]) if "release_date" in c else pd.Series(pd.NaT, index=c.index)
    c["release_date"] = rd.astype("datetime64[ns]").fillna(c["date"] + pd.Timedelta(days=3))
    return c


def anchored_nowcast(agg: pd.Series, cot: pd.DataFrame, col: str = "net_noncomm_pct_oi") -> pd.DataFrame:
    """Daily positioning estimate, strictly point-in-time.

    The report measures Tuesday and publishes Friday, so the official level is always three to
    eight days stale. Three estimates are carried, each using only reports released by that day:

      naive       the last published level carried forward - no model at all
      model_only  a + b * model, with a and b fitted on released reports only
      nowcast     last published level + b * (model today - model on that report's Tuesday)

    `naive` is the benchmark that matters: positioning is persistent, so the last print is already
    a good estimate of the next one, and the model earns its place only by beating it.
    """
    c = _prepare_cot(cot, col)
    a = agg.dropna()
    a.index = pd.DatetimeIndex(a.index).as_unit("ns")
    c = pd.merge_asof(c.sort_values("date"),
                      pd.DataFrame({"d": a.index, "m": a.to_numpy()}).sort_values("d"),
                      left_on="date", right_on="d", direction="backward",
                      tolerance=pd.Timedelta("5D")).dropna(subset=["m"])
    c = c.sort_values("release_date").reset_index(drop=True)

    maps = []                             # refit the model -> positioning map as each report lands
    for i in range(MIN_REPORTS, len(c) + 1):
        known = c.iloc[:i]
        slope, icpt = np.polyfit(known["m"], known[col], 1)
        last = known.iloc[-1]
        maps.append({"release_date": last["release_date"], "anchor_date": last["date"],
                     "anchor_level": last[col], "anchor_model": last["m"],
                     "slope": slope, "intercept": icpt})
    if not maps:
        return pd.DataFrame(columns=_NOWCAST_COLS,
                            index=pd.DatetimeIndex([], name="date").as_unit("ns"))
    j = pd.merge_asof(pd.DataFrame({"date": a.index, "model": a.to_numpy()}).sort_values("date"),
                      pd.DataFrame(maps).sort_values("release_date"), left_on="date",
                      right_on="release_date", direction="backward")
    j["naive"] = j["anchor_level"]
    j["model_only"] = j["intercept"] + j["slope"] * j["model"]
    j["nowcast"] = j["anchor_level"] + j["slope"] * (j["model"] - j["anchor_model"])
    j["days_since_report"] = (j["date"] - j["anchor_date"]).dt.days
    return j.set_index("date")[_NOWCAST_COLS]


def nowcast_scorecard(nowcast: pd.DataFrame, cot: pd.DataFrame,
                      col: str = "net_noncomm_pct_oi") -> dict:
    """Grade each estimate against the level the report eventually published for that Tuesday.

    The estimate is taken as it stood on the measurement Tuesday, before that report was public,
    so nothing here sees the answer.
    """
    c = _prepare_cot(cot, col)
    n = nowcast.dropna(subset=["nowcast"]).reset_index()
    j = pd.merge_asof(c.sort_values("date"), n.sort_values("date"), on="date",
                      direction="backward", tolerance=pd.Timedelta("5D")).dropna(
                          subset=["nowcast", col])
    if len(j) < 30:
        return {"ok": False, "n": len(j)}
    sd = float(c[col].std())
    mae = {k: float((j[k] - j[col]).abs().mean()) for k in ("nowcast", "model_only", "naive")}
    return {"ok": True, "n": int(len(j)),
            "mae_anchored": mae["nowcast"], "mae_model_only": mae["model_only"],
            "mae_naive": mae["naive"],
            "mae_anchored_sd": mae["nowcast"] / sd, "mae_model_only_sd": mae["model_only"] / sd,
            "mae_naive_sd": mae["naive"] / sd,
            "corr_anchored": float(j["nowcast"].corr(j[col])),
            "corr_model_only": float(j["model_only"].corr(j[col])),
            "corr_naive": float(j["naive"].corr(j[col])),
            "improvement": float(1 - mae["nowcast"] / mae["model_only"]),
            "improvement_vs_naive": float(1 - mae["nowcast"] / mae["naive"]),
            "errors": j[["date", col, "nowcast", "model_only", "naive"]]}
