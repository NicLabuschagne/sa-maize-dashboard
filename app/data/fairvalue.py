"""Fair-value models on point-in-time S&D, and the tests that decide whether the residual predicts.

Model A  log(real front price)      = a + b·log(cover) + season
Model B  calendar spread (% ann.)   = a + b·log(cover) + season
Model C  white premium (% of yellow)= a + b·(log white cover − log yellow cover) [+ c·demand mix] + season

`season` is a 2-harmonic Fourier basis on marketing-year month (4 params) rather than 11 dummies:
with ~200 monthly observations that is the difference between a fit and an overfit.

Every fitted value is produced on an expanding window that ends the month before, so the residual
at t is out-of-sample.  Forward returns are roll-adjusted log returns of the front month.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = {"5d": 5, "10d": 10, "1m": 21, "2m": 42, "3m": 63, "6m": 126}
PRICE_TOL = pd.Timedelta("7D")   # a release must map to a close within a week, else it has no price
MIN_OBS = 48          # months of history before the first out-of-sample fit
CPI_LAG_DAYS = 45     # CPI for month m is published ~3 weeks into m+1
ENTRY_LAG_DAYS = 1    # measure forward returns from the first close at least this many days after the release


# ----------------------------------------------------------------------------- helpers
def fourier(my_month: pd.Series, k: int = 2) -> np.ndarray:
    ang = 2 * np.pi * (my_month.to_numpy() - 1) / 12
    cols = [np.ones(len(ang))]
    for j in range(1, k + 1):
        cols += [np.sin(j * ang), np.cos(j * ang)]
    return np.column_stack(cols)


def roll_adjusted_index(cont: pd.DataFrame, symbol: str) -> pd.Series:
    c = cont[cont.symbol == symbol].sort_values("trade_date")
    idx = np.exp(c["log_ret_1"].fillna(0).cumsum())
    return pd.Series(idx.to_numpy(), index=pd.DatetimeIndex(c["trade_date"]).as_unit("ns"), name="idx")


def forward_returns(idx: pd.Series, dates: pd.Series, horizons: dict[str, int] = HORIZONS,
                    lag_days: int = ENTRY_LAG_DAYS) -> pd.DataFrame:
    """Log return of the roll-adjusted index over h trading days, entered at the first close
    at least `lag_days` calendar days after each date (so the release-day reaction is excluded)."""
    entry = pd.DatetimeIndex(dates).as_unit("ns") + pd.Timedelta(days=lag_days)
    pos = idx.index.searchsorted(entry)
    out = {}
    vals = idx.to_numpy()
    for name, h in horizons.items():
        end = pos + h
        ok = end < len(vals)
        r = np.full(len(pos), np.nan)
        r[ok] = np.log(vals[end[ok]] / vals[pos[ok]])
        out[f"fwd_{name}"] = r
    return pd.DataFrame(out, index=dates.index)


def real_price(px: pd.Series, dates: pd.Series, cpi: pd.DataFrame) -> pd.Series:
    """Deflate by the latest CPI that was published by each date (CPI_LAG_DAYS after month-start)."""
    c = cpi[cpi.series == "za_cpi"][["date", "value"]].copy()
    c["avail"] = (c["date"] + pd.Timedelta(days=CPI_LAG_DAYS)).astype("datetime64[ns]")
    c = c.sort_values("avail")
    d = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]")}, index=dates.index).sort_values("date")
    m = pd.merge_asof(d, c[["avail", "value"]], left_on="date", right_on="avail", direction="backward")
    base = c["value"].iloc[-1]
    return (px / (m.set_index(d.index)["value"] / base)).rename("real_px")


# ----------------------------------------------------------------------------- panels
def panel_price(sd: pd.DataFrame, cont: pd.DataFrame, cpi: pd.DataFrame, grain_class: str, symbol: str) -> pd.DataFrame:
    """One row per vintage: cover, real front price, season, forward returns. Model A input."""
    s = sd[sd.grain_class == grain_class].dropna(subset=["months_cover"]).copy()
    s = s[s.months_cover > 0]
    s["vintage_date"] = s["vintage_date"].astype("datetime64[ns]")
    c = cont[cont.symbol == symbol][["trade_date", "close_1", "spread_2_1_pct_ann"]].copy()
    c["trade_date"] = c["trade_date"].astype("datetime64[ns]")
    p = pd.merge_asof(s.sort_values("vintage_date"), c.sort_values("trade_date"),
                      left_on="vintage_date", right_on="trade_date", direction="forward", tolerance=PRICE_TOL)
    p = p.dropna(subset=["close_1"]).reset_index(drop=True)
    p["real_px"] = real_price(p["close_1"], p["vintage_date"], cpi)
    p["y"] = np.log(p["real_px"])
    p["x"] = np.log(p["months_cover"])
    idx = roll_adjusted_index(cont, symbol)
    p = pd.concat([p, forward_returns(idx, p["vintage_date"])], axis=1)
    return p


def panel_spread(sd: pd.DataFrame, cont: pd.DataFrame, grain_class: str, symbol: str) -> pd.DataFrame:
    """Model B input: y = annualised 2nd−1st spread (%), x = log cover. Forward 'return' = change in spread."""
    s = sd[sd.grain_class == grain_class].dropna(subset=["months_cover"]).copy()
    s = s[s.months_cover > 0]
    s["vintage_date"] = s["vintage_date"].astype("datetime64[ns]")
    c = cont[cont.symbol == symbol][["trade_date", "close_1", "spread_2_1_pct_ann"]].copy()
    c["trade_date"] = c["trade_date"].astype("datetime64[ns]")
    p = pd.merge_asof(s.sort_values("vintage_date"), c.sort_values("trade_date"),
                      left_on="vintage_date", right_on="trade_date", direction="forward", tolerance=PRICE_TOL)
    p = p.dropna(subset=["spread_2_1_pct_ann"]).reset_index(drop=True)
    p["y"] = p["spread_2_1_pct_ann"]
    p["x"] = np.log(p["months_cover"])
    sp = c.set_index("trade_date")["spread_2_1_pct_ann"]
    pos = sp.index.searchsorted(p["vintage_date"] + pd.Timedelta(days=ENTRY_LAG_DAYS))
    for name, h in HORIZONS.items():
        end = pos + h
        ok = end < len(sp)
        r = np.full(len(pos), np.nan)
        r[ok] = sp.to_numpy()[end[ok]] - sp.to_numpy()[pos[ok]]
        p[f"fwd_{name}"] = r
    return p


def panel_white_yellow(sd: pd.DataFrame, wy: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """Model C input: y = white premium % of yellow; x = log(white cover) − log(yellow cover);
    demand_mix = trailing-12m human consumption ÷ animal feed (total maize), as published."""
    w = sd[sd.grain_class == "white"].set_index("vintage_date")
    y_ = sd[sd.grain_class == "yellow"].set_index("vintage_date")
    t = sd[sd.grain_class == "total"].set_index("vintage_date")
    p = pd.DataFrame({"months_cover_w": w["months_cover"], "months_cover_y": y_["months_cover"],
                      "my_month": w["my_month"], "latest_month": w["latest_month"],
                      "marketing_year": w["marketing_year"]}).dropna()
    p = p[(p.months_cover_w > 0) & (p.months_cover_y > 0)]
    hc = t["human_consumption"].rolling(12, min_periods=9).sum()
    af = t["animal_feed"].rolling(12, min_periods=9).sum()
    p["demand_mix"] = np.log(hc / af).reindex(p.index)
    p["x"] = np.log(p["months_cover_w"]) - np.log(p["months_cover_y"])
    p = p.reset_index()
    p["vintage_date"] = p["vintage_date"].astype("datetime64[ns]")
    s = wy[["trade_date", "wy_spread_pct"]].copy()
    s["trade_date"] = s["trade_date"].astype("datetime64[ns]")
    p = pd.merge_asof(p.sort_values("vintage_date"), s.sort_values("trade_date"),
                      left_on="vintage_date", right_on="trade_date", direction="forward", tolerance=PRICE_TOL)
    p = p.dropna(subset=["wy_spread_pct", "demand_mix"]).reset_index(drop=True)
    p["y"] = p["wy_spread_pct"]
    sp = s.set_index("trade_date")["wy_spread_pct"]
    pos = sp.index.searchsorted(p["vintage_date"] + pd.Timedelta(days=ENTRY_LAG_DAYS))
    for name, h in HORIZONS.items():
        end = pos + h
        ok = end < len(sp)
        r = np.full(len(pos), np.nan)
        r[ok] = sp.to_numpy()[end[ok]] - sp.to_numpy()[pos[ok]]
        p[f"fwd_{name}"] = r
    return p


# ----------------------------------------------------------------------------- fitting
@dataclass
class FitResult:
    panel: pd.DataFrame          # input + fv, resid, z (expanding, out-of-sample)
    coef_full: pd.Series         # full-sample coefficients (in-sample chart only)
    r2_full: float
    tstat_full: pd.Series
    half_life_months: float


def _design(p: pd.DataFrame, extra: list[str]) -> np.ndarray:
    return np.column_stack([fourier(p["my_month"]), p[["x"] + extra].to_numpy()])


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def fit_expanding(p: pd.DataFrame, extra: list[str] | None = None, min_obs: int = MIN_OBS) -> FitResult:
    extra = extra or []
    p = p.dropna(subset=["y", "x"] + extra).reset_index(drop=True).copy()
    X, y = _design(p, extra), p["y"].to_numpy()
    fv = np.full(len(p), np.nan)
    for t in range(min_obs, len(p)):
        beta, _ = _ols(X[:t], y[:t])
        fv[t] = X[t] @ beta
    p["fv"] = fv
    p["resid"] = p["y"] - p["fv"]
    p["z"] = p["resid"] / p["resid"].expanding(min_periods=12).std().shift(1)
    beta, se = _ols(X, y)
    names = ["const", "sin1", "cos1", "sin2", "cos2", "x"] + extra
    fitted_full = X @ beta
    r2 = 1 - ((y - fitted_full) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p["fv_full"] = fitted_full
    r = p["resid"].dropna()
    phi = np.corrcoef(r.iloc[1:], r.iloc[:-1])[0, 1] if len(r) > 3 else np.nan
    hl = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")
    return FitResult(p, pd.Series(beta, index=names), float(r2), pd.Series(beta / se, index=names), hl)


# ----------------------------------------------------------------------------- tests
def _block_bootstrap_p(a: np.ndarray, b: np.ndarray, block: int = 6, n_boot: int = 2000, seed: int = 0) -> float:
    """Two-sided p-value for Spearman(a, b) = 0 under a circular block bootstrap of each series (vectorised)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    obs = np.corrcoef(ra, rb)[0, 1]
    nb = int(np.ceil(n / block))
    offsets = np.arange(block)

    def draw() -> np.ndarray:
        starts = rng.integers(0, n, (n_boot, nb))
        return ((starts[:, :, None] + offsets) % n).reshape(n_boot, -1)[:, :n]

    A, B = ra[draw()], rb[draw()]
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    null = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return float((np.abs(null) >= abs(obs)).mean())


def ic_table(p: pd.DataFrame, signal_cols: dict[str, str], horizons: dict[str, int] = HORIZONS,
             mask: pd.Series | None = None) -> pd.DataFrame:
    """Spearman IC of each signal vs each forward return, with block-bootstrap p-values."""
    rows = []
    q = p if mask is None else p[mask]
    for sig_name, col in signal_cols.items():
        for h in horizons:
            d = q[[col, f"fwd_{h}"]].dropna()
            if len(d) < 24:
                rows.append({"signal": sig_name, "horizon": h, "n": len(d), "IC": np.nan, "p": np.nan})
                continue
            ic = stats.spearmanr(d[col], d[f"fwd_{h}"]).statistic
            rows.append({"signal": sig_name, "horizon": h, "n": len(d), "IC": ic,
                         "p": _block_bootstrap_p(d[col].to_numpy(), d[f"fwd_{h}"].to_numpy())})
    return pd.DataFrame(rows)


def tercile_table(p: pd.DataFrame, col: str = "z", horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    d = p.dropna(subset=[col]).copy()
    d["bucket"] = pd.qcut(d[col], 3, labels=["cheap (low z)", "middle", "rich (high z)"])
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        for h in horizons:
            r = g[f"fwd_{h}"].dropna()
            rows.append({"bucket": b, "horizon": h, "n": len(r), "mean": r.mean(), "median": r.median(),
                         "hit_neg": (r < 0).mean()})
    return pd.DataFrame(rows)


def seasonal_benchmark(p: pd.DataFrame, horizon: str = "1m", min_obs: int = MIN_OBS) -> pd.Series:
    """Expanding mean forward return by marketing-year month (what a pure seasonal trader knows at t)."""
    out = np.full(len(p), np.nan)
    col = f"fwd_{horizon}"
    for t in range(min_obs, len(p)):
        hist = p.iloc[:t]
        m = hist[hist.my_month == p.my_month.iloc[t]][col].mean()
        out[t] = m
    return pd.Series(out, index=p.index, name=f"seasonal_{horizon}")


# ----------------------------------------------------------------------------- import/export parity
BU_PER_TONNE = 39.3683      # corn: 1 short bushel = 25.4012 kg
SAFEX_MARK_UTC = 10         # 12:00 SAST, no DST


def world_parity(snap: pd.DataFrame) -> pd.Series:
    """CBOT corn converted to R/t using the CBOT and USD/ZAR prints at the SAFEX mark (10:00 UTC).

    CBOT settles ~19:20 UTC, after the SAFEX mark, so the same-day settle is NOT knowable at the
    mark; the 10:00 UTC bar is the overnight Globex print and is.
    """
    w = snap.pivot(index="date", columns="series", values="value").sort_index()
    need = {"cbot_corn_safexclose", "usdzar_safexclose"}
    if not need.issubset(w.columns):
        return pd.Series(dtype=float, name="world_rand")
    s = (w["cbot_corn_safexclose"] * BU_PER_TONNE * w["usdzar_safexclose"]).dropna()
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    return s.rename("world_rand")


def panel_parity(sd: pd.DataFrame, cont: pd.DataFrame, snap: pd.DataFrame, cpi: pd.DataFrame,
                 grain_class: str, symbol: str) -> pd.DataFrame:
    """Model D input: y = log(SAFEX / world parity) — the basis; x = log cover.

    Forward outcome is the *relative* return, SAFEX minus world parity, because the basis is a
    relative-value quantity: it converges by SAFEX moving toward parity or parity moving toward SAFEX.
    Outright forward returns are kept alongside as `fwd_outright_*`.
    """
    world = world_parity(snap)
    if world.empty:
        return pd.DataFrame()
    p = panel_price(sd, cont, cpi, grain_class, symbol)
    p = p.rename(columns={f"fwd_{h}": f"fwd_outright_{h}" for h in HORIZONS})
    p["trade_date"] = p["trade_date"].astype("datetime64[ns]")
    wd = world.reset_index()
    wd.columns = ["date", "world_rand"]
    p = pd.merge_asof(p.sort_values("trade_date"), wd, left_on="trade_date", right_on="date",
                      direction="backward", tolerance=pd.Timedelta("5D"))
    p = p.dropna(subset=["world_rand", "close_1"]).reset_index(drop=True)
    p["basis"] = np.log(p["close_1"] / p["world_rand"])
    p["y"] = p["basis"]
    p["x"] = np.log(p["months_cover"])
    fw = forward_returns(world, p["vintage_date"])
    for h in HORIZONS:
        p[f"fwd_world_{h}"] = fw[f"fwd_{h}"].to_numpy()
        p[f"fwd_{h}"] = p[f"fwd_outright_{h}"] - p[f"fwd_world_{h}"]
    return p


def panel_price_with_world(sd: pd.DataFrame, cont: pd.DataFrame, snap: pd.DataFrame, cpi: pd.DataFrame,
                           grain_class: str, symbol: str) -> pd.DataFrame:
    """Model A augmented with the contemporaneous world price in real rand (`lw`).

    Kept to document a tested-and-rejected specification: it raises R² sharply and lowers forward IC.
    """
    world = world_parity(snap)
    if world.empty:
        return pd.DataFrame()
    p = panel_price(sd, cont, cpi, grain_class, symbol)
    p["trade_date"] = p["trade_date"].astype("datetime64[ns]")
    wd = world.reset_index()
    wd.columns = ["date", "world_rand"]
    p = pd.merge_asof(p.sort_values("trade_date"), wd, left_on="trade_date", right_on="date",
                      direction="backward", tolerance=pd.Timedelta("5D"))
    p = p.dropna(subset=["world_rand", "y"]).reset_index(drop=True)
    p["lw"] = np.log(p["world_rand"] * (p["real_px"] / p["close_1"]))   # same CPI deflator as y
    return p


def decompose_world_r2(paw: pd.DataFrame, snap: pd.DataFrame) -> dict:
    """Where does the R² gain from the world-parity term actually come from?

    World parity is CBOT corn x USD/ZAR. Both SAFEX and parity are quoted in rand, so a large part
    of any common variation is the currency, not the grain. This splits the two legs and also
    reports the nominal specification, where the shared inflation trend is still present.
    """
    w = snap.pivot(index="date", columns="series", values="value").sort_index().reset_index()
    w["date"] = w["date"].astype("datetime64[ns]")
    p = pd.merge_asof(paw.sort_values("trade_date"), w, left_on="trade_date", right_on="date",
                      direction="backward", tolerance=pd.Timedelta("5D"))
    p = p.dropna(subset=["y", "x", "lw", "cbot_corn_safexclose", "usdzar_safexclose"]).reset_index(drop=True)
    if len(p) < 50:
        return {}

    def r2(cols: list[np.ndarray], target: np.ndarray) -> float:
        X = np.column_stack(cols)
        beta, _ = _ols(X, target)
        r = target - X @ beta
        return float(1 - (r @ r) / ((target - target.mean()) ** 2).sum())

    y_real = p["y"].to_numpy()
    y_nom = np.log(p["close_1"].to_numpy())
    lw_nom = np.log(p["world_rand"].to_numpy())
    lc, lf = np.log(p["cbot_corn_safexclose"].to_numpy()), np.log(p["usdzar_safexclose"].to_numpy())
    S, one = fourier(p["my_month"]), np.ones((len(p), 1))
    x = p["x"].to_numpy()
    return {
        "n": len(p),
        "real_base": r2([S, x], y_real), "real_world": r2([S, x, p["lw"].to_numpy()], y_real),
        "nom_base": r2([S, x], y_nom), "nom_world": r2([S, x, lw_nom], y_nom),
        "nom_zar_only": r2([one, lf], y_nom), "nom_cbot_only": r2([one, lc], y_nom),
        "nom_both_free": r2([one, lc, lf], y_nom),
        "real_world_only": r2([one, p["lw"].to_numpy()], y_real),
        "sd_log_cbot": float(lc.std()), "sd_log_zar": float(lf.std()),
        "corr_cbot_zar": float(np.corrcoef(lc, lf)[0, 1]),
    }
