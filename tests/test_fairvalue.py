import numpy as np
import pandas as pd
import pytest

from app.data import fairvalue as FV


def _synthetic_panel(n: int = 160, seed: int = 0) -> pd.DataFrame:
    """y = 5 − 0.3·x + seasonal + noise, with x log-cover-like and dates monthly."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-01", periods=n, freq="MS")
    my_month = ((months.month - 5) % 12) + 1
    x = rng.normal(0, 0.5, n)
    season = 0.1 * np.sin(2 * np.pi * (my_month - 1) / 12)
    y = 5 - 0.3 * x + season + rng.normal(0, 0.05, n)
    p = pd.DataFrame({"vintage_date": months, "latest_month": months, "my_month": my_month, "x": x, "y": y,
                      "marketing_year": "n/a"})
    for h in FV.HORIZONS:
        p[f"fwd_{h}"] = rng.normal(0, 0.05, n)
    return p


def test_fourier_shape_and_periodicity() -> None:
    X = FV.fourier(pd.Series([1, 13, 7]))
    assert X.shape == (3, 5)
    assert np.allclose(X[0], X[1])            # month 13 ≡ month 1
    assert np.allclose(X[:, 0], 1.0)


def test_fit_expanding_recovers_slope_and_is_out_of_sample() -> None:
    p = _synthetic_panel()
    fit = FV.fit_expanding(p, min_obs=48)
    assert abs(fit.coef_full["x"] + 0.3) < 0.03
    assert fit.r2_full > 0.9
    q = fit.panel
    assert q.fv.iloc[:48].isna().all() and q.fv.iloc[48:].notna().all()
    assert q.z.iloc[48 + 12:].notna().all()


def test_forward_returns_uses_entry_lag_and_roll_adjusted_index() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    cont = pd.DataFrame({"symbol": "WMAZ", "trade_date": dates,
                         "log_ret_1": [np.nan] + [0.01] * 29, "expiry_1": "2020-03"})
    idx = FV.roll_adjusted_index(cont, "WMAZ")
    fr = FV.forward_returns(idx, pd.Series([dates[0]]), {"5d": 5}, lag_days=1)
    assert np.isclose(fr["fwd_5d"].iloc[0], 0.05)   # entered on dates[1], five 1% steps


def test_block_bootstrap_p_detects_signal_and_null() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    assert FV._block_bootstrap_p(a, -0.5 * a + rng.normal(size=200)) < 0.01
    assert FV._block_bootstrap_p(a, rng.normal(size=200)) > 0.05


def test_ic_and_tercile_tables_shape() -> None:
    p = _synthetic_panel()
    q = FV.fit_expanding(p).panel
    ic = FV.ic_table(q, {"z": "z"})
    assert set(ic.horizon) == set(FV.HORIZONS) and ic.IC.notna().all()
    t = FV.tercile_table(q)
    assert t.bucket.nunique() == 3 and len(t) == 3 * len(FV.HORIZONS)


def test_half_life_finite_for_mean_reverting_residual() -> None:
    fit = FV.fit_expanding(_synthetic_panel())
    assert 0 < fit.half_life_months < 12
