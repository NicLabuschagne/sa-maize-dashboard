"""Trend replication: the signals must behave, and the COT validation must be honest."""
import numpy as np
import pandas as pd
import pytest

from app.data import trend as T
from config import DB_PATH


def _series(n: int = 900, drift: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n)).as_unit("ns")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=dates)


def test_signals_are_bounded_and_named() -> None:
    panel = T.trend_panel(_series())
    assert panel.shape[1] == len(T.CROSSOVERS) + len(T.MOMENTUM)
    v = panel.to_numpy()
    v = v[np.isfinite(v)]
    assert v.min() >= -1.0 and v.max() <= 1.0


def test_persistent_trend_is_held_on_average_across_paths() -> None:
    """Systematic behaviour, not one path: any single realisation can consolidate at the end,
    and a trend model going flat through a consolidation is correct, not a failure."""
    ups = [T.aggregate(T.trend_panel(_series(drift=0.0015, seed=s))).dropna().mean()
           for s in range(6)]
    dns = [T.aggregate(T.trend_panel(_series(drift=-0.0015, seed=s))).dropna().mean()
           for s in range(6)]
    assert np.mean(ups) > 0.3 and all(u > 0 for u in ups)
    assert np.mean(dns) < -0.3 and all(d < 0 for d in dns)


def test_flat_market_is_near_zero() -> None:
    flat = T.aggregate(T.trend_panel(_series(drift=0.0, seed=4))).dropna()
    assert abs(flat.tail(100).mean()) < 0.6      # noise, but no persistent bias


def test_dispersion_counts_add_up() -> None:
    panel = T.trend_panel(_series())
    d = T.dispersion(panel).dropna()
    assert (d.n_long + d.n_short + d.n_flat == d.n).all()
    assert d.agreement.between(-1, 1).all()


def test_latest_state_has_one_row_per_signal() -> None:
    panel = T.trend_panel(_series())
    last = T.latest_state(panel)
    assert len(last) == panel.shape[1]
    assert set(last.columns) == {"signal", "position"}


def test_validate_returns_not_ok_on_a_short_sample() -> None:
    trend = T.aggregate(T.trend_panel(_series(n=300)))
    cot = pd.DataFrame({"date": pd.date_range("2015-01-06", periods=10, freq="W-TUE"),
                        "net_noncomm_pct_oi": np.linspace(-0.1, 0.2, 10)})
    assert T.validate_against_cot(trend, cot)["ok"] is False


def test_validate_recovers_a_planted_relationship() -> None:
    """If reported positioning IS the trend signal, the validation must find it."""
    px = _series(drift=0.0008, seed=9)
    trend = T.aggregate(T.trend_panel(px)).dropna()
    weekly = trend.resample("W-TUE").last().dropna()
    cot = pd.DataFrame({"date": weekly.index, "net_noncomm_pct_oi": weekly.to_numpy() * 0.3})
    out = T.validate_against_cot(trend, cot)
    assert out["ok"] and out["corr_level"] > 0.95


def test_validate_finds_nothing_in_noise() -> None:
    trend = T.aggregate(T.trend_panel(_series(seed=2))).dropna()
    rng = np.random.default_rng(11)
    weekly = trend.resample("W-TUE").last().dropna()
    cot = pd.DataFrame({"date": weekly.index,
                        "net_noncomm_pct_oi": rng.normal(size=len(weekly))})
    out = T.validate_against_cot(trend, cot)
    assert out["ok"] and abs(out["corr_level"]) < 0.4


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
def test_corn_validation_holds_on_real_data() -> None:
    """Regression guard on the headline claim made on the Positioning page."""
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "cot" not in tables:
            pytest.skip("COT not fetched")
        m = con.execute("SELECT date, value FROM macro WHERE series='cbot_corn' ORDER BY date").df()
        cot = con.execute("SELECT * FROM cot WHERE symbol='ZC'").df()
    finally:
        con.close()
    px = pd.Series(m.value.to_numpy(), index=pd.DatetimeIndex(m.date).as_unit("ns"))
    out = T.validate_against_cot(T.aggregate(T.trend_panel(px)), cot)
    assert out["ok"] and out["n"] > 500
    assert out["corr_level"] > 0.6, "method no longer recovers reported corn positioning"
    assert out["corr_change"] > 0.5, "weekly-change tracking has degraded"


def test_price_benchmark_detects_pure_momentum_relabelled() -> None:
    """If reported positioning IS a price return, the trend model must add nothing on top."""
    px = _series(drift=0.0006, seed=21, n=1400)
    lp = np.log(px)
    ret12 = (lp - lp.shift(252)).dropna()
    weekly = ret12.resample("W-TUE").last().dropna()
    cot = pd.DataFrame({"date": weekly.index, "net_noncomm_pct_oi": weekly.to_numpy()})
    b = T.price_benchmark(px, cot)
    assert b["ok"]
    assert b["r2_ret12"] > 0.95                       # returns explain it, as constructed
    assert abs(b["partial_trend"]) < 0.5              # little left for the trend model


def test_price_benchmark_reports_all_keys() -> None:
    px = _series(n=1400, seed=3)
    weekly = T.aggregate(T.trend_panel(px)).dropna().resample("W-TUE").last().dropna()
    cot = pd.DataFrame({"date": weekly.index, "net_noncomm_pct_oi": weekly.to_numpy()})
    b = T.price_benchmark(px, cot)
    for k in ("corr", "r2_trend", "r2_rets", "r2_both", "partial_trend", "dchg_trend", "dchg_ret3m"):
        assert k in b
    assert b["r2_both"] >= b["r2_rets"] - 1e-9        # adding a regressor cannot lower R²


def test_dynamics_reports_sane_numbers() -> None:
    d = T.dynamics(_series(n=1400, seed=5))
    assert 0 < d["ann_vol"] < 2
    assert -1 <= d["ac1"] <= 1
    assert d["flips_per_year"] >= 0
    assert 0 <= d["mean_abs_trend"] <= 1
    assert 0 <= d["pct_conviction"] <= 1


def test_dynamics_sees_a_trending_market_as_more_persistent() -> None:
    """A strongly drifting series should flip less often than a random walk."""
    trending = T.dynamics(_series(n=1600, drift=0.0015, seed=6))
    choppy = T.dynamics(_series(n=1600, drift=0.0, seed=6))
    assert trending["flips_per_year"] < choppy["flips_per_year"]
    assert trending["mean_abs_trend"] > choppy["mean_abs_trend"]


def _cot_from(agg: pd.Series, scale: float = 0.3, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Weekly Tuesday report built from a model path, released three days later."""
    rng = np.random.default_rng(seed)
    w = agg.dropna().resample("W-TUE").last().dropna()
    v = scale * w.to_numpy() + rng.normal(0, noise, len(w))
    return pd.DataFrame({"date": w.index, "release_date": w.index + pd.Timedelta(days=3),
                         "net_noncomm_pct_oi": v})


def test_flow_is_the_change_in_position() -> None:
    agg = T.aggregate(T.trend_panel(_series(n=600)))
    f = T.flow(agg, window=5).dropna()
    assert np.allclose(f, (agg - agg.shift(5)).dropna())


def test_flow_state_reports_direction() -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=60)).as_unit("ns")
    rising = pd.Series(np.linspace(-0.8, 0.8, 60), index=idx)
    falling = pd.Series(np.linspace(0.8, -0.8, 60), index=idx)
    assert T.flow_state(rising)["direction"] == "buying"
    assert T.flow_state(falling)["direction"] == "selling"
    assert T.flow_state(pd.Series(np.zeros(60), index=idx))["direction"] == "flat"


def test_anchoring_corrects_level_bias_the_model_cannot_see() -> None:
    """Anchoring is a bias correction. Give reported positioning a slow drift the model has no
    way to know about - as real positioning has - and the anchor must recover it."""
    px = _series(n=1600, drift=0.0004, seed=31)
    agg = T.aggregate(T.trend_panel(px))
    cot = _cot_from(agg, noise=0.01, seed=1)
    rng = np.random.default_rng(5)
    bias = np.cumsum(rng.normal(0, 0.012, len(cot)))        # slow wander, invisible to the model
    cot["net_noncomm_pct_oi"] = cot["net_noncomm_pct_oi"] + bias
    sc = T.nowcast_scorecard(T.anchored_nowcast(agg, cot), cot)
    assert sc["ok"]
    assert sc["mae_anchored"] < sc["mae_model_only"]
    assert sc["improvement"] > 0.2


def test_anchoring_adds_noise_when_the_model_is_already_unbiased() -> None:
    """The mirror case, documenting the mechanism: with no bias to correct, anchoring only
    re-imports the previous report's measurement error."""
    px = _series(n=1600, drift=0.0004, seed=31)
    agg = T.aggregate(T.trend_panel(px))
    cot = _cot_from(agg, noise=0.02, seed=1)                # pure function of the model, plus noise
    sc = T.nowcast_scorecard(T.anchored_nowcast(agg, cot), cot)
    assert sc["ok"] and sc["improvement"] < 0


def test_anchor_only_uses_released_reports() -> None:
    """The anchor must not update before the release date - that would be look-ahead."""
    px = _series(n=1200, seed=17)
    agg = T.aggregate(T.trend_panel(px))
    cot = _cot_from(agg, seed=2)
    nc = T.anchored_nowcast(agg, cot).dropna(subset=["anchor_date"])
    assert (nc.index >= nc.anchor_date).all()
    assert nc.days_since_report.min() >= 3          # never fresher than the publication lag


def test_nowcast_handles_missing_release_dates() -> None:
    px = _series(n=1200, seed=19)
    agg = T.aggregate(T.trend_panel(px))
    cot = _cot_from(agg, seed=3)
    cot.loc[cot.index[:20], "release_date"] = pd.NaT
    nc = T.anchored_nowcast(agg, cot)
    assert nc.nowcast.notna().sum() > 500


def test_nowcast_scorecard_needs_a_sample() -> None:
    px = _series(n=400, seed=23)
    agg = T.aggregate(T.trend_panel(px))
    cot = _cot_from(agg, seed=4).head(10)
    assert T.nowcast_scorecard(T.anchored_nowcast(agg, cot), cot)["ok"] is False
