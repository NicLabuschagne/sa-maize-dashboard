"""Unit tests for research/band_position: band arithmetic, point-in-time rules, nowcast, models."""
import numpy as np
import pandas as pd
import pytest

from research.band_position import band, models, stocks
from research.band_position.inputs import Inputs, inputs_as_of
from research.band_position.settings import config_hash, load_settings


# ----------------------------------------------------------------------------- settings
def test_settings_load_and_hash_is_stable() -> None:
    settings = load_settings()
    assert settings["band"]["export_quantile"] < settings["band"]["import_quantile"]
    assert settings["stocks"]["primary"] in settings["stocks"]["variants"]
    assert config_hash() == config_hash()


# ----------------------------------------------------------------------------- band
def test_world_price_converts_bushels_and_currency() -> None:
    snapshots = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "cbot_usd_per_bushel": [4.0],
                              "usdzar": [15.0]})
    world = band.world_price_rand(snapshots, 39.3683)
    assert np.isclose(world.iloc[0], 4.0 * 39.3683 * 15.0)


def test_band_position_is_zero_and_one_at_edges_and_not_clipped() -> None:
    price = pd.Series([100.0, 200.0, 50.0, 260.0])
    position = band.band_position(price, pd.Series([100.0] * 4), pd.Series([200.0] * 4))
    assert position.tolist() == [0.0, 1.0, -0.5, 1.6]


def _synthetic_band_frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    world = 2000 + rng.normal(0, 50, n).cumsum()
    safex = world * np.exp(rng.normal(0.2, 0.1, n))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "safex": safex, "world": world,
                         "basis": np.log(safex / world), "cost_width": np.full(n, 1500.0)})


def test_edges_on_day_t_use_only_earlier_days() -> None:
    frame = _synthetic_band_frame()
    edges = band.prior_day_edges(frame, 0.05, 0.95, 20)
    shocked = frame.copy()
    shocked.loc[40:, "safex"] *= 3                       # change day 40 onwards
    shocked["basis"] = np.log(shocked["safex"] / shocked["world"])
    edges_shocked = band.prior_day_edges(shocked, 0.05, 0.95, 20)
    columns = ["hybrid_export", "hybrid_import", "implied_import"]
    pd.testing.assert_frame_equal(edges.loc[:40, columns], edges_shocked.loc[:40, columns])
    assert edges.loc[:19, "hybrid_export"].isna().all()   # no value before min history


def test_hybrid_ceiling_is_floor_plus_scaled_cost_width() -> None:
    edges = band.prior_day_edges(_synthetic_band_frame(), 0.05, 0.95, 20)
    row = edges.iloc[-1]
    assert np.isclose(row.hybrid_import - row.hybrid_export, row.excess_quantile * row.cost_width)


def test_outside_band_summary_counts_spells() -> None:
    position = pd.Series([0.5, -0.1, -0.2, 0.3, 1.2, 0.4, 1.1, 1.3, 1.4])
    summary = band.outside_band_summary(position)
    assert summary["spells_below"] == 1 and summary["longest_spell_below"] == 2
    assert summary["spells_above"] == 2 and summary["longest_spell_above"] == 3
    assert np.isclose(summary["share_above"], 4 / 9)


# ----------------------------------------------------------------------------- stocks
def _release(closing: float = 1000.0) -> pd.Series:
    return pd.Series({"month_end": pd.Timestamp("2020-01-31"), "closing_stock": closing,
                      "utilisation_12m": 1200.0})


def _flows() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-24", "2020-02-07", "2020-02-14"])
    return pd.DataFrame({"deliveries": [5.0, 10.0, 20.0], "imports": [0.0, 1.0, 2.0], "exports": [0.0, 3.0, 4.0],
                         "available_date": pd.to_datetime(["2020-02-05", "2020-02-19", "2020-02-26"]),
                         "has_deliveries": [True, True, True]}, index=index)


def test_nowcast_adds_published_weeks_after_month_end_only() -> None:
    stock, weeks = stocks.nowcast_stock(_release(), _flows(), pd.Timestamp("2020-02-20"), None, 30.44)
    # week of 24 Jan is inside the reported month; week of 14 Feb is not yet published
    assert weeks == 1
    assert np.isclose(stock, 1000 + 10 + 1 - 3 - 1200 / 12 * 7 / 30.44)


def test_nowcast_is_nan_when_trade_published_without_deliveries() -> None:
    flows = _flows()
    flows.loc[pd.Timestamp("2020-02-07"), "has_deliveries"] = False
    stock, _ = stocks.nowcast_stock(_release(), flows, pd.Timestamp("2020-02-20"), None, 30.44)
    assert np.isnan(stock)


def test_daily_ratios_are_not_backfilled() -> None:
    events = pd.DataFrame({"date": pd.to_datetime(["2020-02-10"]), "stu_total": [0.3]})
    daily = stocks.daily_stocks_to_use(pd.Series(pd.to_datetime(["2020-02-07", "2020-02-10", "2020-02-11"])), events)
    assert np.isnan(daily["stu_total"].iloc[0])
    assert daily["stu_total"].iloc[1:].tolist() == [0.3, 0.3]


# ----------------------------------------------------------------------------- inputs
def test_inputs_as_of_drops_unpublished_rows() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-10"])
    frame = pd.DataFrame({"trade_date": dates, "date": dates, "available_date": dates, "vintage_date": dates})
    cut = inputs_as_of(Inputs(frame, frame, frame, frame, frame, frame), pd.Timestamp("2020-01-05"))
    for table in (cut.prices, cut.snapshots, cut.parity, cut.balance_sheet, cut.weekly, cut.cpi):
        assert len(table) == 1


# ----------------------------------------------------------------------------- models
def _weekly_sample(n: int = 400, slope: float = -0.4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    stu = np.exp(rng.normal(np.log(0.4), 0.4, n))
    position = 0.3 + slope * np.log(stu) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"date": dates, "stu": stu, "position": position})


def test_log_model_recovers_slope() -> None:
    fitted = models.fit_model("log", _weekly_sample(), "stu", harmonics=2)
    assert abs(fitted.slope + 0.4) < 0.03


def test_logistic_and_isotonic_fit_and_isotonic_is_non_increasing() -> None:
    sample = _weekly_sample()
    sample["position"] = 1 / (1 + np.exp(-(-1.0 - 2.0 * np.log(sample["stu"]))))
    assert models.fit_model("logistic", sample, "stu", 2).slope < 0
    isotonic = models.fit_model("isotonic", sample, "stu", 2)
    grid = pd.DataFrame({"stu": np.linspace(0.1, 1.5, 50)})
    assert np.all(np.diff(isotonic.predict(grid)) <= 1e-12)


def test_expanding_prediction_ignores_future_weeks() -> None:
    sample = _weekly_sample()
    first = sample["date"].iloc[200]
    base = models.expanding_predictions(sample, "log", "stu", first, 2)
    shocked = sample.copy()
    shocked.loc[301:, "position"] += 5
    after = models.expanding_predictions(shocked, "log", "stu", first, 2)
    assert np.allclose(base["fair_position"].iloc[:302], after["fair_position"].iloc[:302], equal_nan=True)
    assert base["fair_position"].iloc[:200].isna().all()


def test_first_out_of_sample_date_counts_complete_marketing_years() -> None:
    assert models.first_out_of_sample_date(pd.Timestamp("2011-06-01"), 5) == pd.Timestamp("2017-05-01")
    assert models.first_out_of_sample_date(pd.Timestamp("2011-05-01"), 5) == pd.Timestamp("2016-05-01")


def test_regime_uses_previous_week() -> None:
    labels = models.regime_labels(pd.Series([0.5, 0.9, 0.5, 0.1]), 0.2, 0.8)
    assert pd.isna(labels.iloc[0])
    assert labels.iloc[1:].tolist() == ["mid", "edge", "mid"]


def test_out_of_sample_r2_is_one_for_perfect_and_zero_for_benchmark() -> None:
    actual = pd.Series(np.arange(20, dtype=float))
    benchmark = pd.Series(np.full(20, 5.0))
    assert models.out_of_sample_r2(actual, actual, benchmark) == pytest.approx(1.0)
    assert models.out_of_sample_r2(actual, benchmark, benchmark) == pytest.approx(0.0)
