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


# ----------------------------------------------------------------------------- addendum 1
def _contracts() -> pd.DataFrame:
    """Two dates; on the first, contracts at 60 and 120 days bracket the 90-day tenor."""
    rows = [("2020-01-02", "2020-03-02", 60, 100.0), ("2020-01-02", "2020-05-01", 120, 400.0),
            ("2020-01-03", "2020-05-01", 95, 200.0), ("2020-01-03", "2020-07-01", 150, 300.0)]
    frame = pd.DataFrame(rows, columns=["trade_date", "expiry_date", "days_to_expiry", "close"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"])
    frame["symbol"] = "YMAZ"
    return frame


def test_constant_maturity_interpolates_log_price_in_days() -> None:
    series = band.constant_maturity(_contracts(), "YMAZ", 90, 7).set_index("date")["safex"]
    assert np.isclose(series.iloc[0], np.exp(0.5 * np.log(100) + 0.5 * np.log(400)))  # halfway: 200
    assert np.isclose(series.iloc[1], 200.0)   # every contract beyond the tenor: nearest one


def test_score_windows_only_selects_predictions() -> None:
    dates = pd.date_range("2019-12-06", periods=60, freq="W-FRI")
    predictions = pd.DataFrame({"date": dates, "position": np.linspace(0, 1, 60),
                                "fair_position": np.linspace(0, 1, 60), "naive": 0.5})
    table = models.score_windows(predictions, [["2020-06-01", "2020-12-31"]]).set_index("period")
    assert table.loc["2020", "weeks"] == 52
    assert table.loc["2020-06 to 2020-12", "oos_r2"] == pytest.approx(1.0)


def test_ic_statistics_detects_signal_and_reports_effective_n() -> None:
    from research.band_position import tradeability

    rng = np.random.default_rng(0)
    signal = pd.Series(rng.normal(size=2000))
    outcome = 0.3 * signal + pd.Series(rng.normal(size=2000))
    result = tradeability.ic_statistics(signal, outcome, horizon=10)
    assert result["ic"] > 0.2 and result["nw_t"] > 3
    assert result["effective_n"] == pytest.approx(200)


def test_daily_fair_position_uses_only_earlier_weeks() -> None:
    from research.band_position import tradeability

    days = pd.bdate_range("2012-01-02", periods=1500)
    rng = np.random.default_rng(3)
    stu = np.exp(rng.normal(np.log(0.4), 0.3, len(days)))
    daily = pd.DataFrame({"date": days, "safex": 100.0, "position": 0.3 - 0.4 * np.log(stu),
                          "hybrid_export": 90.0, "hybrid_import": 110.0, "stu": stu})
    first = pd.Timestamp("2016-01-01")
    base = tradeability.daily_fair_position(daily, "stu", "linear", first, 2)
    shocked = daily.copy()
    cut = shocked["date"] > pd.Timestamp("2016-06-03")        # a Friday
    shocked.loc[cut, "position"] += 10
    after = tradeability.daily_fair_position(shocked, "stu", "linear", first, 2)
    upto = base["date"] <= pd.Timestamp("2016-06-10")          # the week after is fitted on data to 6/3
    assert np.allclose(base.loc[upto, "fair_position"], after.loc[upto, "fair_position"], equal_nan=True)
    # the first out-of-sample week can start a few days before `first`; nothing earlier is predicted
    assert base.loc[base["date"] < first - pd.Timedelta(days=7), "fair_position"].isna().all()


# ----------------------------------------------------------------------------- addendum 2: revealed floors
from research.band_position import revealed  # noqa: E402


def _weekly_rows(flow: str, week_ends: list[str], tons: list[float], lag_days: int = 12) -> pd.DataFrame:
    week_end = pd.to_datetime(week_ends)
    return pd.DataFrame({"grain_class": "yellow", "flow": flow, "week_end": week_end, "tons_week": tons,
                         "available_date": week_end + pd.Timedelta(days=lag_days),
                         "season": "2020/21"})


def test_published_pace_only_uses_published_weeks() -> None:
    weekly = _weekly_rows("exports", ["2020-06-05", "2020-06-12"], [40_000.0, 20_000.0])
    dates = pd.Series(pd.to_datetime(["2020-06-16", "2020-06-17", "2020-06-24"]))
    pace = revealed.published_pace(weekly, "yellow", "exports", dates, n_weeks=4)
    assert np.isnan(pace.iloc[0])                  # first week published on 17 June
    assert pace.iloc[1] == pytest.approx(40.0)
    assert pace.iloc[2] == pytest.approx(30.0)


def _harvest_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-05-01", "2020-09-30")
    basis = np.where(dates.month <= 7, 0.05, 0.30)
    return pd.DataFrame({"date": dates, "basis": basis})


def test_season_rule_needs_export_season_and_waits_for_publication() -> None:
    frame = _harvest_frame()
    weeks = [str(d.date()) for d in pd.date_range("2020-05-01", "2020-07-31", freq="W-FRI")]
    exporting = pd.concat([_weekly_rows("exports", weeks, [50_000.0] * len(weeks)),
                           _weekly_rows("imports", weeks, [1_000.0] * len(weeks))])
    floor = revealed.season_rule_floor_basis(frame, exporting, "yellow", 0.10)
    usable_from = pd.Timestamp(weeks[-1]) + pd.Timedelta(days=12)
    assert floor[frame["date"] < usable_from].isna().all()
    assert np.allclose(floor[frame["date"] >= usable_from], 0.05)
    importing = pd.concat([_weekly_rows("exports", weeks, [1_000.0] * len(weeks)),
                           _weekly_rows("imports", weeks, [50_000.0] * len(weeks))])
    assert revealed.season_rule_floor_basis(frame, importing, "yellow", 0.10).isna().all()


def _kalman_frame(z: np.ndarray, start: str = "2020-08-03") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(z))
    world = np.full(len(z), 3000.0)
    costs = np.full(len(z), 500.0)
    return pd.DataFrame({"date": dates, "world": world, "export_deductions": costs,
                         "safex": world * np.exp(z) - costs})


def test_kalman_ignores_high_prices_without_exports_but_follows_prices_below() -> None:
    params = revealed.KalmanSettings()
    z = np.r_[np.full(50, 0.20), np.full(50, 0.40)]          # price rises, no exports
    out = revealed.kalman_floor(_kalman_frame(z), pd.Series(np.zeros(100)), params)
    assert out["state"].iloc[-1] == pytest.approx(0.20, abs=1e-9)
    z = np.r_[np.full(50, 0.20), np.full(50, 0.05)]          # price falls below the floor
    out = revealed.kalman_floor(_kalman_frame(z), pd.Series(np.zeros(100)), params)
    assert out["state"].iloc[-1] < 0.08


def test_kalman_follows_high_prices_when_exports_flow() -> None:
    z = np.r_[np.full(50, 0.20), np.full(150, 0.30)]
    out = revealed.kalman_floor(_kalman_frame(z), pd.Series(np.full(200, 40.0)), revealed.KalmanSettings())
    assert out["state"].iloc[-1] > 0.28


def test_kalman_floor_on_day_t_ignores_day_t_price() -> None:
    z = np.full(80, 0.20)
    base = revealed.kalman_floor(_kalman_frame(z), pd.Series(np.full(80, 40.0)), revealed.KalmanSettings())
    z_shocked = z.copy()
    z_shocked[60] = -0.5
    shocked = revealed.kalman_floor(_kalman_frame(z_shocked), pd.Series(np.full(80, 40.0)), revealed.KalmanSettings())
    assert base["floor"].iloc[60] == pytest.approx(shocked["floor"].iloc[60])
    assert shocked["floor"].iloc[61] < base["floor"].iloc[61]


def test_kalman_uncertainty_jumps_at_new_marketing_year() -> None:
    z = np.full(60, 0.20)
    out = revealed.kalman_floor(_kalman_frame(z, start="2021-03-01"), pd.Series(np.full(60, 40.0)),
                                revealed.KalmanSettings())
    dates = pd.bdate_range("2021-03-01", periods=60)
    first_may = np.flatnonzero(dates >= "2021-05-01")[0]
    assert out["state_sd"].iloc[first_may] > 5 * out["state_sd"].iloc[first_may - 1]


def test_season_rule_publishes_nothing_from_a_partial_harvest_window() -> None:
    frame = _harvest_frame()
    frame = frame[frame["date"] <= "2020-06-30"]                  # data cut mid-harvest
    weeks = [str(d.date()) for d in pd.date_range("2020-05-01", "2020-06-19", freq="W-FRI")]
    exporting = pd.concat([_weekly_rows("exports", weeks, [50_000.0] * len(weeks)),
                           _weekly_rows("imports", weeks, [1_000.0] * len(weeks))])
    assert revealed.season_rule_floor_basis(frame, exporting, "yellow", 0.10).isna().all()


def test_border_baseline_uses_first_published_months_only() -> None:
    months = pd.date_range("2019-01-01", periods=12, freq="MS")
    rows = [{"period_type": "latest_month", "is_final": False, "attribute": "exports_whole_border",
             "grain_class": "yellow", "latest_month": m, "vintage_date": m + pd.Timedelta(days=55),
             "value_t": 52_180.0} for m in months]
    # a later release revises March 2019 upward; the baseline must keep the first-published value
    rows.append({**rows[2], "vintage_date": months[-1] + pd.Timedelta(days=60), "value_t": 999_999.0})
    balance_sheet = pd.DataFrame(rows)
    dates = pd.Series(pd.to_datetime(["2019-06-01", "2020-03-01"]))
    baseline = revealed.border_baseline(balance_sheet, "yellow", dates)
    assert np.isnan(baseline.iloc[0])                     # fewer than 9 months published by then
    assert baseline.iloc[1] == pytest.approx(12.0)        # 12 x 52 180 t / 52.18 weeks = 12 kt/week


# ----------------------------------------------------------------------------- addendum 4: edge test
from research.band_position import edge_test  # noqa: E402
from app.data.backtest import CostModel  # noqa: E402


def _edge_frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "arb_ret": np.full(n, 0.01),
                         "usdzar": 15.0, "safex": 3000.0, "roll_day": False})


def test_backtest_earns_from_the_day_after_execution() -> None:
    frame = _edge_frame()
    target = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)   # decided at closes 2, 3, 4
    daily = edge_test.backtest_rule(frame, target, CostModel())
    # traded at closes 3–5, so it earns the returns of days 4, 5, 6
    assert daily["held"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert daily["gross"].sum() == pytest.approx(0.03)


def test_backtest_charges_entry_exit_and_rolls() -> None:
    frame = _edge_frame()
    frame.loc[5, "roll_day"] = True
    target = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    costs = CostModel()
    daily = edge_test.backtest_rule(frame, target, costs)
    one_way = (costs.one_way_r_t + 0.25 * 15.0) / 3000.0
    assert daily.loc[3, "cost"] == pytest.approx(one_way)            # entry at close 3
    assert daily.loc[6, "cost"] == pytest.approx(one_way)            # exit at close 6
    assert daily.loc[5, "cost"] == pytest.approx(costs.round_trip_r_t / 3000.0)   # roll while held
    assert daily["cost"].sum() == pytest.approx(2 * one_way + costs.round_trip_r_t / 3000.0)


def test_holm_and_spell_count() -> None:
    assert edge_test.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert edge_test.spell_count(pd.Series([False, True, True, False, True])) == 2


# ----------------------------------------------------------------------------- addendum 5: extremes
from research.band_position import extremes  # noqa: E402


def _extreme_frame(position: list[float], fair: float = 0.5) -> pd.DataFrame:
    n = len(position)
    frame = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "position": position,
                          "fair_position": fair, "export_edge": 1000.0, "band_width": 1000.0,
                          "arb_ret": 0.01, "safex_ret": 0.02})
    frame["gap"] = frame["fair_position"] - frame["position"]
    frame["safex"] = frame["export_edge"] + frame["position"] * frame["band_width"]
    return frame


def test_long_events_need_rearming_and_are_classified_by_gap() -> None:
    frame = _extreme_frame([0.3, 0.05, 0.15, 0.05, 0.25, 0.05])
    events = extremes.find_events(frame, "long")
    # second dip (row 3) is ignored: position never got back above 0.2 in between
    assert events["row"].tolist() == [1, 5]
    assert events["confirmed"].all()                 # gap 0.45 >= 0.2
    frame["fair_position"] = 0.1
    frame["gap"] = frame["fair_position"] - frame["position"]
    assert not extremes.find_events(frame, "long")["confirmed"].any()


def test_short_outcomes_are_sign_adjusted_and_position_change_is_split() -> None:
    frame = _extreme_frame([0.5, 0.95] + [0.95] * 70)
    events = extremes.find_events(frame, "short")
    out = extremes.event_outcomes(frame, events)
    assert out["fwd_arb_20"].iloc[0] == pytest.approx(-0.20)       # arb rose 1%/day; short loses
    assert out["position_change_20"].iloc[0] == pytest.approx(0.0)
    assert out["from_safex_20"].iloc[0] + out["from_edges_20"].iloc[0] == pytest.approx(out["position_change_20"].iloc[0])


# ----------------------------------------------------------------------------- addendum 6: z combo
from research.band_position import zcombo  # noqa: E402


def test_zcombo_entries_rearm_after_five_days_off() -> None:
    mask = pd.Series([False, True, True, False, False, True, False, False, False, False, False, True])
    assert zcombo.entries(mask).tolist() == [1, 11]      # row 5 is within 5 days of row 2


def test_gap_z_is_standardised_on_the_trailing_window() -> None:
    gap = pd.Series(np.r_[np.zeros(62), 1.0])
    z = zcombo.gap_z(gap, 63)
    assert z.iloc[-1] == pytest.approx((1 - 1 / 63) / pd.Series(np.r_[np.zeros(62), 1.0]).std())
