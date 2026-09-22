"""The backtest engine and, more importantly, the controls that stop it manufacturing alpha."""
import numpy as np
import pandas as pd
import pytest

from app.data import analogues as AN
from app.data import backtest as BT


def _index(n: int = 300, drift: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    r = rng.normal(drift, 0.01, n)
    return pd.Series(np.exp(np.cumsum(r)), index=pd.DatetimeIndex(dates).as_unit("ns"))


def _entries(idx: pd.Series, zs: list[float], every: int = 21) -> pd.DataFrame:
    dates = [idx.index[i * every] for i in range(len(zs))]
    return pd.DataFrame({"vintage_date": dates, "z": zs,
                         "front_close": [4000.0] * len(zs)})


# ----------------------------------------------------------------------------- costs
def test_cost_model_round_trip_and_drag() -> None:
    c = BT.CostModel(half_spread_r_t=3.0, brokerage_r_t=0.30, slippage_r_t=2.0)
    assert c.one_way_r_t == pytest.approx(5.30)
    assert c.round_trip_r_t == pytest.approx(10.60)
    assert c.log_drag(4000) == pytest.approx(np.log1p(10.60 / 4000))
    assert c.log_drag(0) == 0.0


def test_costs_reduce_net_return() -> None:
    idx = _index()
    ent = _entries(idx, [1.0] * 5)
    free = BT.run_backtest(idx, ent, horizon=10, costs=BT.CostModel(0, 0, 0))
    dear = BT.run_backtest(idx, ent, horizon=10, costs=BT.CostModel(20, 1, 10))
    assert dear.stats["avg_net"] < free.stats["avg_net"]
    assert np.allclose(free.trades.gross_ret, dear.trades.gross_ret)


# ----------------------------------------------------------------------------- direction
def test_direction_fades_the_signal() -> None:
    idx = _index()
    r = BT.run_backtest(idx, _entries(idx, [2.0, -2.0]), horizon=5, costs=BT.CostModel(0, 0, 0))
    assert r.trades.iloc[0]["direction"] == "short"   # rich -> sell
    assert r.trades.iloc[1]["direction"] == "long"    # cheap -> buy


def test_a_rising_market_pays_a_long_and_costs_a_short() -> None:
    up = _index(drift=0.004, seed=3)
    zero = BT.CostModel(0, 0, 0)
    long_ = BT.run_backtest(up, _entries(up, [-1.0] * 6), horizon=10, costs=zero)
    short = BT.run_backtest(up, _entries(up, [1.0] * 6), horizon=10, costs=zero)
    assert long_.stats["avg_net"] > 0 > short.stats["avg_net"]
    assert long_.stats["avg_net"] == pytest.approx(-short.stats["avg_net"], rel=1e-6)


# ----------------------------------------------------------------------------- exits
def test_stop_closes_earlier_than_the_horizon() -> None:
    idx = _index(seed=7)
    ent = _entries(idx, [1.0] * 8)
    plain = BT.run_backtest(idx, ent, horizon=40, costs=BT.CostModel(0, 0, 0))
    stopped = BT.run_backtest(idx, ent, horizon=40, stop=0.01, costs=BT.CostModel(0, 0, 0))
    assert stopped.trades.held_days.mean() < plain.trades.held_days.mean()


def test_size_by_z_scales_the_position() -> None:
    idx = _index(drift=0.003, seed=5)
    ent = _entries(idx, [-2.0] * 5)
    flat = BT.run_backtest(idx, ent, horizon=10, costs=BT.CostModel(0, 0, 0))
    scaled = BT.run_backtest(idx, ent, horizon=10, size_by_z=True, costs=BT.CostModel(0, 0, 0))
    assert scaled.trades["size"].iloc[0] == pytest.approx(2.0)
    assert scaled.stats["avg_net"] == pytest.approx(2 * flat.stats["avg_net"], rel=1e-6)


def test_size_is_capped_at_three_sigma() -> None:
    idx = _index()
    r = BT.run_backtest(idx, _entries(idx, [9.0]), horizon=5, size_by_z=True)
    assert r.trades["size"].iloc[0] == 3.0


# ----------------------------------------------------------------------------- episodes
def test_consecutive_same_direction_entries_are_one_episode() -> None:
    idx = _index()
    r = BT.run_backtest(idx, _entries(idx, [1.5] * 5, every=21), horizon=5)
    assert r.stats["n_trades"] == 5
    assert r.stats["n_episodes"] == 1


def test_direction_flip_starts_a_new_episode() -> None:
    idx = _index()
    r = BT.run_backtest(idx, _entries(idx, [1.5, 1.5, -1.5, -1.5], every=21), horizon=5)
    assert r.stats["n_episodes"] == 2


# ----------------------------------------------------------------------------- overfitting
def test_expected_max_sharpe_grows_with_trials() -> None:
    a = BT.expected_max_sharpe(5, 1.0)
    b = BT.expected_max_sharpe(50, 1.0)
    assert 0 < a < b
    assert BT.expected_max_sharpe(1, 1.0) == 0.0
    assert BT.expected_max_sharpe(20, 0.0) == 0.0


def test_haircut_removes_the_selection_bound() -> None:
    hc = BT.sharpe_haircut(observed=2.0, n_trials=20, sharpe_std=1.0)
    assert hc["haircut"] == pytest.approx(2.0 - hc["expected_max"])
    assert hc["haircut"] < 2.0


def test_deflated_sharpe_falls_as_trials_rise() -> None:
    kw = dict(sr_daily=0.08, n_obs=400, skew=0.0, kurtosis=3.0, sharpe_std=0.05)
    few = BT.deflated_sharpe(n_trials=2, **kw)
    many = BT.deflated_sharpe(n_trials=200, **kw)
    assert 0 <= many < few <= 1


def test_deflated_sharpe_needs_a_sample() -> None:
    assert np.isnan(BT.deflated_sharpe(0.1, 5, 0.0, 3.0, 2, 0.05))


def test_pbo_detects_a_broken_selection_procedure() -> None:
    rng = np.random.default_rng(0)
    # In-sample ranking carries no information about out-of-sample ranking.
    is_s = rng.normal(size=(60, 8))
    oos_s = rng.normal(size=(60, 8))
    assert 0.3 < BT.pbo(is_s, oos_s) < 0.7          # ~coin flip
    # A procedure that generalises perfectly: OOS mirrors IS.
    assert BT.pbo(is_s, is_s) == 0.0


# ----------------------------------------------------------------------------- integration
def test_empty_entries_produce_empty_but_valid_output() -> None:
    idx = _index()
    r = BT.run_backtest(idx, pd.DataFrame({"vintage_date": [], "z": [], "front_close": []}),
                        horizon=10)
    assert r.trades.empty and r.stats["n_trades"] == 0
    assert len(r.daily) == len(idx) and r.daily.sum() == 0


def test_seasonal_surprise_is_point_in_time() -> None:
    months = pd.date_range("2010-05-01", periods=12 * 8, freq="MS")
    p = pd.DataFrame({"latest_month": months, "my_month": ((months.month - 5) % 12) + 1,
                      "months_cover": np.tile(np.arange(1.0, 13.0), 8)})
    s = AN.seasonal_surprise(p, min_prior=4)
    assert s.iloc[:12 * 4].isna().all()               # needs four prior seasons
    assert np.allclose(s.dropna(), 0.0, atol=1e-9)    # identical every season -> no surprise


def test_build_mask_multi_ands_conditions() -> None:
    p = pd.DataFrame({"z": [2.0, 2.0, -1.0], "months_cover_surprise": [-0.5, 0.5, -0.5],
                      "my_month": [1, 2, 3]})
    both = AN.build_mask_multi(p, [("z", "high", 1.0), ("months_cover_surprise", "low", -0.1)])
    assert both.tolist() == [True, False, False]
    one = AN.build_mask_multi(p, [("z", "high", 1.0)])
    assert one.sum() == 2
