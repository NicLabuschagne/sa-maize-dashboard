import numpy as np
import pandas as pd

from app.data import analogues as AN


def test_episodes_clusters_consecutive_runs() -> None:
    m = pd.Series([False, True, True, False, True, False, False, True])
    ids = AN.episodes(m)
    assert ids.dropna().astype(int).tolist() == [1, 1, 2, 3]
    assert ids.isna().sum() == 4


def test_pct_rank_same_month_is_point_in_time() -> None:
    months = pd.date_range("2005-05-01", periods=12 * 8, freq="MS")
    p = pd.DataFrame({"latest_month": months, "my_month": ((months.month - 5) % 12) + 1})
    p["months_cover"] = np.tile(np.arange(1, 13), 8) + np.repeat(np.arange(8), 12) * 0.1  # rising each year
    r = AN.pct_rank_same_month(p, min_prior=5)
    assert r.iloc[:12 * 5].isna().all()              # needs 5 prior years
    assert np.allclose(r.iloc[12 * 5:].dropna(), 1.0)  # each later value exceeds all prior same-month values


def test_yoy_log_change_aligns_same_month() -> None:
    months = pd.date_range("2010-05-01", periods=24, freq="MS")
    p = pd.DataFrame({"latest_month": months, "months_cover": np.r_[np.full(12, 2.0), np.full(12, 4.0)]})
    y = AN.yoy_log_change(p)
    assert y.iloc[:12].isna().all()
    assert np.allclose(y.iloc[12:], np.log(2))


def test_forward_paths_and_fan() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    idx = pd.Series(np.exp(np.arange(30) * 0.01), index=pd.DatetimeIndex(dates).as_unit("ns"))
    paths = AN.forward_paths(idx, pd.Series([dates[0], dates[20]]), n_days=5, lag_days=1)
    assert paths.shape == (2, 6)
    assert np.allclose(paths.iloc[0].to_numpy(), np.arange(6) * 0.01)
    assert np.allclose(paths.iloc[1].to_numpy(), np.arange(6) * 0.01)
    f = AN.fan(paths)
    assert list(f.columns) == ["q10", "q25", "q50", "q75", "q90"] and len(f) == 6


def test_build_mask_direction_and_months() -> None:
    p = pd.DataFrame({"z": [2.0, -2.0, 0.5, np.nan], "my_month": [1, 2, 3, 4],
                      "months_cover_pct_same_month": np.nan, "months_cover_yoy": np.nan, "spread_z": np.nan})
    assert AN.build_mask(p, "z", 1.0, "high", None).tolist() == [True, False, False, False]
    assert AN.build_mask(p, "z", -1.0, "low", [2]).tolist() == [False, True, False, False]
    assert AN.build_mask(p, "z", -1.0, "low", [3]).sum() == 0
