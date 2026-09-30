"""Point-in-time and arithmetic checks for app/data/band_fairvalue.py."""
import numpy as np
import pandas as pd
import pytest

from app.data import band_fairvalue as B


def _sd() -> pd.DataFrame:
    return pd.DataFrame({"grain_class": "yellow", "vintage_date": pd.to_datetime(["2020-02-25", "2020-03-25"]),
                         "latest_month": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                         "closing_stock": [1000.0, 900.0], "utilisation_12m": [1200.0, 1200.0],
                         "disappearance_12m": [1500.0, 1500.0]})


def _weekly() -> pd.DataFrame:
    rows = []
    for flow, tons in (("deliveries", 10.0), ("imports", 1.0), ("exports", 3.0)):
        rows.append({"grain_class": "yellow", "flow": flow, "week_end": pd.Timestamp("2020-02-07"),
                     "tons_week": tons, "available_date": pd.Timestamp("2020-02-19")})
    return pd.DataFrame(rows)


def test_nowcast_adds_weeks_after_the_reported_month_once_published() -> None:
    dates = pd.Series(pd.to_datetime(["2020-02-20", "2020-02-26", "2020-03-02", "2020-03-26"]))
    stu = B.stocks_to_use_daily(_sd(), _weekly(), "yellow", dates)
    assert np.isnan(stu["stu_domestic"].iloc[0])                     # no release yet: nothing backfilled
    expected = (1000 + 10 + 1 - 3 - 1200 / 12 * 7 / 30.44) / 1200
    assert stu["stu_domestic_nowcast"].iloc[1] == pytest.approx(expected)
    assert stu["nowcast_weeks"].iloc[1] == 1
    assert stu["stu_domestic"].iloc[3] == pytest.approx(900 / 1200)  # next release replaces it
    assert stu["stu_total"].iloc[3] == pytest.approx(900 / 1500)


def _frame(n_weeks: int = 520, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n_weeks * 5)
    stu = np.exp(rng.normal(np.log(0.5), 0.3, len(dates)))
    return pd.DataFrame({"date": dates, "position": 0.9 - 0.8 * stu + rng.normal(0, 0.02, len(dates)),
                         "stu_domestic_nowcast": stu, "export_edge": 2000.0, "import_edge": 3500.0,
                         "safex": 2500.0})


def test_fair_value_uses_only_earlier_weeks_and_recovers_slope() -> None:
    frame = _frame()
    base = B.fair_value_daily(frame)
    assert base["stu_slope"].dropna().iloc[-1] == pytest.approx(-0.8, abs=0.05)
    cut = pd.Timestamp("2018-06-01")                                  # a Friday
    shocked = frame.copy()
    shocked.loc[shocked.date > cut, "position"] += 5
    after = B.fair_value_daily(shocked)
    upto = base.date <= cut + pd.Timedelta(days=7)
    assert np.allclose(base.loc[upto, "fair_position"], after.loc[upto, "fair_position"], equal_nan=True)
    first = B.first_fit_date(frame.date.min())
    assert base.loc[base.date < first - pd.Timedelta(days=7), "fair_position"].isna().all()


def test_fair_value_in_rand_is_edge_plus_position_times_width() -> None:
    out = B.fair_value_daily(_frame()).dropna(subset=["fair_value"]).iloc[-1]
    assert out.fair_value == pytest.approx(2000 + out.fair_position * 1500)
    assert out.gap_rand == pytest.approx(2500 - out.fair_value)


def test_first_fit_date_counts_complete_marketing_years() -> None:
    assert B.first_fit_date(pd.Timestamp("2011-06-01")) == pd.Timestamp("2017-05-01")
    assert B.first_fit_date(pd.Timestamp("2011-05-01")) == pd.Timestamp("2016-05-01")
