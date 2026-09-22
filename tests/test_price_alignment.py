"""Guard against stale-price contamination in the release -> price joins.

A SAGIS release with no SAFEX close within PRICE_TOL must be dropped, not forward-filled with the
first price that ever existed. Without the tolerance every 2002-2009 vintage was assigned the
29 Apr 2009 close, which flattened the early residuals and blew up the expanding z-score.
"""
import numpy as np
import pandas as pd
import pytest

from app.data import fairvalue as FV


def _sd(dates: list[str], cover: list[float]) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    return pd.DataFrame({"vintage_date": d, "latest_month": d - pd.offsets.MonthBegin(1),
                         "marketing_year": "x", "grain_class": "white", "months_cover": cover,
                         "my_month": ((d.month - 5) % 12) + 1})


def _cont(start: str, n: int) -> pd.DataFrame:
    td = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"symbol": "WMAZ", "trade_date": td, "close_1": np.linspace(2000, 3000, n),
                         "spread_2_1_pct_ann": 5.0, "log_ret_1": 0.0, "expiry_1": "2020-03"})


def _cpi() -> pd.DataFrame:
    d = pd.date_range("2000-01-01", periods=400, freq="MS")
    return pd.DataFrame({"series": "za_cpi", "date": d, "value": np.linspace(30, 110, 400)})


def test_releases_before_price_history_are_dropped() -> None:
    sd = _sd(["2005-06-20", "2006-06-20", "2020-01-20", "2020-02-20"], [4.0, 5.0, 6.0, 7.0])
    p = FV.panel_price(sd, _cont("2020-01-01", 200), _cpi(), "white", "WMAZ")
    assert len(p) == 2
    assert p.vintage_date.min() >= pd.Timestamp("2020-01-01")


def test_release_just_before_a_close_is_kept() -> None:
    """A Friday release with the next close on Monday is inside the tolerance and must survive."""
    sd = _sd(["2020-01-17"], [4.0])                       # Friday
    p = FV.panel_price(sd, _cont("2020-01-20", 50), _cpi(), "white", "WMAZ")   # first close Monday
    assert len(p) == 1
    assert p.trade_date.iloc[0] == pd.Timestamp("2020-01-20")


def test_release_beyond_tolerance_is_dropped() -> None:
    sd = _sd(["2020-01-01"], [4.0])
    p = FV.panel_price(sd, _cont("2020-02-01", 50), _cpi(), "white", "WMAZ")   # gap > 7 days
    assert len(p) == 0


@pytest.mark.parametrize("panel_fn", ["panel_price", "panel_spread"])
def test_panels_use_the_tolerance(panel_fn: str) -> None:
    sd = _sd(["2005-06-20", "2020-01-20"], [4.0, 6.0])
    cont = _cont("2020-01-01", 200)
    p = (FV.panel_price(sd, cont, _cpi(), "white", "WMAZ") if panel_fn == "panel_price"
         else FV.panel_spread(sd, cont, "white", "WMAZ"))
    assert len(p) == 1


def test_expanding_z_stays_bounded_on_real_panel() -> None:
    """Regression: contaminated flat residuals drove the expanding std to ~0 and |z| to 1e14."""
    from config import DB_PATH

    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    import duckdb

    from app.data import features as F

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
    finally:
        con.close()
    q = FV.fit_expanding(FV.panel_price(F.sd_monthly(bs), F.continuous(px), cpi, "white", "WMAZ")).panel
    assert q.z.abs().max() < 6
