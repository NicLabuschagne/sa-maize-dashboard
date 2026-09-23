"""SAGIS historic import/export parity: cleaning helpers and the warehouse table."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from config import DB_PATH
from ingest.sagis_parity_parser import despike


def test_despike_blanks_typo_keeps_trend() -> None:
    s = pd.Series([100.0, 101, 102, 1030, 104, 105, 106, 107, 108])
    out = despike(s)
    assert np.isnan(out.iloc[3])
    assert out.drop(3).notna().all()


def test_despike_absolute_mode_for_rates() -> None:
    s = pd.Series([10.5, 10.5, 10.5, 0.75, 10.5, 10.25, 10.25])
    out = despike(s, tol=1.0, log=False)
    assert np.isnan(out.iloc[3]) and out.drop(3).notna().all()


@pytest.fixture(scope="module")
def parity() -> pd.DataFrame:
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if "sagis_parity" not in con.execute("SHOW TABLES").df().name.tolist():
            pytest.skip("parity table not built")
        return con.execute("SELECT * FROM sagis_parity").df()
    finally:
        con.close()


def test_band_is_ordered(parity: pd.DataFrame) -> None:
    b = parity.dropna(subset=["export_randfontein", "import_randfontein"])
    assert len(b) > 900
    assert (b.import_randfontein > b.export_randfontein).all()


def test_prime_rate_in_historical_range(parity: pd.DataFrame) -> None:
    """SA prime has been between 7% and 17% since 2001; the backed-out series should agree."""
    p = parity.prime_rate.dropna()
    assert p.between(6.5, 17.5).all()
    jan09 = parity[(parity.date >= "2009-01-01") & (parity.date < "2009-02-01")].prime_rate.median()
    assert jan09 == pytest.approx(15.0, abs=0.3)


def test_available_after_calculation(parity: pd.DataFrame) -> None:
    assert ((parity.available_date - parity.date).dt.days >= 7).all()
