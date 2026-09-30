import numpy as np
import pandas as pd
import pytest

from app.data import fairvalue as FV
from config import DB_PATH


def test_world_parity_converts_bushels_and_currency() -> None:
    snap = pd.DataFrame({
        "series": ["cbot_corn_safexclose", "usdzar_safexclose"] * 2,
        "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
        "value": [4.0, 15.0, 5.0, 16.0],
    })
    w = FV.world_parity(snap)
    assert len(w) == 2
    assert np.isclose(w.iloc[0], 4.0 * FV.BU_PER_TONNE * 15.0)
    assert np.isclose(w.iloc[1], 5.0 * FV.BU_PER_TONNE * 16.0)


def test_world_parity_empty_when_snapshot_missing() -> None:
    snap = pd.DataFrame({"series": ["za_cpi"], "date": pd.to_datetime(["2020-01-02"]), "value": [100.0]})
    assert FV.world_parity(snap).empty


def test_safex_mark_hour_is_utc_ten() -> None:
    """12:00 SAST with no DST is fixed at 10:00 UTC; CBOT settles later, so the settle is not knowable."""
    assert FV.SAFEX_MARK_UTC == 10


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
def test_snapshot_series_are_at_the_mark_hour() -> None:
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "macro_snap" not in tables:
            pytest.skip("macro_snap not fetched")
        n = con.execute("SELECT count(*) FROM macro_snap WHERE date <> date_trunc('day', date)").fetchone()[0]
        assert n == 0, "snapshot dates should be normalised to the day"
        series = {r[0] for r in con.execute("SELECT DISTINCT series FROM macro_snap").fetchall()}
        assert {"cbot_corn_safexclose", "usdzar_safexclose"} <= series
    finally:
        con.close()


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
def test_parity_panel_relative_forward_is_local_minus_world() -> None:
    import duckdb

    from app.data import features as F

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "macro_snap" not in tables:
            pytest.skip("macro_snap not fetched")
        bs = con.execute("SELECT * FROM balance_sheet").df()
        px = con.execute("SELECT * FROM prices").df()
        cpi = con.execute("SELECT * FROM macro").df()
        snap = con.execute("SELECT * FROM macro_snap").df()
    finally:
        con.close()
    p = FV.panel_parity(F.sd_monthly(bs), F.continuous(px), snap, cpi, "white", "WMAZ")
    assert len(p) > 100
    d = p.dropna(subset=["fwd_10d", "fwd_outright_10d", "fwd_world_10d"])
    assert np.allclose(d.fwd_10d, d.fwd_outright_10d - d.fwd_world_10d)
    assert np.allclose(np.exp(d.basis), d.close_cm / d.world_rand)   # level on the 90-day constant maturity
