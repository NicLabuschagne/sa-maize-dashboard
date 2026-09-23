"""SAGIS weekly deliveries and trade: parser helpers, pace maths, and a reconciliation of the
weekly season totals against the independently parsed monthly balance sheet."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from app.data import weekly as W
from config import DB_PATH
from ingest import sagis_weekly_parser as SP


# ----------------------------------------------------------------------------- parser helpers
@pytest.mark.parametrize("label, year, expected", [
    ("25/04 - 01/05/2026", 2026, "2026-05-01"),
    ("3 - 9 May 2008", 2008, "2008-05-09"),
    ("10 - 16 May", 2011, "2011-05-16"),
    ("25 Apr - 01 May/Mei 2026", 2026, "2026-05-01"),
    ("31 Dec/Des 2011 - 6 Jan 2012", 2011, "2012-01-06"),
    ("10 - 30 Dec/Des", 2011, "2011-12-30"),
])
def test_end_date_formats(label: str, year: int, expected: str) -> None:
    assert SP._end_date(label, year) == pd.Timestamp(expected)


def test_end_date_unparseable() -> None:
    assert SP._end_date("Total / Totaal", 2020) is None


def test_week_rows_holiday_markers() -> None:
    df = pd.DataFrame({0: ["Week", 1, "*2", "3 - 5", 6.0, "Total"],
                       1: [None, "a", "b", "c", "d", "e"]})
    assert SP._week_rows(df)["week"].tolist() == [1, 2, 5, 6]


def test_week_ends_step_from_anchor() -> None:
    rows = pd.DataFrame({0: [1, 2, 5], 1: ["no date", "07 - 13 May 2011", "x"], "week": [1, 2, 5]})
    ends = SP._week_ends(rows, 2011).tolist()
    assert ends == [pd.Timestamp("2011-05-06"), pd.Timestamp("2011-05-13"), pd.Timestamp("2011-06-03")]


def test_checked_catches_misread_column() -> None:
    SP._checked(np.array([1.0, 2, 3]), np.array([1.0, 3, 6]), "ok")
    with pytest.raises(ValueError):
        SP._checked(np.array([1.0, 2, 3]), np.array([1.0, 3, 60]), "bad")


def test_season_from_filename() -> None:
    assert SP.season_start(Path("IMP-EXP_Progressive_-_Mielies_(2019-20)52.8.xls")) == 2019
    assert SP.season_start(Path("ProdProgressive-Mielies2024-2552.8.xlsx")) == 2024


def test_class_and_flow_labels() -> None:
    assert SP._class_of("White Maize / Witmielies") == "white"
    assert SP._class_of("Yellow Maize / Geelmielies") == "yellow"
    assert SP._class_of("Exports / Uitvoere") is None          # 'uit' must not read as 'wit'
    assert SP._flow_of("Exports / Uitvoere") == "exports"


# ----------------------------------------------------------------------------- pace maths
def _toy() -> pd.DataFrame:
    rows = []
    for s, vals in (("2024/25", [10, 10, 10, 10]), ("2025/26", [20, 0, 20, 20]), ("2026/27", [30, 30])):
        for i, v in enumerate(vals, 1):
            for cls in ("white", "yellow"):
                rows.append({"season": s, "week": i, "week_end": pd.Timestamp(f"{s[:4]}-05-01") + pd.Timedelta(weeks=i),
                             "grain_class": cls, "flow": "exports", "tons_week": float(v), "tons_prog": np.nan,
                             "available_date": pd.Timestamp(f"{s[:4]}-05-13") + pd.Timedelta(weeks=i)})
    return pd.DataFrame(rows)


def test_total_is_white_plus_yellow() -> None:
    w = W.with_total(_toy())
    tot = w[(w.grain_class == "total") & (w.season == "2026/27")].tons_week.tolist()
    assert tot == [60.0, 60.0]


def test_season_to_date_stops_at_last_week() -> None:
    cum = W.season_to_date(_toy(), "exports", "white")
    assert cum["2026/27"].tolist()[:2] == [30.0, 60.0]
    assert cum["2026/27"].iloc[2:].isna().all()
    assert cum["2025/26"].tolist() == [20.0, 20.0, 40.0, 60.0]   # empty week is a flat step


def test_pace_same_week_comparison() -> None:
    p = W.pace(W.season_to_date(_toy(), "exports", "white"))
    assert (p["season"], p["week"], p["to_date"], p["last_season"]) == ("2026/27", 2, 60.0, 20.0)
    assert p["avg_5y"] == 20.0
    assert p["vs_last_pct"] == pytest.approx(200.0)


def test_as_of_hides_unpublished_weeks() -> None:
    w = _toy()
    cut = w[w.season == "2026/27"].available_date.min()
    seen = W.as_of(w, cut)
    assert seen[seen.season == "2026/27"].week.max() == 1


# ----------------------------------------------------------------------------- warehouse
@pytest.fixture(scope="module")
def warehouse() -> dict:
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if "sagis_weekly" not in con.execute("SHOW TABLES").df().name.tolist():
            pytest.skip("weekly table not built")
        w = con.execute("SELECT * FROM sagis_weekly").df()
        m = con.execute("""SELECT marketing_year AS season, attribute, grain_class, value_t FROM balance_sheet
            WHERE period_type = 'ytd' AND month(latest_month) = 4 AND grain_class IN ('white', 'yellow')
              AND attribute IN ('deliveries', 'exports_whole')
            QUALIFY row_number() OVER (PARTITION BY marketing_year, attribute, grain_class
                                       ORDER BY vintage_date) = 1""").df()
    finally:
        con.close()
    return {"w": w, "m": m}


def test_weeks_end_on_friday(warehouse: dict) -> None:
    assert (warehouse["w"].week_end.dt.dayofweek == 4).all()


def test_available_after_week_end(warehouse: dict) -> None:
    w = warehouse["w"]
    assert ((w.available_date - w.week_end).dt.days >= 5).all()


def test_weekly_totals_reconcile_to_monthly(warehouse: dict) -> None:
    """Two independent parsers of two different SAGIS products must agree on each season's total."""
    w, m = warehouse["w"], warehouse["m"]
    m["flow"] = m.attribute.map({"deliveries": "deliveries", "exports_whole": "exports"})
    t = w.groupby(["season", "flow", "grain_class"]).tons_week.sum().reset_index()
    j = t.merge(m, on=["season", "flow", "grain_class"])
    j = j[j.value_t > 50_000]                             # ratios on near-zero trade are meaningless
    assert len(j) >= 70
    assert (j.tons_week / j.value_t).between(0.97, 1.03).all()
