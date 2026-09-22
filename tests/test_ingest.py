from pathlib import Path

import pytest

from config import DATA_DIR
from ingest.safex_loader import load_safex
from ingest.sagis_parser import _parse_month_label, _vintage_from_name, parse_sagis_file

SAGIS_DIR = DATA_DIR / "raw" / "sagis"
IDENTITY = ["opening_stock", "acquisition", "utilisation", "exports", "sundries", "closing_stock"]


@pytest.mark.parametrize("text,expected", [
    ("Dec/Des 2003", (2003, 12)), ("Jul 2026", (2026, 7)), ("May/Mei 2010", (2010, 5)),
    ("Mar/Mrt 2015", (2015, 3)), ("Progressive/Progressief", None),
])
def test_parse_month_label(text: str, expected: tuple[int, int] | None) -> None:
    assert _parse_month_label(text) == expected


def test_vintage_from_name() -> None:
    d, final = _vintage_from_name(Path("Mielies20260626F.xls"))
    assert (d.isoformat(), final) == ("2026-06-26", True)
    d, final = _vintage_from_name(Path("Mielies20200928a.xls"))
    assert (d.isoformat(), final) == ("2020-09-28", False)
    assert _vintage_from_name(Path("Mielies20252026_2026-06-26F.xls"))[0] is None


@pytest.mark.parametrize("name,mult", [("Mielies20260825.xls", 1.0), ("Mielies20040126.xls", 1000.0)])
def test_parse_sagis_file_identity(name: str, mult: float) -> None:
    path = SAGIS_DIR / name
    if not path.exists():
        pytest.skip("raw SAGIS file not downloaded")
    r = parse_sagis_file(path)
    assert r["ok"], r["error"]
    assert r["meta"]["unit_multiplier"] == mult
    t = r["rows"].query("grain_class == 'total'").pivot(
        index="attribute", columns="period_type", values="value_t").loc[IDENTITY]
    resid = t.loc["opening_stock"] + t.loc["acquisition"] - t.loc["utilisation"] \
        - t.loc["exports"] - t.loc["sundries"] - t.loc["closing_stock"]
    assert (resid.abs() < 1).all()


def test_load_safex_dedupes_and_nulls_zero_highs() -> None:
    if not list((DATA_DIR / "raw").glob("Physical Settled Grain contracts*.xlsx")):
        pytest.skip("raw SAFEX files not present")
    r = load_safex(DATA_DIR / "raw")
    assert r["ok"]
    p = r["prices"]
    assert not p.duplicated(["trade_date", "symbol", "expiry"]).any()
    assert set(p.symbol.unique()) == {"WMAZ", "YMAZ"}
    assert (p["high"].dropna() > 0).all()
