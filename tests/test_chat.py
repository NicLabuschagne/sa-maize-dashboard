"""The SQL guard and the provenance logic are safety-critical: they decide what runs against the
warehouse and whether an answer is labelled as coming from the user's data."""
import re

import pandas as pd
import pytest

from app.data import chat as C
from config import DB_PATH


@pytest.mark.parametrize("sql", [
    "DROP TABLE prices",
    "DELETE FROM prices",
    "INSERT INTO prices VALUES (1)",
    "UPDATE prices SET close = 0",
    "CREATE TABLE x AS SELECT 1",
    "ATTACH 'other.db' AS o",
    "COPY prices TO 'out.csv'",
    "PRAGMA database_list",
    "INSTALL httpfs",
    "SELECT 1; DROP TABLE prices",
    "",
    "   ",
])
def test_guard_rejects_non_select(sql: str) -> None:
    ok, _ = C.guard_sql(sql)
    assert not ok


@pytest.mark.parametrize("sql", [
    "SELECT * FROM prices",
    "  select close from prices where symbol='WMAZ'",
    "WITH t AS (SELECT 1 AS a) SELECT a FROM t",
    "SELECT 1;",                      # single trailing semicolon is fine
])
def test_guard_accepts_reads(sql: str) -> None:
    ok, out = C.guard_sql(sql)
    assert ok and re.match(r"^\s*(select|with)", out, re.I)


def test_guard_adds_limit_when_missing_and_keeps_existing() -> None:
    _, out = C.guard_sql("SELECT * FROM prices")
    assert f"LIMIT {C.DEFAULT_LIMIT}" in out
    _, out2 = C.guard_sql("SELECT * FROM prices LIMIT 3")
    assert out2.lower().count("limit") == 1


def test_guard_ignores_keywords_inside_comments() -> None:
    ok, out = C.guard_sql("SELECT 1 -- drop table prices\n")
    assert ok and "drop" not in out.lower()


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
def test_run_sql_executes_and_is_read_only() -> None:
    r = C.run_sql("SELECT count(*) AS n FROM prices")
    assert r["ok"] and r["rows"].iloc[0]["n"] > 0
    bad = C.run_sql("DROP TABLE prices")
    assert not bad["ok"]
    still = C.run_sql("SELECT count(*) AS n FROM prices")
    assert still["ok"], "table must still exist after a rejected write"


def test_run_sql_reports_errors_without_raising() -> None:
    r = C.run_sql("SELECT * FROM no_such_table")
    assert not r["ok"] and r["error"]


def test_answer_data_backed_is_code_enforced() -> None:
    """A model claiming data provenance cannot make data_backed true."""
    lying = C.Answer(text="According to your database, cover is 7.0 months.")
    assert not lying.data_backed
    failed = C.Answer(text="x", queries=[{"ok": False, "sql": "SELECT 1", "error": "boom"}])
    assert not failed.data_backed
    real = C.Answer(text="x", queries=[{"ok": True, "sql": "SELECT * FROM signals", "rows": pd.DataFrame()}])
    assert real.data_backed


def test_tables_used_reports_only_successful_queries() -> None:
    a = C.Answer(text="x", queries=[
        {"ok": True, "sql": "SELECT * FROM signals JOIN prices USING (x)", "rows": pd.DataFrame()},
        {"ok": False, "sql": "SELECT * FROM balance_sheet", "error": "boom"},
    ])
    assert a.tables_used == ["prices", "signals"]


def test_system_prompt_switches_on_data_only() -> None:
    strict = C._system_prompt("schema", data_only=True)
    loose = C._system_prompt("schema", data_only=False)
    assert "DATA-ONLY MODE IS ON" in strict and "[GK]" not in strict
    assert "DATA-ONLY MODE IS OFF" in loose and "[GK]" in loose


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
def test_schema_text_lists_core_tables() -> None:
    s = C.schema_text()
    for t in ("prices", "balance_sheet", "signals"):
        assert t in s



@pytest.mark.parametrize("key,ok", [
    ("sk-ant-api03-abc", True),
    ("apikey_01FfSrWMqBJQk76ua1pFxkYL", False),   # Console key ID, not the secret
    ("", False),
    (None, False),
    ("random-string", False),
])
def test_key_looks_valid_catches_the_key_id_paste_error(key, ok: bool) -> None:
    assert C.key_looks_valid(key)[0] is ok


def test_key_id_reason_is_explanatory() -> None:
    _, why = C.key_looks_valid("apikey_0123")
    assert "key ID" in why and "sk-ant-api03-" in why


def test_cost_model_prices_cache_reads_cheaply() -> None:
    """A cached repeat must cost far less than the first call that wrote the cache."""
    first = C.Answer(text="x", usage=[{"in": 300, "out": 600, "cache_write": 3000, "cache_read": 0}])
    repeat = C.Answer(text="x", usage=[{"in": 300, "out": 600, "cache_write": 0, "cache_read": 3000}])
    assert repeat.cost_usd < first.cost_usd
    assert C.Answer(text="x").cost_usd == 0.0
