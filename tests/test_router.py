"""The rules-based backend must be deterministic and must never claim data it did not query."""
import pytest

from app.data import chat as C
from app.data import router as R
from config import DB_PATH


@pytest.mark.parametrize("q,intent", [
    ("is white rich right now?", "current_state"),
    ("where are we currently", "current_state"),
    ("when was white last more than 1.5 sigma rich?", "extreme_history"),
    ("show me episodes where yellow was cheap", "extreme_history"),
    ("do rich readings underperform at 10 days?", "bucket_performance"),
    ("what happens after a rich signal", "bucket_performance"),
    ("show me months of cover", "cover_history"),
    ("largest revisions to closing stock", "revisions"),
    ("widest parity basis for yellow", "parity"),
    ("show the forward curve", "curve"),
    ("what data do you have?", "coverage"),
    ("white premium over yellow", "white_yellow"),
])
def test_intents_classify(q: str, intent: str) -> None:
    hit = R.route(q)
    assert hit is not None and hit.intent == intent


@pytest.mark.parametrize("q", ["what is the weather in Cape Town", "tell me a joke", "hello", ""])
def test_unmatched_questions_return_none(q: str) -> None:
    assert R.route(q) is None


def test_entity_extraction_class_model_threshold_horizon() -> None:
    assert R._cls("show me yellow cover") == "yellow"
    assert R._cls("nothing here") == "white"            # default
    assert R._model("calendar spread history") == "B"
    assert R._model("parity basis") == "D"
    assert R._horizon("underperform at 3 months") == "fwd_3m"
    assert R._num("more than 2.5 sigma rich", 1.5) == 2.5
    assert R._num("above 1.2", 1.5) == 1.2
    assert R._topn("top 25 releases") == 25
    assert R._topn("no number here") == 10


def test_topn_is_clamped() -> None:
    assert R._topn("top 9999 rows") == 100
    assert R._topn("top 0 rows") == 1


@pytest.mark.parametrize("q,intent", [
    ("show me exports", "balance_sheet"),
    ("show me deliveries", "balance_sheet"),
    ("biggest revision", "revisions"),
    ("when was white dislocated", "extreme_history"),
    ("yellow calendar inversion", "spread"),
    ("show me prices", "price_history"),
])
def test_plural_and_stem_forms_match(q: str, intent: str) -> None:
    """Regression: a trailing \b after a word stem can never match, e.g. \brevis\b vs 'revisions'."""
    hit = R.route(q)
    assert hit is not None and hit.intent == intent


def test_cheap_question_flips_the_comparison() -> None:
    rich = R.route("when was white last 1.5 sigma rich")
    cheap = R.route("when was white last 1.5 sigma cheap")
    assert "z > 1.5" in rich.sql and "z < -1.5" in cheap.sql


def test_every_generated_query_passes_the_read_only_guard() -> None:
    for q in [e for e in R.EXAMPLES] + ["yellow calendar spread", "white premium over yellow",
                                        "exports in 2016", "show me deliveries"]:
        hit = R.route(q)
        if hit:
            ok, _ = C.guard_sql(hit.sql)
            assert ok, f"router produced a query the guard rejects: {q}"


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
@pytest.mark.parametrize("q", R.EXAMPLES)
def test_examples_execute_and_return_rows(q: str) -> None:
    a = C.local_answer([{"role": "user", "content": q}])
    assert a.local and a.queries, q
    assert a.queries[0]["ok"], a.queries[0]["error"]
    assert a.queries[0]["n"] > 0, f"no rows for: {q}"
    assert a.data_backed


def test_unmatched_question_is_not_data_backed() -> None:
    a = C.local_answer([{"role": "user", "content": "what is the weather in Cape Town"}])
    assert a.local and not a.queries and not a.data_backed
    assert "rules-based backend" in a.text


def test_local_answer_never_calls_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against an accidental model call from the free backend."""
    import anthropic

    def boom(*_a, **_k):
        raise AssertionError("local_answer must not construct an API client")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    a = C.local_answer([{"role": "user", "content": "is white rich right now?"}])
    assert a.local
