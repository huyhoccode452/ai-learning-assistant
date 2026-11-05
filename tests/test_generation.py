"""Tests for app/generation.py and the `generations` cache table: the quota-saving cache, the call counter and the
text helpers. Real SQLite in a temporary folder; the LLM is a script that fails the test on any unexpected call."""

import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest

from app.db import Database
from app.generation import CountingLLM, cache_key, content_hash, llm_identity, normalize_text, run_cached
from app.rag import RagError, RagErrorCode
from tests.fakes import DOC, ScriptedLLM, make_doc_chunk, make_doc_chunks, make_settings


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "app.sqlite3")
    database.init_db()
    database.claim_document(DOC, "tai_lieu.pdf", 3, 6)
    database.mark_ready(DOC, 6)
    return database


def run(db: Database, llm: ScriptedLLM, compute=None, **overrides):
    values = dict(kind="summary", doc_id=DOC, text_hash="t1", params={"style": "short"}, model="m|thinking=minimal",
                  prompt_version="p1", refresh=False)
    values.update(overrides)

    def default_compute(counting: CountingLLM):
        return {"summary": counting.generate("s", "p").text}, True

    return run_cached(db, llm, compute=compute or default_compute, **values)


# ---------------------------------------------------------------------- the cache saves quota


def test_a_repeated_request_is_answered_from_sqlite_with_zero_llm_calls(db: Database) -> None:
    llm = ScriptedLLM("first answer")
    first = run(db, llm)
    again = run(db, llm)  # the script is empty now: a second LLM call would fail the test
    assert (first.cached, first.llm_calls, first.payload) == (False, 1, {"summary": "first answer"})
    assert (again.cached, again.llm_calls, again.original_llm_calls, again.payload) == (True, 0, 1, {"summary": "first answer"})
    assert len(llm.calls) == 1 and again.llm_ms == 0.0


@pytest.mark.parametrize(
    "change",
    [{"params": {"style": "detailed"}}, {"model": "another|thinking=minimal"}, {"prompt_version": "p2"}, {"text_hash": "t2"},
     {"kind": "quiz"}],
    ids=["params", "model", "prompt-version", "indexed-text", "kind"],
)
def test_anything_that_can_change_the_answer_changes_the_key(db: Database, change: dict) -> None:
    llm = ScriptedLLM("old", "new")
    run(db, llm)
    result = run(db, llm, **change)
    assert (result.cached, result.payload) == (False, {"summary": "new"}) and len(llm.calls) == 2


def test_refresh_skips_the_lookup_but_stores_the_new_result(db: Database) -> None:
    llm = ScriptedLLM("old", "new")
    run(db, llm)
    refreshed = run(db, llm, refresh=True)
    after = run(db, ScriptedLLM())  # served from the cache again, now with the new text
    assert refreshed.payload == {"summary": "new"} and not refreshed.cached
    assert after.payload == {"summary": "new"} and after.cached and db.count_generations() == 1


def test_a_result_marked_not_cacheable_is_returned_but_never_stored(db: Database) -> None:
    llm = ScriptedLLM("a", "b")
    run(db, llm, compute=lambda c: ({"summary": c.generate("s", "p").text}, False))
    second = run(db, llm, compute=lambda c: ({"summary": c.generate("s", "p").text}, False))
    assert second.payload == {"summary": "b"} and db.count_generations() == 0


def test_a_failed_generation_stores_nothing_and_the_error_reaches_the_caller(db: Database) -> None:
    quota = RagError(RagErrorCode.RATE_LIMITED, "quota")
    with pytest.raises(RagError):
        run(db, ScriptedLLM(quota))
    assert db.count_generations() == 0
    assert run(db, ScriptedLLM("ok")).payload == {"summary": "ok"}  # the next request simply tries again


def test_an_unreadable_stored_result_is_regenerated(db: Database) -> None:
    llm = ScriptedLLM("fresh")
    key = cache_key("summary", DOC, "t1", {"style": "short"}, "m|thinking=minimal", "p1")
    db.put_generation(key, DOC, "summary", "{}", "m", "p1", "{broken json", 1)
    result = run(db, llm)
    assert result.payload == {"summary": "fresh"} and not result.cached


def test_the_result_is_stored_with_everything_that_identifies_it(db: Database) -> None:
    run(db, ScriptedLLM("x"))
    with sqlite3.connect(db.path) as conn:
        row = conn.execute("SELECT doc_id, kind, params_json, model, prompt_version, llm_calls FROM generations").fetchone()
    assert row == (DOC, "summary", '{"style": "short"}', "m|thinking=minimal", "p1", 1)


def test_a_document_deleted_during_generation_is_not_cached_and_does_not_crash(db: Database) -> None:
    def compute(counting: CountingLLM):
        payload = {"summary": counting.generate("s", "p").text}
        db.delete_document(DOC)  # the user deletes it while the model is working
        return payload, True

    result = run(db, ScriptedLLM("done"), compute=compute)
    assert result.payload == {"summary": "done"} and db.count_generations() == 0


def test_deleting_a_document_removes_its_stored_results(db: Database) -> None:
    run(db, ScriptedLLM("x"))
    run(db, ScriptedLLM("y"), params={"style": "detailed"})
    assert db.count_generations(DOC) == 2
    db.delete_document(DOC)
    assert db.count_generations() == 0  # ON DELETE CASCADE: no orphan rows, and a re-upload starts clean


def test_the_cache_table_survives_a_second_init_and_rejects_orphans(db: Database) -> None:
    db.init_db()
    with pytest.raises(sqlite3.IntegrityError):
        db.put_generation("k", "f" * 16, "summary", "{}", "m", "p", "{}", 1)  # no such document
    with pytest.raises(sqlite3.IntegrityError):
        db.put_generation("k", DOC, "essay", "{}", "m", "p", "{}", 1)  # kind is checked


# ---------------------------------------------------------------------- counting and helpers


def test_counting_llm_counts_calls_even_when_they_fail() -> None:
    counting = CountingLLM(ScriptedLLM("a", RagError(RagErrorCode.TIMEOUT, "slow")))
    counting.generate("s", "p")
    with pytest.raises(RagError):
        counting.generate("s", "p")
    assert counting.calls == 2 and counting.llm_seconds >= 0


def test_counting_llm_passes_the_schema_and_token_limit_through() -> None:
    inner = ScriptedLLM("{}")
    CountingLLM(inner).generate("s", "p", response_schema={"type": "object"}, max_output_tokens=123)
    assert inner.calls[0]["response_schema"] == {"type": "object"} and inner.calls[0]["max_output_tokens"] == 123


def test_normalize_text_handles_layout_and_unicode_composition() -> None:
    assert normalize_text("  a \n b\t\tc  ") == "a b c"
    decomposed = unicodedata.normalize("NFD", "mặc định")
    assert decomposed != "mặc định" and normalize_text(decomposed) == "mặc định"
    assert normalize_text("") == ""


def test_content_hash_follows_the_indexed_text() -> None:
    chunks = make_doc_chunks(3)
    assert content_hash(chunks) == content_hash(make_doc_chunks(3))
    assert content_hash(chunks) != content_hash(make_doc_chunks(4))  # another chunking
    changed = [*chunks[:2], make_doc_chunk(2, text="nội dung khác hẳn")]
    assert content_hash(chunks) != content_hash(changed)
    assert content_hash(chunks) != content_hash([make_doc_chunk(0, page=9), *chunks[1:]])  # page numbers count too


def test_llm_identity_includes_the_thinking_level() -> None:
    assert llm_identity(make_settings(gemini_llm_model="m", llm_thinking_level="minimal")) == "m|thinking=minimal"
    assert llm_identity(make_settings(gemini_llm_model="m")) == "m|thinking=default"


def test_cache_keys_are_stable_and_order_independent() -> None:
    a = cache_key("quiz", DOC, "t", {"n": 5, "topic": None}, "m", "p")
    assert a == cache_key("quiz", DOC, "t", {"topic": None, "n": 5}, "m", "p")
    assert len({a, cache_key("quiz", DOC, "t", {"n": 6, "topic": None}, "m", "p"), cache_key("quiz", "b" * 16, "t", {"n": 5, "topic": None}, "m", "p")}) == 3
    assert json.loads(json.dumps({"k": a}))["k"] == a and len(a) == 64
