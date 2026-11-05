"""Tests for evaluation/evaluate.py: question loading, the quota guard, caching + resuming after the
quota ran out, and the judge stage. Fakes only, no network."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.embeddings import EmbeddingError
from app.embeddings import ErrorCode as EmbeddingErrorCode
from app.rag import LLMResult, RagError, RagErrorCode, RagPipeline
from app.vectorstore import RetrievedChunk
from evaluation import evaluate
from evaluation.evaluate import (
    QUESTIONS_PATH,
    Question,
    QuotaGuardLLM,
    RetryCounter,
    assemble,
    build_index,
    choose_sample,
    load_questions,
    run_end_to_end,
    run_judge,
    run_retrieval,
)
from evaluation.judge import DiskCache
from tests.fakes import FakeEmbedder, FakeStore, make_settings

JUDGE_REPLY = json.dumps({"notes": "ok", "correctness": "correct", "is_refusal": False, "unsupported_claims": []})


def rate_limited() -> RagError:
    return RagError(RagErrorCode.RATE_LIMITED, "daily quota")


class ScriptedLLM:
    """Returns the next scripted item: a string, or an Exception to raise."""

    def __init__(self, *script: "str | Exception") -> None:
        self.script = list(script)
        self.calls = 0

    def generate(self, system_instruction: str, prompt: str) -> LLMResult:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResult(text=item)


def question(qid: str, *, expect_fallback: bool = False) -> Question:
    return Question(qid, "insufficient" if expect_fallback else "single", f"câu hỏi số {qid}", f"đáp án {qid}",
                    () if expect_fallback else (1,), expect_fallback)


# ---------------------------------------------------------------------- questions


def test_the_real_question_file_is_valid() -> None:
    document, questions = load_questions(QUESTIONS_PATH)
    assert document.endswith(".pdf") and len(questions) == 20
    counts = {t: sum(1 for q in questions if q.type == t) for t in ("single", "multi", "insufficient", "unrelated")}
    assert counts == {"single": 10, "multi": 5, "insufficient": 3, "unrelated": 2}
    assert all(bool(q.relevant_pages) != q.expect_fallback for q in questions)


def _write(tmp_path: Path, *questions: dict) -> Path:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"document": "d.pdf", "questions": list(questions)}), encoding="utf-8")
    return path


def _raw(qid: str = "q1", **changes: object) -> dict:
    return {"id": qid, "type": "single", "question": "Q", "reference_answer": "R", "relevant_pages": [1],
            "expect_fallback": False, **changes}


@pytest.mark.parametrize(
    "bad",
    [
        [_raw(type="essay")],
        [_raw(expect_fallback=True)],  # says "no answer" but lists relevant pages
        [_raw(relevant_pages=[])],  # says "has answer" but lists none
        [_raw("q1"), _raw("q1")],  # duplicate id
    ],
)
def test_load_questions_rejects_inconsistent_labels(tmp_path: Path, bad: list[dict]) -> None:
    with pytest.raises(ValueError):
        load_questions(_write(tmp_path, *bad))


# ---------------------------------------------------------------------- quota guard / retry counter


def test_guard_fails_fast_after_the_first_rate_limit() -> None:
    inner = ScriptedLLM(rate_limited())
    guard = QuotaGuardLLM(inner)
    for _ in range(3):
        with pytest.raises(RagError) as caught:
            guard.generate("s", "p")
        assert caught.value.code is RagErrorCode.RATE_LIMITED
    assert inner.calls == 1  # the 2nd and 3rd call never reached the model


def test_guard_stays_open_after_other_errors() -> None:
    inner = ScriptedLLM(RagError(RagErrorCode.SERVICE_UNAVAILABLE, "503"), "fine")
    guard = QuotaGuardLLM(inner)
    with pytest.raises(RagError):
        guard.generate("s", "p")
    assert guard.generate("s", "p").text == "fine" and guard.exhausted is None


def test_retry_counter_counts_only_retry_warnings_of_the_app() -> None:
    counter = RetryCounter()
    root = logging.getLogger()  # on the root logger the handler sees every logger, ours or not
    root.addHandler(counter)
    try:
        logging.getLogger("app.fake_client").warning("LLM call: HTTP 429, retry 1/5 in 2.0s")
        logging.getLogger("app.fake_client").warning("something else")
        logging.getLogger("app.fake_client").info("retry in an INFO message is ignored")
        logging.getLogger("other.module").warning("retry 1/5 from a library that is not ours")
    finally:
        root.removeHandler(counter)
    assert counter.count == 1


# ---------------------------------------------------------------------- end-to-end stage


def e2e_setup(tmp_path: Path, llm: ScriptedLLM, questions: list[Question], *, model: str = "m",
              thinking: str = "minimal", threshold: float = 0.5, top_k: int = 3, fingerprint: dict | None = None):
    settings = make_settings(gemini_llm_model=model, llm_thinking_level=thinking, top_k=top_k)
    store = FakeStore([RetrievedChunk(0.9, "d" * 16, "x.pdf", 1, 0, "text zero")])
    guard = QuotaGuardLLM(llm)
    pipeline = RagPipeline(settings, embedder=FakeEmbedder(), store=store, llm=guard, score_threshold=threshold)
    index = SimpleNamespace(fingerprint=fingerprint or {"f": 1})
    records: dict[str, dict] = {q.id: {} for q in questions}
    cache = DiskCache(tmp_path / "cache")
    run_end_to_end(questions, index, pipeline, guard, settings, cache, records)
    return records


def test_quota_runs_out_midway_then_the_next_run_resumes_from_the_cache(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2"), question("q3")]
    first_llm = ScriptedLLM("answer one [1]", rate_limited())
    first = e2e_setup(tmp_path, first_llm, questions)
    assert first["q1"]["e2e"]["answer"] == "answer one [1]" and first["q1"]["e2e"]["from_cache"] is False
    for qid in ("q2", "q3"):
        assert "e2e" not in first[qid] and first[qid]["e2e_error"]["code"] == "rate_limited"  # missing, not invented
    assert first_llm.calls == 2  # q3 was refused by the guard without calling the model

    second_llm = ScriptedLLM("answer two [1]", "answer three [1]")
    second = e2e_setup(tmp_path, second_llm, questions)
    assert second["q1"]["e2e"]["from_cache"] is True and second["q1"]["e2e"]["answer"] == "answer one [1]"
    assert [second[q]["e2e"]["answer"] for q in ("q2", "q3")] == ["answer two [1]", "answer three [1]"]
    assert second_llm.calls == 2  # q1 was NOT asked again

    third_llm = ScriptedLLM()  # nothing scripted: any call would crash
    third = e2e_setup(tmp_path, third_llm, questions)
    assert all(third[q]["e2e"]["from_cache"] for q in ("q1", "q2", "q3")) and third_llm.calls == 0


@pytest.mark.parametrize(
    "change",
    [{"model": "another-model"}, {"thinking": "high"}, {"threshold": 0.6}, {"top_k": 4}, {"fingerprint": {"f": 2}}],
    ids=["model", "thinking", "threshold", "top_k", "indexed-document"],
)
def test_changing_any_setting_does_not_reuse_cached_answers(tmp_path: Path, change: dict) -> None:
    questions = [question("q1")]
    e2e_setup(tmp_path, ScriptedLLM("old [1]"), questions)
    fresh = e2e_setup(tmp_path, ScriptedLLM("new [1]"), questions, **change)
    assert fresh["q1"]["e2e"]["answer"] == "new [1]" and fresh["q1"]["e2e"]["from_cache"] is False


def test_the_same_settings_reuse_the_cached_answer(tmp_path: Path) -> None:
    questions = [question("q1")]
    e2e_setup(tmp_path, ScriptedLLM("old [1]"), questions)
    again = e2e_setup(tmp_path, ScriptedLLM(), questions)  # nothing scripted: asking would crash
    assert again["q1"]["e2e"]["answer"] == "old [1]" and again["q1"]["e2e"]["from_cache"] is True


def test_the_record_keeps_the_measured_latency_and_sources(tmp_path: Path) -> None:
    record = e2e_setup(tmp_path, ScriptedLLM("fact [1]"), [question("q1")])["q1"]["e2e"]
    assert record["used_fallback"] is False and record["sources"] == [
        {"number": 1, "page": 1, "chunk_index": 0, "score": 0.9, "cited": True}]
    assert set(record["latency"]) == {"embed_ms", "search_ms", "llm_ms", "total_ms"} and record["retries"] == 0


# ---------------------------------------------------------------------- judge stage


def answered(answer: str = "answer [1]") -> dict:
    return {"answer": answer, "used_fallback": False, "sources": [{"number": 1, "page": 1, "chunk_index": 0, "score": 0.9, "cited": True}]}


FALLBACK = {"used_fallback": True, "answer": "fallback", "sources": []}
INDEX = SimpleNamespace(chunks_by_index={0: SimpleNamespace(text="chunk text")})


def test_the_judge_skips_fallbacks_and_missing_answers(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2", expect_fallback=True), question("q3")]
    records = {"q1": {"e2e": answered()}, "q2": {"e2e": FALLBACK}, "q3": {}}
    llm = ScriptedLLM(JUDGE_REPLY)
    run_judge(questions, INDEX, llm, "judge-model", DiskCache(tmp_path), records)
    assert llm.calls == 1  # exactly one batched call, only for q1
    assert records["q1"]["judge"]["correctness"] == "correct" and records["q1"]["judge"]["faithful"] is True
    assert "judge" not in records["q2"] and "judge" not in records["q3"]


def test_the_judge_stops_asking_once_the_quota_is_gone(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2"), question("q3")]
    records = {q.id: {"e2e": answered(f"answer {q.id} [1]")} for q in questions}
    inner = ScriptedLLM(rate_limited())
    run_judge(questions, INDEX, QuotaGuardLLM(inner), "judge-model", DiskCache(tmp_path), records)
    assert inner.calls == 1
    assert all("judge" not in r and r["judge_error"]["code"] == "rate_limited" for r in records.values())


def test_a_garbled_judge_reply_is_recorded_as_an_error(tmp_path: Path) -> None:
    records = {"q1": {"e2e": answered()}}
    run_judge([question("q1")], INDEX, ScriptedLLM("I cannot decide."), "m", DiskCache(tmp_path), records)
    assert "judge" not in records["q1"] and records["q1"]["judge_error"]["code"] == "JudgeError"


# ---------------------------------------------------------------------- assembling


def test_assemble_counts_chunks_per_page_and_classifies_outcomes() -> None:
    questions = [question("q1"), question("q2", expect_fallback=True)]
    index = SimpleNamespace(ingested=SimpleNamespace(chunks=[SimpleNamespace(page=1), SimpleNamespace(page=1), SimpleNamespace(page=2)]))
    records = {
        "q1": {"hits": [], "e2e": answered(), "judge": {"correctness": "partially_correct", "is_refusal": False}},
        "q2": {"hits": [], "e2e": FALLBACK},
    }
    results = assemble({"stages": []}, questions, records, index)
    assert results["chunks_per_page"] == {"1": 2, "2": 1}
    assert [q["outcome"] for q in results["questions"]] == ["partially_correct", "correct_fallback"]
    assert assemble({}, [question("q9")], {"q9": {}}, index)["questions"][0]["outcome"] is None  # nothing run yet


def test_the_random_sample_is_reproducible_judged_only_and_at_most_five() -> None:
    results = {"questions": [{"id": f"q{i:02d}", "judge": {"x": 1} if i % 3 else None} for i in range(1, 21)]}
    first = choose_sample(results, seed=7)
    assert first == choose_sample(results, seed=7) and len(first) == 5
    assert all(int(qid[1:]) % 3 for qid in first)  # never a question without a verdict
    assert first != choose_sample(results, seed=8)
    assert choose_sample({"questions": [{"id": "q1", "judge": {"x": 1}}]}, seed=1) == ["q1"]


# ---------------------------------------------------------------------- retrieval stage


class CountingEmbedder(FakeEmbedder):
    """Counts embeddings; `fail_on` = the 1-based query call that raises an embedding rate limit."""

    def __init__(self, fail_on: int | None = None) -> None:
        super().__init__()
        self.queries = 0
        self.documents = 0
        self.fail_on = fail_on

    def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        if self.queries == self.fail_on:
            raise EmbeddingError(EmbeddingErrorCode.RATE_LIMITED, "embedding quota")
        return super().embed_query(text)

    def embed_documents(self, texts):
        self.documents += 1
        return super().embed_documents(texts)


def retrieval_index() -> SimpleNamespace:
    chunks = [RetrievedChunk(0.9, "d" * 16, "x.pdf", 2, 0, "a"), RetrievedChunk(0.7, "d" * 16, "x.pdf", None, 1, "b")]
    return SimpleNamespace(store=FakeStore(chunks), fingerprint={"f": 1})


def test_retrieval_records_hits_without_any_threshold_and_caches_them(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2")]
    index, cache = retrieval_index(), DiskCache(tmp_path)
    first_embedder = CountingEmbedder()
    records = {q.id: {} for q in questions}
    run_retrieval(questions, index, first_embedder, cache, records)
    assert records["q1"]["hits"] == [{"score": 0.9, "page": 2, "chunk_index": 0}, {"score": 0.7, "page": None, "chunk_index": 1}]
    assert index.store.search_calls[0]["score_threshold"] is None and index.store.search_calls[0]["top_k"] == 5
    assert first_embedder.queries == 2

    second_embedder, second = CountingEmbedder(), {q.id: {} for q in questions}
    run_retrieval(questions, index, second_embedder, cache, second)
    assert second_embedder.queries == 0 and second == records  # fully served from the cache


def test_retrieval_stops_when_the_embedding_quota_is_gone(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2"), question("q3")]
    embedder = CountingEmbedder(fail_on=2)
    records = {q.id: {} for q in questions}
    run_retrieval(questions, retrieval_index(), embedder, DiskCache(tmp_path), records)
    assert "hits" in records["q1"] and "hits" not in records["q2"] and "hits" not in records["q3"]
    assert records["q2"]["retrieval_error"]["code"] == "rate_limited"
    assert embedder.queries == 2  # q3 was never tried; the stage stopped


def test_retrieval_cache_depends_on_the_indexed_document(tmp_path: Path) -> None:
    cache, one_question = DiskCache(tmp_path), [question("q1")]
    run_retrieval(one_question, retrieval_index(), CountingEmbedder(), cache, {"q1": {}})
    other_index = retrieval_index()
    other_index.fingerprint = {"f": 2}
    embedder = CountingEmbedder()
    run_retrieval(one_question, other_index, embedder, cache, {"q1": {}})
    assert embedder.queries == 1  # a different indexed document must not reuse the old hits


# ---------------------------------------------------------------------- indexing (document vectors cache)


class RecordingStore:
    def __init__(self, settings, collection: str) -> None:
        self.collection, self.indexed = collection, None

    def ensure_collection(self) -> bool:
        return True

    def index_document(self, doc_id, chunks, vectors) -> int:
        self.indexed = (doc_id, len(chunks), len(vectors))
        return len(chunks)


def _sample(tmp_path: Path, text: str, name: str = "sample.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_document_vectors_are_cached_per_document_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "QdrantStore", RecordingStore)
    settings, cache = make_settings(chunk_size=300, chunk_overlap=30), DiskCache(tmp_path / "cache")
    body = " ".join(f"Câu số {i} nói về PostgreSQL và chỉ mục." for i in range(30))
    sample = _sample(tmp_path, body)

    first = CountingEmbedder()
    index = build_index(settings, first, cache, "temp_collection", sample)
    assert first.documents == 1 and index.store.indexed[1] == index.store.indexed[2] == len(index.ingested.chunks)
    assert index.store.collection == "temp_collection"
    assert set(index.chunks_by_index) == {c.chunk_index for c in index.ingested.chunks}

    again = CountingEmbedder()
    build_index(settings, again, cache, "temp_collection", sample)
    assert again.documents == 0  # same text: vectors come from the cache

    changed = CountingEmbedder()
    build_index(settings, changed, cache, "temp_collection", _sample(tmp_path, body + " Thêm một câu mới hoàn toàn khác.", "changed.txt"))
    assert changed.documents == 1  # different text: embedded again, never stale vectors

    other_chunking = CountingEmbedder()
    build_index(make_settings(chunk_size=500, chunk_overlap=30), other_chunking, cache, "temp_collection", sample)
    assert other_chunking.documents == 1  # different chunking: different texts, different vectors

    cache.put("doc_vectors", index.fingerprint, [[0.0]])  # a damaged cache entry: one vector for many chunks
    repaired = CountingEmbedder()
    build_index(settings, repaired, cache, "temp_collection", sample)
    assert repaired.documents == 1  # not trusted, embedded again


def test_vectors_are_not_reused_when_the_ingestion_rules_change_but_the_file_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same file bytes -> same doc_id. If the cleaning/splitting code changes, the chunk texts change;
    the cached vectors belong to the OLD texts and must not be reused (see KI-08)."""
    from dataclasses import replace

    monkeypatch.setattr(evaluate, "QdrantStore", RecordingStore)
    settings, cache = make_settings(chunk_size=300, chunk_overlap=30), DiskCache(tmp_path / "cache")
    sample = _sample(tmp_path, " ".join(f"Câu số {i} nói về PostgreSQL và chỉ mục." for i in range(30)))
    build_index(settings, CountingEmbedder(), cache, "c", sample)

    original = evaluate.ingest_file

    def changed_rules(filename, data, settings_):
        result = original(filename, data, settings_)
        first = replace(result.chunks[0], text=result.chunks[0].text + " (rule change)")
        return replace(result, chunks=[first, *result.chunks[1:]])

    monkeypatch.setattr(evaluate, "ingest_file", changed_rules)
    embedder = CountingEmbedder()
    build_index(settings, embedder, cache, "c", sample)
    assert embedder.documents == 1
