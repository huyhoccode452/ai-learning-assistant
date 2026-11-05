"""API tests for POST /documents/{doc_id}/summary and /quiz (TestClient, every dependency a fake; no network)."""

import inspect
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.generation import GenerationError, GenerationErrorCode
from app.main import api_error_from
from app.rag import RagError, RagErrorCode
from app.vectorstore import StoreErrorCode, VectorStoreError
from tests.fakes import ScriptedLLM
from tests.test_api import API_KEY, TEXT, build_env, error_of, processing_document, ready_document

QUOTE = "Giá trị mặc định là 4MB"


def one_question(system: str, prompt: str, kwargs: dict) -> str:
    chunk_id = int(re.findall(r"\[#(\d+)\]", prompt)[0])
    return json.dumps({"questions": [{
        "question": "Giá trị mặc định của work_mem là bao nhiêu?", "options": ["4MB", "8MB", "16MB", "64MB"], "answer_index": 0,
        "explanation": "Tài liệu nêu mặc định là 4MB.", "source_chunk": chunk_id, "quote": QUOTE}]}, ensure_ascii=False)


def verdict_all(system: str, prompt: str, kwargs: dict) -> str:
    count = prompt.count("Question ")
    return json.dumps({"results": [{"id": i, "key_supported": True, "another_option_correct": False, "reason": "ok"}
                                   for i in range(1, count + 1)]})


@pytest.fixture
def env(tmp_path: Path) -> Iterator[SimpleNamespace]:
    e = build_env(tmp_path)
    e.llm = ScriptedLLM()
    e.app.dependency_overrides[main.get_llm] = lambda: e.llm
    with TestClient(e.app) as client:
        e.client = client
        yield e


def summary(env, doc_id: str, **body):
    return env.client.post(f"/documents/{doc_id}/summary", json=body) if body else env.client.post(f"/documents/{doc_id}/summary")


def quiz(env, doc_id: str, **body):
    return env.client.post(f"/documents/{doc_id}/quiz", json=body) if body else env.client.post(f"/documents/{doc_id}/quiz")


# ====================================================================== summary


def test_summary_of_a_ready_document(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = ["Tóm tắt: work_mem mặc định 4MB (trang 1)."]
    response = summary(env, doc_id, style="detailed")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"] == "Tóm tắt: work_mem mặc định 4MB (trang 1)." and body["style"] == "detailed"
    assert (body["num_chunks_used"], body["truncated"], body["warning"]) == (body["num_chunks_total"], False, None)
    assert (body["llm_calls"], body["cached"]) == (1, False) and body["latency_ms"] >= 0 and "DETAILED" in env.llm.calls[0]["prompt"]
    assert env.db.count_generations(doc_id) == 1


def test_the_same_summary_request_again_costs_zero_llm_calls(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = ["Một lần duy nhất."]
    first = summary(env, doc_id).json()
    again = summary(env, doc_id).json()  # no body at all: defaults to the short style, and the script is empty now
    assert first["style"] == "short" and again["summary"] == first["summary"]
    assert (again["llm_calls"], again["cached"]) == (0, True) and len(env.llm.calls) == 1


def test_refresh_asks_the_model_again(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = ["bản một", "bản hai"]
    summary(env, doc_id)
    assert summary(env, doc_id, refresh=True).json()["summary"] == "bản hai"


def test_an_unknown_style_is_a_validation_error(env) -> None:
    doc_id = ready_document(env)
    response = summary(env, doc_id, style="epic")
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request" and env.llm.calls == []


@pytest.mark.parametrize("action", [summary, quiz])
def test_unknown_malformed_and_not_ready_documents_are_refused_before_any_llm_call(env, action) -> None:
    assert action(env, "f" * 16).status_code == 404 and error_of(action(env, "f" * 16))["code"] == "document_not_found"
    assert action(env, "not-a-hex-id").status_code == 422
    pending = processing_document(env)
    response = action(env, pending)
    assert response.status_code == 409 and error_of(response)["code"] == "document_not_ready"
    env.db.mark_failed(pending, "embedding.rate_limited", "x")
    assert action(env, pending).status_code == 409
    assert env.llm.calls == []


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (RagError(RagErrorCode.RATE_LIMITED, "SECRET quota text"), 429, "rag.rate_limited"),
        (RagError(RagErrorCode.TIMEOUT, "SECRET"), 504, "rag.timeout"),
        (RagError(RagErrorCode.SERVICE_UNAVAILABLE, "SECRET"), 503, "rag.service_unavailable"),
        (RagError(RagErrorCode.BLOCKED, "SECRET"), 422, "rag.blocked"),
        (RagError(RagErrorCode.AUTH_ERROR, f"bad key {API_KEY}"), 502, "rag.auth_error"),
    ],
)
@pytest.mark.parametrize("action", [summary, quiz])
def test_llm_failures_use_the_existing_error_mapping_and_leak_nothing(env, action, failure, status, code) -> None:
    doc_id = ready_document(env)
    env.llm.script = [failure]
    response = action(env, doc_id)
    assert response.status_code == status and error_of(response)["code"] == code
    assert "SECRET" not in response.text and API_KEY not in response.text and "Traceback" not in response.text
    if status == 429:
        assert response.headers["Retry-After"] == "60"
    assert env.db.count_generations() == 0  # a failed call stores nothing


@pytest.mark.parametrize("action", [summary, quiz])
def test_a_vector_store_failure_is_a_503(env, action) -> None:
    doc_id = ready_document(env)
    env.store.fail_read = VectorStoreError(StoreErrorCode.BACKEND_UNAVAILABLE, "connection refused at http://internal:6333")
    response = action(env, doc_id)
    assert response.status_code == 503 and error_of(response)["code"] == "store.backend_unavailable"
    assert "internal" not in response.text and env.llm.calls == []


@pytest.mark.parametrize("action", [summary, quiz])
def test_a_ready_document_whose_vectors_are_missing_is_a_clear_409(env, action) -> None:
    doc_id = ready_document(env)
    env.store.indexed[doc_id] = []  # the Qdrant volume was wiped (KI-25)
    response = action(env, doc_id)
    assert response.status_code == 409 and error_of(response)["code"] == "generation.no_content"
    assert "xóa tài liệu" in error_of(response)["message"] and env.llm.calls == []


# ====================================================================== quiz


def test_quiz_of_a_ready_document_with_verification(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question, verdict_all]
    response = quiz(env, doc_id, num_questions=1)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["requested"], body["returned"], body["llm_calls"], body["cached"]) == (1, 1, 2, False)
    assert (body["verify_requested"], body["verified"], body["message"], body["dropped"]) == (True, True, None, {})
    q = body["questions"][0]
    assert len(q["options"]) == 4 and q["options"][q["answer_index"]] == "4MB" and q["verified"] is True
    assert q["source"]["quote"] == QUOTE and q["source"]["page"] is None and q["source"]["filename"] == "notes.txt"
    assert body["verifier_decisions"] == [{"question": q["question"], "key_supported": True, "another_option_correct": False,
                                           "reason": "ok", "kept": True, "in_quiz": True}]


def test_the_same_quiz_request_again_costs_zero_llm_calls(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question, verdict_all]
    first = quiz(env, doc_id, num_questions=1).json()
    again = quiz(env, doc_id, num_questions=1).json()
    assert again["questions"] == first["questions"] and (again["llm_calls"], again["cached"]) == (0, True)
    assert len(env.llm.calls) == 2


def test_verification_can_be_switched_off_for_one_call(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question]
    body = quiz(env, doc_id, num_questions=1, verify=False).json()
    assert body["llm_calls"] == 1 and body["verified"] is False and body["questions"][0]["verified"] is None


def test_refresh_makes_a_new_quiz_and_a_topic_reaches_the_search(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question, one_question]
    first = quiz(env, doc_id, num_questions=1, verify=False).json()
    assert (first["cached"], first["llm_calls"]) == (False, 1)
    assert quiz(env, doc_id, num_questions=1, verify=False).json()["cached"] is True  # same request: stored
    renewed = quiz(env, doc_id, num_questions=1, verify=False, refresh=True).json()
    assert (renewed["cached"], renewed["llm_calls"]) == (False, 1) and env.llm.script == []  # the second script entry was used

    searches: list[tuple] = []
    original = env.store.search
    env.store.search = lambda *args, **kwargs: (searches.append((args, kwargs)), original(*args, **kwargs))[1]  # type: ignore[method-assign]
    env.llm.script = [one_question]
    quiz(env, doc_id, num_questions=1, verify=False, topic="work_mem")
    assert len(searches) == 1 and searches[0][1]["doc_ids"] == [doc_id]
    assert "Focus on this topic" in env.llm.calls[-1]["prompt"] and "work_mem" in env.llm.calls[-1]["prompt"]


def test_default_request_body_is_five_questions_with_verification(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question, verdict_all]
    body = quiz(env, doc_id).json()
    assert body["requested"] == 5 and body["verify_requested"] is True and body["returned"] == 1
    assert "Chỉ tạo được 1/5" in body["message"] and "không bịa" in body["message"]  # the model wrote fewer; nothing is padded


@pytest.mark.parametrize(
    "body",
    [{"num_questions": 0}, {"num_questions": 11}, {"num_questions": "five"}, {"page_from": 0}, {"page_from": 5, "page_to": 2},
     {"topic": "x" * 201}, {"verify": "maybe"}],
)
def test_invalid_quiz_requests_are_422_and_never_reach_the_model(env, body: dict) -> None:
    doc_id = ready_document(env)
    response = quiz(env, doc_id, **body)
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request" and env.llm.calls == []


def test_a_malformed_model_reply_is_a_502_without_the_raw_text(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = ["SECRET-RAW-MODEL-OUTPUT {not json"]
    response = quiz(env, doc_id)
    assert response.status_code == 502 and error_of(response)["code"] == "generation.malformed_output"
    assert "SECRET" not in response.text and "định dạng" in error_of(response)["message"]
    assert env.db.count_generations() == 0


def test_a_page_range_with_no_content_is_a_422(env) -> None:
    doc_id = ready_document(env)
    response = quiz(env, doc_id, page_from=40)
    assert response.status_code == 422 and error_of(response)["code"] == "generation.empty_selection" and env.llm.calls == []


def test_a_verifier_that_rejects_everything_returns_no_questions_with_a_message(env) -> None:
    doc_id = ready_document(env)
    reject = lambda s, p, k: json.dumps({"results": [{"id": 1, "key_supported": False, "another_option_correct": False, "reason": "no"}]})  # noqa: E731
    env.llm.script = [one_question, reject]
    body = quiz(env, doc_id, num_questions=1).json()
    assert body["questions"] == [] and body["returned"] == 0 and "Không tạo được câu hỏi hợp lệ" in body["message"]
    assert body["dropped"] == {"verifier_key_not_supported": 1} and env.db.count_generations() == 0


def test_a_quota_error_during_verification_still_returns_the_quiz_flagged_unverified(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = [one_question, RagError(RagErrorCode.RATE_LIMITED, "quota")]
    body = quiz(env, doc_id, num_questions=1).json()
    assert body["returned"] == 1 and body["verified"] is False and body["questions"][0]["verified"] is None
    assert "chưa được xác minh" in body["warnings"][0] and env.db.count_generations() == 0


# ====================================================================== plumbing


def test_deleting_a_document_removes_its_stored_summary_and_quiz(env) -> None:
    doc_id = ready_document(env)
    env.llm.script = ["tóm tắt", one_question]
    summary(env, doc_id)
    quiz(env, doc_id, num_questions=1, verify=False)
    assert env.db.count_generations(doc_id) == 2
    assert env.client.delete(f"/documents/{doc_id}").status_code == 200
    assert env.db.count_generations() == 0


def test_the_new_endpoints_are_plain_def_so_they_do_not_block_the_event_loop(env) -> None:
    routes = {r.path: r for r in env.app.routes if getattr(r, "path", "").endswith(("/summary", "/quiz"))}
    assert set(routes) == {"/documents/{doc_id}/summary", "/documents/{doc_id}/quiz"}
    assert all(not inspect.iscoroutinefunction(r.endpoint) for r in routes.values())


def test_the_endpoints_are_documented_in_openapi(env) -> None:
    paths = env.client.get("/openapi.json").json()["paths"]
    for path in ("/documents/{doc_id}/summary", "/documents/{doc_id}/quiz"):
        assert "post" in paths[path] and {"404", "409", "422", "429", "502", "503", "504"} <= set(paths[path]["post"]["responses"])


@pytest.mark.parametrize(
    ("code", "status"),
    [(GenerationErrorCode.NO_CONTENT, 409), (GenerationErrorCode.EMPTY_SELECTION, 422), (GenerationErrorCode.MALFORMED_OUTPUT, 502)],
)
def test_every_generation_error_code_is_mapped_with_a_vietnamese_message(code, status) -> None:
    error = api_error_from(GenerationError(code, "English internal detail"))
    assert (error.status, error.code) == (status, f"generation.{code.value}")
    assert "English" not in error.message and re.search(r"[À-ỹ]", error.message)


def test_the_text_fixture_is_what_the_quote_refers_to() -> None:
    assert QUOTE in TEXT.decode()  # guards the tests above against a silent change of the sample text
