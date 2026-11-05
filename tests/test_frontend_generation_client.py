"""frontend/api_client.py: summary() and quiz() with the fake HTTP layer (no network)."""

import pytest
import requests

from tests.test_frontend_api_client import FakeResponse, client, error_body, failure

SUMMARY = {"summary": "Tóm tắt (trang 1).", "style": "short", "num_chunks_used": 29, "num_chunks_total": 29, "truncated": False,
           "warning": None, "llm_calls": 1, "cached": False, "latency_ms": 1620.4, "llm_ms": 1600.0}
QUESTION = {"id": 1, "question": "Giá trị mặc định?", "options": ["4MB", "8MB", "16MB", "64MB"], "answer_index": 2,
            "explanation": "Vì tài liệu nói vậy.", "source": {"chunk_index": 12, "page": 5, "filename": "a.pdf", "quote": "mặc định 4MB"}, "verified": True}
QUIZ = {"questions": [QUESTION], "requested": 3, "returned": 1, "dropped": {"quote_not_in_source": 2}, "message": "Chỉ tạo được 1/3.",
        "warnings": ["cảnh báo"], "verify_requested": True, "verified": True, "llm_calls": 2, "cached": False, "latency_ms": 6800.0,
        "llm_ms": 6700.0, "verifier_decisions": [{"question": "Giá trị mặc định?", "key_supported": True, "another_option_correct": False,
                                                   "reason": "ok", "kept": True, "in_quiz": True}]}
DOC = "a" * 16


def test_summary_sends_the_style_and_refresh_with_the_long_timeout() -> None:
    api, session = client(FakeResponse(200, SUMMARY))
    result = api.summary(DOC, "detailed", refresh=True)
    call = session.calls[0]
    assert (call["method"], call["url"], call["json"]) == ("POST", f"http://api.test:8000/documents/{DOC}/summary", {"style": "detailed", "refresh": True})
    assert call["timeout"] == (5.0, 120.0)
    assert (result.summary, result.num_chunks_used, result.truncated, result.llm_calls, result.cached) == ("Tóm tắt (trang 1).", 29, False, 1, False)
    assert result.latency_ms == 1620.4 and result.warning is None


def test_the_summary_defaults_to_short_without_refresh() -> None:
    api, session = client(FakeResponse(200, SUMMARY))
    api.summary(DOC)
    assert session.calls[0]["json"] == {"style": "short", "refresh": False}


def test_quiz_sends_only_the_options_that_were_set() -> None:
    api, session = client(FakeResponse(200, QUIZ), FakeResponse(200, QUIZ))
    api.quiz(DOC)
    api.quiz(DOC, num_questions=7, topic="replication", page_from=2, page_to=4, verify=False, refresh=True)
    assert session.calls[0]["json"] == {"num_questions": 5, "verify": True, "refresh": False}  # no topic / page keys at all
    assert session.calls[1]["json"] == {"num_questions": 7, "verify": False, "refresh": True, "topic": "replication", "page_from": 2, "page_to": 4}
    assert session.calls[0]["url"].endswith(f"/documents/{DOC}/quiz") and session.calls[0]["timeout"] == (5.0, 120.0)


def test_the_quiz_reply_is_parsed_completely() -> None:
    quiz = client(FakeResponse(200, QUIZ))[0].quiz(DOC)
    q = quiz.questions[0]
    assert (q.id, q.options, q.answer_index, q.verified) == (1, ("4MB", "8MB", "16MB", "64MB"), 2, True)
    assert (q.source.page, q.source.quote, q.source.filename, q.explanation) == (5, "mặc định 4MB", "a.pdf", "Vì tài liệu nói vậy.")
    assert (quiz.requested, quiz.dropped, quiz.message, quiz.warnings) == (3, {"quote_not_in_source": 2}, "Chỉ tạo được 1/3.", ("cảnh báo",))
    assert (quiz.verified, quiz.llm_calls, quiz.cached, quiz.latency_ms) == (True, 2, False, 6800.0)
    assert quiz.decisions[0].kept and quiz.decisions[0].in_quiz and quiz.decisions[0].reason == "ok"


def test_a_question_without_a_page_and_an_unverified_one_are_accepted() -> None:
    odd = {**QUESTION, "source": {**QUESTION["source"], "page": None}, "verified": None}
    reply = {**QUIZ, "questions": [odd], "message": None, "warnings": [], "verifier_decisions": []}
    quiz = client(FakeResponse(200, reply))[0].quiz(DOC)
    assert quiz.questions[0].source.page is None and quiz.questions[0].verified is None and quiz.decisions == () and quiz.message is None


@pytest.mark.parametrize(
    "broken",
    [
        {**QUESTION, "options": ["a", "b", "c"]},
        {**QUESTION, "options": ["a", "b", "c", "d", "e"]},
        {**QUESTION, "answer_index": 4},
        {**QUESTION, "answer_index": -1},
        {k: v for k, v in QUESTION.items() if k != "source"},
        {k: v for k, v in QUESTION.items() if k != "explanation"},
    ],
)
def test_a_quiz_question_that_breaks_the_contract_is_a_bad_response(broken: dict) -> None:
    error = failure(lambda api: api.quiz(DOC), FakeResponse(200, {**QUIZ, "questions": [broken]}))
    assert error.kind == "bad_response"


def test_generation_timeouts_tell_the_user_that_asking_again_may_be_free_but_chat_does_not() -> None:
    for call in (lambda api: api.summary(DOC), lambda api: api.quiz(DOC)):
        error = failure(call, requests.ReadTimeout("x"))
        assert error.kind == "timeout" and "120 giây" in error.message and "không tốn thêm lượt gọi" in error.message
    chat = failure(lambda api: api.chat("q"), requests.ReadTimeout("x"))
    assert "không tốn thêm lượt gọi" not in chat.message


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [(409, "generation.no_content", "Không đọc được nội dung của tài liệu trong chỉ mục. Hãy xóa tài liệu và tải lên lại."),
     (422, "generation.empty_selection", "Không có nội dung nào khớp với khoảng trang hoặc chủ đề đã chọn."),
     (502, "generation.malformed_output", "Mô hình AI trả về kết quả không đúng định dạng. Hãy thử lại."),
     (429, "rag.rate_limited", "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau.")],
)
@pytest.mark.parametrize("call", [lambda api: api.summary(DOC), lambda api: api.quiz(DOC)])
def test_generation_errors_show_the_backends_message(call, status: int, code: str, message: str) -> None:
    error = failure(call, FakeResponse(status, error_body(code, message)))
    assert (error.kind, error.status, error.code, error.message) == ("http", status, code, message)
