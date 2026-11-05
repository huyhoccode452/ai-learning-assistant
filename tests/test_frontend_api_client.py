"""Tests for frontend/api_client.py with a fake HTTP layer: no network, no server."""

from typing import Any

import pytest
import requests

from frontend.api_client import (
    CHAT_READ_TIMEOUT_S,
    CONNECT_TIMEOUT_S,
    ApiClient,
    ApiError,
)

DOC = {"doc_id": "a" * 16, "filename": "bao_cao.pdf", "status": "ready", "error_code": None, "error_message": None,
       "num_pages": 9, "num_chunks": 29, "created_at": "2026-01-01T00:00:00+00:00"}
CHAT = {
    "answer": "work_mem mặc định là 4MB [1].", "used_fallback": False, "best_score": 0.83, "truncated": False,
    "latency": {"embed_ms": 100.0, "search_ms": 20.0, "llm_ms": 1500.0, "total_ms": 1620.0},
    "sources": [
        {"number": 1, "doc_id": "a" * 16, "filename": "bao_cao.pdf", "page": 5, "chunk_index": 12, "score": 0.83, "excerpt": "work_mem (mặc định 4MB)", "cited": True},
        {"number": 2, "doc_id": "a" * 16, "filename": "ghi_chu.txt", "page": None, "chunk_index": 0, "score": 0.7, "excerpt": "nội dung", "cited": False},
    ],
}


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, *, text: str | None = None, headers: dict | None = None) -> None:
        self.status_code = status
        self._body = body
        self.text = text if text is not None else ""
        self.headers = headers or {}

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Plays back one scripted reply (a FakeResponse, or an exception to raise) per request."""

    def __init__(self, *script: "FakeResponse | Exception") -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*script: "FakeResponse | Exception", base_url: str = "http://api.test:8000") -> tuple[ApiClient, FakeSession]:
    session = FakeSession(*script)
    return ApiClient(base_url, session=session), session


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def failure(call, *script: "FakeResponse | Exception") -> ApiError:
    api, _ = client(*script)
    with pytest.raises(ApiError) as caught:
        call(api)
    return caught.value


# ---------------------------------------------------------------------- the API's own messages are shown as they are


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (409, "document_processing", "Tài liệu đang được xử lý; hãy đợi xử lý xong rồi xóa."),
        (413, "ingestion.file_too_large", "Tệp vượt quá dung lượng tối đa cho phép."),
        (415, "ingestion.unsupported_extension", "Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .pdf, .docx và .txt."),
        (422, "ingestion.scanned_pdf", "PDF này là bản quét (chỉ có ảnh), không có lớp văn bản. Hệ thống chưa hỗ trợ OCR."),
        (429, "rag.rate_limited", "Dịch vụ AI đang giới hạn số lượt gọi hoặc đã hết hạn mức (gói miễn phí). Hãy thử lại sau."),
        (500, "internal_error", "Đã xảy ra lỗi nội bộ. Vui lòng thử lại sau."),
        (503, "store.backend_unavailable", "Không kết nối được cơ sở dữ liệu vector (Qdrant). Hãy kiểm tra Qdrant đang chạy."),
        (504, "rag.timeout", "Mô hình AI phản hồi quá lâu. Hãy thử lại."),
    ],
)
def test_http_errors_show_the_backends_vietnamese_message(status: int, code: str, message: str) -> None:
    error = failure(lambda api: api.upload("a.pdf", b"x"), FakeResponse(status, error_body(code, message)))
    assert (error.kind, error.status, error.code, error.message) == ("http", status, code, message)


def test_a_429_adds_the_servers_retry_hint_only_when_it_is_a_number() -> None:
    body = error_body("rag.rate_limited", "Hãy thử lại sau.")
    hinted = failure(lambda api: api.chat("q"), FakeResponse(429, body, headers={"Retry-After": "60"}))
    assert hinted.message == "Hãy thử lại sau. (Máy chủ gợi ý chờ khoảng 60 giây.)" and hinted.retry_after == 60
    plain = failure(lambda api: api.chat("q"), FakeResponse(429, body, headers={"Retry-After": "Wed, 21 Oct 2026"}))
    assert plain.message == "Hãy thử lại sau." and plain.retry_after is None
    other = failure(lambda api: api.chat("q"), FakeResponse(503, error_body("x", "Lỗi."), headers={"Retry-After": "60"}))
    assert other.message == "Lỗi."  # the hint belongs to 429 only


# ---------------------------------------------------------------------- no usable body -> our own text, never theirs


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "Yêu cầu không hợp lệ."),
        (404, "Không tìm thấy dữ liệu yêu cầu."),
        (409, "Thao tác không phù hợp với trạng thái hiện tại của tài liệu."),
        (413, "Tệp vượt quá dung lượng tối đa cho phép."),
        (415, "Định dạng tệp không được hỗ trợ."),
        (422, "Dữ liệu gửi lên không hợp lệ."),
        (429, "Dịch vụ đang giới hạn số lượt gọi. Hãy thử lại sau ít phút."),
        (502, "Máy chủ gặp lỗi khi gọi dịch vụ phía sau. Hãy thử lại sau."),
        (504, "Máy chủ phản hồi quá lâu. Hãy thử lại."),
        (503, "Dịch vụ tạm thời không khả dụng. Hãy thử lại sau."),
        (500, "Máy chủ gặp lỗi (mã 500). Hãy thử lại sau."),
        (507, "Máy chủ gặp lỗi (mã 507). Hãy thử lại sau."),
        (418, "Yêu cầu bị từ chối (mã 418)."),
    ],
)
def test_without_an_error_body_the_message_comes_from_the_status(status: int, expected: str) -> None:
    html = FakeResponse(status, None, text="<html>Traceback: File C:\\secret\\app.py line 3, key=AIzaSECRET</html>")
    error = failure(lambda api: api.list_documents(), html)
    assert error.message == expected and error.kind == "http" and error.code is None
    assert "Traceback" not in error.message and "secret" not in error.message.lower() and "AIza" not in error.message


@pytest.mark.parametrize(
    "body",
    [{"detail": "Not Found"}, {"error": "string"}, {"error": {"code": "x"}}, {"error": {"code": "x", "message": "   "}},
     {"error": {"message": 5}}, ["list"], "text"],
)
def test_an_error_body_of_the_wrong_shape_falls_back_to_the_status_text(body: Any) -> None:
    error = failure(lambda api: api.list_documents(), FakeResponse(404, body))
    assert error.message == "Không tìm thấy dữ liệu yêu cầu." and error.code is None


def test_a_very_long_error_message_is_clipped() -> None:
    error = failure(lambda api: api.chat("q"), FakeResponse(500, error_body("x", "a" * 5000)))
    assert len(error.message) == 500 and error.message.endswith("…")


# ---------------------------------------------------------------------- network failures


def test_connection_failure_is_unreachable_and_leaks_nothing() -> None:
    boom = requests.ConnectionError("HTTPConnectionPool(host='api.test'): C:\\Users\\x\\secret.py token=abc123")
    error = failure(lambda api: api.health(), boom)
    assert error.kind == "unreachable" and error.status is None
    assert "http://api.test:8000" in error.message and "uvicorn app.main:app" in error.message
    assert "abc123" not in error.message and "secret" not in error.message and "Pool" not in error.message


def test_a_connect_timeout_means_unreachable_not_slow() -> None:
    assert failure(lambda api: api.health(), requests.ConnectTimeout("x")).kind == "unreachable"


def test_a_read_timeout_does_not_claim_the_request_failed() -> None:
    error = failure(lambda api: api.chat("q"), requests.ReadTimeout("x"))
    assert error.kind == "timeout" and "60 giây" in error.message
    assert "vẫn đang được xử lý ở phía máy chủ" in error.message and "chưa chắc nó đã thất bại" in error.message


def test_other_network_errors_get_a_generic_message() -> None:
    error = failure(lambda api: api.health(), requests.exceptions.ChunkedEncodingError("secret"))
    assert error.kind == "unreachable" and "secret" not in error.message


def test_credentials_in_the_base_url_are_never_shown() -> None:
    api, _ = client(requests.ConnectionError("x"), base_url="http://user:pa55word@api.test:9000/")
    with pytest.raises(ApiError) as caught:
        api.health()
    assert "pa55word" not in caught.value.message and "http://api.test:9000" in caught.value.message
    assert api.base_url == "http://user:pa55word@api.test:9000"


@pytest.mark.parametrize("bad", ["", "api.test:8000", "ftp://api.test", "http://", "file:///etc/passwd"])
def test_a_bad_base_url_is_rejected_at_start(bad: str) -> None:
    with pytest.raises(ValueError):
        ApiClient(bad)


# ---------------------------------------------------------------------- unusable success replies


def test_a_success_reply_that_is_not_json_is_a_bad_response() -> None:
    error = failure(lambda api: api.list_documents(), FakeResponse(200, None, text="<html>proxy</html>"))
    assert error.kind == "bad_response" and "proxy" not in error.message


@pytest.mark.parametrize("call", [lambda a: a.list_documents(), lambda a: a.get_document("a" * 16), lambda a: a.chat("q"),
                                  lambda a: a.upload("a.pdf", b"x"), lambda a: a.delete_document("a" * 16), lambda a: a.health()])
def test_a_success_reply_of_the_wrong_shape_is_a_bad_response(call) -> None:
    error = failure(call, FakeResponse(200, {"unexpected": True}))
    assert error.kind == "bad_response"


# ---------------------------------------------------------------------- requests that are sent


def test_every_request_has_an_explicit_connect_and_read_timeout() -> None:
    api, session = client(FakeResponse(200, {"status": "ok", "qdrant_reachable": True}), FakeResponse(200, [DOC]),
                          FakeResponse(200, DOC), FakeResponse(202, DOC), FakeResponse(200, {"doc_id": "a" * 16, "deleted_chunks": 3}),
                          FakeResponse(200, CHAT))
    api.health()
    api.list_documents()
    api.get_document("a" * 16)
    api.upload("a.pdf", b"x")
    api.delete_document("a" * 16)
    api.chat("q")
    read_timeouts = [c["timeout"][1] for c in session.calls]
    assert all(c["timeout"][0] == CONNECT_TIMEOUT_S == 5.0 for c in session.calls)  # pinned: a hung connect must fail fast
    assert read_timeouts == [3.0, 10.0, 10.0, 60.0, 10.0, CHAT_READ_TIMEOUT_S] and CHAT_READ_TIMEOUT_S == 60.0


def test_methods_paths_and_payloads() -> None:
    api, session = client(FakeResponse(200, [DOC]), FakeResponse(200, DOC), FakeResponse(202, DOC),
                          FakeResponse(200, {"doc_id": "a" * 16, "deleted_chunks": 3}), FakeResponse(200, CHAT), FakeResponse(200, CHAT))
    api.list_documents()
    api.get_document("b" * 16)
    api.upload("báo cáo.pdf", b"%PDF-bytes")
    assert api.delete_document("c" * 16) == 3
    api.chat("work_mem là gì?")
    api.chat("work_mem là gì?", ["a" * 16])
    assert [(c["method"], c["url"]) for c in session.calls] == [
        ("GET", "http://api.test:8000/documents"), ("GET", f"http://api.test:8000/documents/{'b' * 16}"),
        ("POST", "http://api.test:8000/documents"), ("DELETE", f"http://api.test:8000/documents/{'c' * 16}"),
        ("POST", "http://api.test:8000/chat"), ("POST", "http://api.test:8000/chat")]
    assert session.calls[2]["files"] == {"file": ("báo cáo.pdf", b"%PDF-bytes")}
    assert session.calls[4]["json"] == {"question": "work_mem là gì?"}  # no doc_ids key: search every ready document
    assert session.calls[5]["json"] == {"question": "work_mem là gì?", "doc_ids": ["a" * 16]}
    assert all("headers" not in c and "auth" not in c for c in session.calls)  # nothing secret is ever sent


def test_an_empty_document_list_is_sent_as_empty_never_as_search_everything() -> None:
    api, session = client(FakeResponse(200, CHAT))
    api.chat("q", [])  # the backend rejects it (422); it must not silently become "all documents"
    assert session.calls[0]["json"] == {"question": "q", "doc_ids": []}


def test_the_trailing_slash_of_the_base_url_is_dropped() -> None:
    api, session = client(FakeResponse(200, []), base_url="http://api.test:8000/")
    api.list_documents()
    assert session.calls[0]["url"] == "http://api.test:8000/documents"


# ---------------------------------------------------------------------- successful replies


def test_upload_tells_a_new_file_from_one_that_is_already_indexed() -> None:
    api, _ = client(FakeResponse(202, {**DOC, "status": "processing"}), FakeResponse(200, DOC))
    fresh, duplicate = api.upload("a.pdf", b"x"), api.upload("a.pdf", b"x")
    assert (fresh.already_indexed, fresh.document.status) == (False, "processing")
    assert (duplicate.already_indexed, duplicate.document.status) == (True, "ready")


def test_chat_reply_is_parsed_with_sources_and_latency() -> None:
    result, = [client(FakeResponse(200, CHAT))[0].chat("q")]
    assert result.answer.endswith("[1].") and not result.used_fallback and result.best_score == 0.83
    assert [(s.number, s.filename, s.page, s.cited) for s in result.sources] == [(1, "bao_cao.pdf", 5, True), (2, "ghi_chu.txt", None, False)]
    assert (result.latency.embed_ms, result.latency.llm_ms, result.latency.total_ms) == (100.0, 1500.0, 1620.0)


def test_a_fallback_reply_without_hits_has_no_score() -> None:
    fallback = {**CHAT, "used_fallback": True, "best_score": None, "sources": [], "answer": "Không tìm thấy."}
    result = client(FakeResponse(200, fallback))[0].chat("q")
    assert result.used_fallback and result.best_score is None and result.sources == ()


def test_documents_are_parsed_including_failed_ones() -> None:
    failed = {**DOC, "doc_id": "d" * 16, "status": "failed", "error_code": "interrupted", "error_message": "Máy chủ đã khởi động lại.", "num_pages": None}
    docs = client(FakeResponse(200, [DOC, failed]))[0].list_documents()
    assert [d.status for d in docs] == ["ready", "failed"]
    assert (docs[1].error_code, docs[1].error_message, docs[1].num_pages) == ("interrupted", "Máy chủ đã khởi động lại.", None)
