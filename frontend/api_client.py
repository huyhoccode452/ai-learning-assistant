"""Thin HTTP client for the FastAPI backend. No Streamlit import: it can be tested without a browser.

Rules this module keeps:
  * It holds no secret. The only configuration is the base URL; the Gemini key stays in the backend.
  * Every request has an explicit (connect, read) timeout. `requests` applies the read timeout to each
    socket read, not to the whole call; for the backend's single JSON reply that is effectively the
    total wait, but it is not a hard wall-clock limit (a server that trickled bytes could exceed it).
  * Every failure becomes an `ApiError` whose `message` is safe to show to a user: Vietnamese, taken from
    the backend's error body when there is one, and never a stack trace, exception text or raw response body.
"""

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
CONNECT_TIMEOUT_S = 5.0
HEALTH_READ_TIMEOUT_S = 3.0
LIST_READ_TIMEOUT_S = 10.0
UPLOAD_READ_TIMEOUT_S = 60.0
CHAT_READ_TIMEOUT_S = 60.0  # the UI gives up after about this long (the server may still be working)
# Summary and quiz make 1 to 7 LLM calls (a real quiz of 10 with verification took 72 s because of one 504 + retry),
# so they get a longer wait. Their result is stored by the server: asking again after a timeout is free if it finished.
GENERATE_READ_TIMEOUT_S = 120.0
MAX_MESSAGE_CHARS = 500
GENERATE_TIMEOUT_HINT = " Nếu máy chủ đã làm xong, bấm lại sẽ lấy kết quả đã lưu ngay, không tốn thêm lượt gọi mô hình."

# Used only when the response has no usable error body (for example a proxy page): the text is ours, not theirs.
_STATUS_MESSAGES = {
    400: "Yêu cầu không hợp lệ.",
    404: "Không tìm thấy dữ liệu yêu cầu.",
    409: "Thao tác không phù hợp với trạng thái hiện tại của tài liệu.",
    413: "Tệp vượt quá dung lượng tối đa cho phép.",
    415: "Định dạng tệp không được hỗ trợ.",
    422: "Dữ liệu gửi lên không hợp lệ.",
    429: "Dịch vụ đang giới hạn số lượt gọi. Hãy thử lại sau ít phút.",
    502: "Máy chủ gặp lỗi khi gọi dịch vụ phía sau. Hãy thử lại sau.",
    503: "Dịch vụ tạm thời không khả dụng. Hãy thử lại sau.",
    504: "Máy chủ phản hồi quá lâu. Hãy thử lại.",
}


class ApiError(Exception):
    """A failed call. `kind`: 'unreachable' | 'timeout' | 'bad_response' | 'http'."""

    def __init__(self, kind: str, message: str, *, status: int | None = None, code: str | None = None,
                 retry_after: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message  # always safe to display
        self.status = status
        self.code = code
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class Health:
    status: str
    qdrant_reachable: bool


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    filename: str
    status: str  # 'processing' | 'ready' | 'failed'
    error_code: str | None
    error_message: str | None
    num_pages: int | None
    num_chunks: int | None


@dataclass(frozen=True, slots=True)
class Upload:
    document: Document
    already_indexed: bool  # HTTP 200: this exact file was already 'ready'


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    filename: str
    page: int | None
    score: float
    excerpt: str
    cited: bool


@dataclass(frozen=True, slots=True)
class Latency:
    embed_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class ChatResult:
    answer: str
    sources: tuple[Source, ...]
    used_fallback: bool
    best_score: float | None
    truncated: bool
    latency: Latency


@dataclass(frozen=True, slots=True)
class Summary:
    summary: str
    style: str
    num_chunks_used: int
    num_chunks_total: int
    truncated: bool
    warning: str | None
    llm_calls: int  # made by this request; 0 = answered from the server's stored result
    cached: bool
    latency_ms: float


@dataclass(frozen=True, slots=True)
class QuizSource:
    chunk_index: int
    page: int | None
    filename: str
    quote: str


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    id: int
    question: str
    options: tuple[str, ...]  # exactly 4, already shuffled by the server
    answer_index: int  # kept out of the page until the user presses "Kiểm tra"
    explanation: str
    source: QuizSource
    verified: bool | None


@dataclass(frozen=True, slots=True)
class VerifierDecision:
    question: str
    key_supported: bool
    another_option_correct: bool
    reason: str
    kept: bool
    in_quiz: bool


@dataclass(frozen=True, slots=True)
class Quiz:
    questions: tuple[QuizQuestion, ...]
    requested: int
    dropped: dict[str, int]
    message: str | None
    warnings: tuple[str, ...]
    verify_requested: bool
    verified: bool
    decisions: tuple[VerifierDecision, ...]
    llm_calls: int
    cached: bool
    latency_ms: float


def _clip(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 1] + "…"


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class ApiClient:
    """One method per endpoint. `session` can be replaced by a fake in tests."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, session: Any | None = None) -> None:
        parts = urlsplit(base_url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("API_BASE_URL must start with http:// or https:// and contain a host")
        self.base_url = base_url.strip().rstrip("/")
        # What we show to users: scheme://host:port only, so credentials in the URL can never reach the screen.
        port = f":{parts.port}" if parts.port else ""
        self.display_url = f"{parts.scheme}://{parts.hostname}{port}"
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, *, read_timeout: float, timeout_hint: str = "",
                 **kwargs: Any) -> tuple[Any, requests.Response]:
        url = self.base_url + path
        try:
            response = self._session.request(method, url, timeout=(CONNECT_TIMEOUT_S, read_timeout), **kwargs)
        except requests.ConnectTimeout as exc:  # ConnectTimeout is also a ConnectionError: check it first
            raise self._unreachable() from exc
        except requests.Timeout as exc:
            raise ApiError("timeout", (
                f"Máy chủ chưa trả lời sau {read_timeout:g} giây. Yêu cầu có thể vẫn đang được xử lý ở phía máy chủ, "
                "nên chưa chắc nó đã thất bại. Hãy đợi một chút, kiểm tra lại danh sách tài liệu rồi thử lại." + timeout_hint)) from exc
        except requests.ConnectionError as exc:
            raise self._unreachable() from exc
        except requests.RequestException as exc:  # anything else from the HTTP layer
            logger.warning("HTTP layer error: %s", type(exc).__name__)
            raise ApiError("unreachable", "Lỗi mạng khi nói chuyện với máy chủ API. Hãy thử lại.") from exc

        if response.status_code >= 400:
            raise self._http_error(response)
        try:
            return response.json(), response
        except ValueError as exc:
            raise ApiError("bad_response", "Máy chủ trả về dữ liệu không đọc được. Hãy thử lại sau.",
                           status=response.status_code) from exc

    def _unreachable(self) -> ApiError:
        return ApiError("unreachable", f"Không kết nối được tới máy chủ API tại {self.display_url}. "
                                       "Hãy kiểm tra máy chủ đang chạy (uvicorn app.main:app) rồi thử lại.")

    @staticmethod
    def _http_error(response: requests.Response) -> ApiError:
        status = response.status_code
        retry_after = _int_or_none(response.headers.get("Retry-After"))
        code = message = None
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"].strip():
                message = _clip(error["message"].strip())
                code = error.get("code") if isinstance(error.get("code"), str) else None
        except ValueError:
            pass  # not JSON: fall back to our own text, never the raw body
        if message is None:
            message = _STATUS_MESSAGES.get(status) or (
                f"Máy chủ gặp lỗi (mã {status}). Hãy thử lại sau." if status >= 500 else f"Yêu cầu bị từ chối (mã {status}).")
        if status == 429 and retry_after:
            message += f" (Máy chủ gợi ý chờ khoảng {retry_after} giây.)"
        return ApiError("http", message, status=status, code=code, retry_after=retry_after)

    # ------------------------------------------------------------------ parsing

    @staticmethod
    def _parse(factory: Any, data: Any) -> Any:
        try:
            return factory(data)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ApiError("bad_response", "Máy chủ trả về dữ liệu không đúng định dạng. Hãy thử lại sau.") from exc

    @staticmethod
    def _document(d: dict) -> Document:
        return Document(doc_id=d["doc_id"], filename=d["filename"], status=d["status"], error_code=d.get("error_code"),
                        error_message=d.get("error_message"), num_pages=d.get("num_pages"), num_chunks=d.get("num_chunks"))

    @staticmethod
    def _chat(d: dict) -> ChatResult:
        lat = d["latency"]
        return ChatResult(
            answer=d["answer"], used_fallback=bool(d["used_fallback"]), truncated=bool(d.get("truncated", False)),
            best_score=None if d.get("best_score") is None else float(d["best_score"]),
            latency=Latency(float(lat["embed_ms"]), float(lat["search_ms"]), float(lat["llm_ms"]), float(lat["total_ms"])),
            sources=tuple(Source(number=int(s["number"]), filename=s["filename"], page=s.get("page"), score=float(s["score"]),
                                 excerpt=s.get("excerpt", ""), cited=bool(s["cited"])) for s in d["sources"]),
        )

    @staticmethod
    def _summary(d: dict) -> Summary:
        return Summary(summary=d["summary"], style=d["style"], num_chunks_used=int(d["num_chunks_used"]),
                       num_chunks_total=int(d["num_chunks_total"]), truncated=bool(d["truncated"]), warning=d.get("warning"),
                       llm_calls=int(d["llm_calls"]), cached=bool(d["cached"]), latency_ms=float(d["latency_ms"]))

    @staticmethod
    def _quiz(d: dict) -> Quiz:
        questions = []
        for q in d["questions"]:
            options = tuple(str(o) for o in q["options"])
            if len(options) != 4 or not 0 <= int(q["answer_index"]) <= 3:
                raise ValueError("a quiz question must have 4 options and a key between 0 and 3")
            src = q["source"]
            questions.append(QuizQuestion(
                id=int(q["id"]), question=q["question"], options=options, answer_index=int(q["answer_index"]),
                explanation=q["explanation"], verified=q.get("verified"),
                source=QuizSource(int(src["chunk_index"]), src.get("page"), src["filename"], src["quote"])))
        return Quiz(
            questions=tuple(questions), requested=int(d["requested"]), dropped={str(k): int(v) for k, v in d["dropped"].items()},
            message=d.get("message"), warnings=tuple(str(w) for w in d.get("warnings", [])),
            verify_requested=bool(d["verify_requested"]), verified=bool(d["verified"]),
            decisions=tuple(VerifierDecision(v["question"], bool(v["key_supported"]), bool(v["another_option_correct"]),
                                             v["reason"], bool(v["kept"]), bool(v.get("in_quiz", True)))
                            for v in d.get("verifier_decisions", [])),
            llm_calls=int(d["llm_calls"]), cached=bool(d["cached"]), latency_ms=float(d["latency_ms"]))

    # ------------------------------------------------------------------ endpoints

    def health(self) -> Health:
        data, _ = self._request("GET", "/health", read_timeout=HEALTH_READ_TIMEOUT_S)
        return self._parse(lambda d: Health(str(d["status"]), bool(d["qdrant_reachable"])), data)

    def list_documents(self) -> list[Document]:
        data, _ = self._request("GET", "/documents", read_timeout=LIST_READ_TIMEOUT_S)
        return self._parse(lambda d: [self._document(x) for x in d], data)

    def get_document(self, doc_id: str) -> Document:
        data, _ = self._request("GET", f"/documents/{doc_id}", read_timeout=LIST_READ_TIMEOUT_S)
        return self._parse(self._document, data)

    def upload(self, filename: str, content: bytes) -> Upload:
        data, response = self._request("POST", "/documents", read_timeout=UPLOAD_READ_TIMEOUT_S,
                                       files={"file": (filename, content)})
        return self._parse(lambda d: Upload(self._document(d), already_indexed=response.status_code == 200), data)

    def delete_document(self, doc_id: str) -> int:
        """Returns how many chunks were removed."""
        data, _ = self._request("DELETE", f"/documents/{doc_id}", read_timeout=LIST_READ_TIMEOUT_S)
        return self._parse(lambda d: int(d["deleted_chunks"]), data)

    def chat(self, question: str, doc_ids: list[str] | None = None) -> ChatResult:
        body: dict[str, Any] = {"question": question}
        if doc_ids is not None:
            body["doc_ids"] = doc_ids
        data, _ = self._request("POST", "/chat", read_timeout=CHAT_READ_TIMEOUT_S, json=body)
        return self._parse(self._chat, data)

    def summary(self, doc_id: str, style: str = "short", refresh: bool = False) -> Summary:
        data, _ = self._request("POST", f"/documents/{doc_id}/summary", read_timeout=GENERATE_READ_TIMEOUT_S,
                                timeout_hint=GENERATE_TIMEOUT_HINT, json={"style": style, "refresh": refresh})
        return self._parse(self._summary, data)

    def quiz(self, doc_id: str, *, num_questions: int = 5, topic: str | None = None, page_from: int | None = None,
             page_to: int | None = None, verify: bool = True, refresh: bool = False) -> Quiz:
        body: dict[str, Any] = {"num_questions": num_questions, "verify": verify, "refresh": refresh}
        for key, value in (("topic", topic), ("page_from", page_from), ("page_to", page_to)):
            if value is not None:
                body[key] = value
        data, _ = self._request("POST", f"/documents/{doc_id}/quiz", read_timeout=GENERATE_READ_TIMEOUT_S,
                                timeout_hint=GENERATE_TIMEOUT_HINT, json=body)
        return self._parse(self._quiz, data)
