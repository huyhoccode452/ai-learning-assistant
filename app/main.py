"""FastAPI application: document upload (indexed in the background), document management, chat.

Choices worth knowing (details in docs/known_issues.md):
  * Endpoints are plain `def`, not `async def`. Everything they do is blocking (SQLite, Qdrant and
    Gemini calls, PDF parsing). FastAPI runs `def` endpoints in a worker-thread pool, so the event
    loop stays free to serve other requests; an `async def` endpoint that called these would freeze
    the whole server while it waited. The indexing background job is also a plain `def`, so
    Starlette runs it in the same thread pool.
  * Upload returns 202 right away; embedding happens in a BackgroundTask (free-tier embedding takes
    minutes, KI-12). `documents.status` is the contract with the client: processing -> ready|failed.
  * Request bodies are size-capped by middleware BEFORE the multipart parser spools them (KI-09).
  * Every expected failure becomes {"error": {"code", "message", "details"?}} with a Vietnamese
    message and a stable machine-readable code. Raw exception text is never sent to the client.
"""

import logging
import os
import time
import uuid
from collections.abc import Callable, MutableMapping
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Path as PathParam, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.db import Database, Document, DocStatus
from app.embeddings import Embedder, EmbeddingError, GeminiEmbedder
from app.embeddings import ErrorCode as EmbeddingCode
from app.ingestion import (
    ALLOWED_EXTENSIONS,
    IngestionError,
    IngestResult,
    compute_doc_id,
    ingest_file,
    validate_upload,
)
from app.generation import GenerationError, GenerationErrorCode, GenResult
from app.ingestion import ErrorCode as IngestionCode
from app.quiz import generate_quiz
from app.rag import GeminiLLM, LLM, MAX_QUESTION_CHARS, RagAnswer, RagError, RagErrorCode, RagPipeline
from app.summary import SummaryStyle, summarize_document
from app.vectorstore import QdrantStore, StoreErrorCode, VectorStoreError

logger = logging.getLogger(__name__)

STREAMLIT_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]  # Streamlit's default port
READ_CHUNK_BYTES = 64 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries + part headers around the file in the body
DEFAULT_BODY_LIMIT_BYTES = 1024 * 1024  # every endpoint except the upload only takes small JSON
MAX_DOC_IDS = 50
TOP_K_MAX = 20
DOC_ID_PATTERN = r"^[0-9a-f]{16}$"  # compute_doc_id: 16 hex characters

DocIdPath = Annotated[str, PathParam(pattern=DOC_ID_PATTERN, description="Document id (16 hex chars)")]
DocIdStr = Annotated[str, StringConstraints(pattern=DOC_ID_PATTERN)]

# ====================================================================== errors


class ApiError(Exception):
    """An error with everything the HTTP layer needs. Messages are written for end users."""

    def __init__(self, status: int, code: str, message: str, details: Any = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


INTERNAL_ERROR = ("internal_error", "Đã xảy ra lỗi nội bộ. Vui lòng thử lại sau.")

# enum member -> (HTTP status, Vietnamese message). Written by hand so that no English exception
# text, file name or path can reach a user through these.
_INGESTION_ERRORS: dict[IngestionCode, tuple[int, str]] = {
    IngestionCode.UNSUPPORTED_EXTENSION: (415, "Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .pdf, .docx và .txt."),
    IngestionCode.EMPTY_FILE: (400, "Tệp rỗng."),
    IngestionCode.FILE_TOO_LARGE: (413, "Tệp vượt quá dung lượng tối đa cho phép."),
    IngestionCode.CONTENT_MISMATCH: (415, "Nội dung tệp không khớp với phần mở rộng (ví dụ tệp .pdf nhưng không phải PDF thật)."),
    IngestionCode.CORRUPTED_FILE: (422, "Tệp bị hỏng hoặc không đọc được."),
    IngestionCode.ENCRYPTED_PDF: (422, "PDF được bảo vệ bằng mật khẩu. Hãy gỡ mật khẩu rồi tải lại."),
    IngestionCode.SCANNED_PDF: (422, "PDF này là bản quét (chỉ có ảnh), không có lớp văn bản. Hệ thống chưa hỗ trợ OCR."),
    IngestionCode.UNDECODABLE_TEXT: (422, "Không đọc được bảng mã của tệp văn bản. Hãy lưu lại dưới dạng UTF-8."),
    IngestionCode.NO_TEXT: (422, "Không trích xuất được văn bản nào từ tệp."),
}
_EMBEDDING_ERRORS: dict[EmbeddingCode, tuple[int, str]] = {
    EmbeddingCode.NOT_CONFIGURED: (503, "Dịch vụ embedding chưa được cấu hình trên máy chủ."),
    EmbeddingCode.EMPTY_TEXT: (400, "Nội dung cần xử lý đang trống."),
    EmbeddingCode.TEXT_TOO_LONG: (422, "Nội dung quá dài để xử lý."),
    EmbeddingCode.DIMENSION_MISMATCH: (500, "Cấu hình embedding không khớp với chỉ mục hiện có. Hãy liên hệ quản trị viên."),
    EmbeddingCode.BAD_RESPONSE: (502, "Dịch vụ embedding trả về dữ liệu không hợp lệ."),
    EmbeddingCode.REQUEST_REJECTED: (502, "Dịch vụ embedding từ chối yêu cầu."),
    EmbeddingCode.AUTH_ERROR: (502, "Máy chủ không xác thực được với dịch vụ AI. Hãy liên hệ quản trị viên."),
    EmbeddingCode.RATE_LIMITED: (429, "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau ít phút."),
    EmbeddingCode.SERVICE_UNAVAILABLE: (503, "Dịch vụ AI tạm thời không khả dụng. Hãy thử lại sau."),
}
_STORE_ERRORS: dict[StoreErrorCode, tuple[int, str]] = {
    StoreErrorCode.NOT_CONFIGURED: (503, "Cơ sở dữ liệu vector chưa được cấu hình trên máy chủ."),
    StoreErrorCode.COLLECTION_MISMATCH: (500, "Cơ sở dữ liệu vector không khớp cấu hình hiện tại; cần tạo lại và index lại tài liệu."),
    StoreErrorCode.INVALID_INPUT: (500, "Lỗi nội bộ khi truy cập cơ sở dữ liệu vector."),
    StoreErrorCode.BACKEND_UNAVAILABLE: (503, "Không kết nối được cơ sở dữ liệu vector (Qdrant). Hãy kiểm tra Qdrant đang chạy."),
    StoreErrorCode.BACKEND_ERROR: (502, "Cơ sở dữ liệu vector trả về lỗi."),
}
_RAG_ERRORS: dict[RagErrorCode, tuple[int, str]] = {
    RagErrorCode.NOT_CONFIGURED: (503, "Hệ thống chưa được cấu hình đầy đủ (mô hình hoặc ngưỡng điểm). Hãy liên hệ quản trị viên."),
    RagErrorCode.INVALID_INPUT: (400, "Câu hỏi không hợp lệ."),
    RagErrorCode.AUTH_ERROR: (502, "Máy chủ không xác thực được với dịch vụ AI. Hãy liên hệ quản trị viên."),
    RagErrorCode.REQUEST_REJECTED: (502, "Dịch vụ AI từ chối yêu cầu."),
    RagErrorCode.RATE_LIMITED: (429, "Dịch vụ AI đang giới hạn số lượt gọi hoặc đã hết hạn mức (gói miễn phí). Hãy thử lại sau."),
    RagErrorCode.SERVICE_UNAVAILABLE: (503, "Dịch vụ AI tạm thời không khả dụng. Hãy thử lại sau."),
    RagErrorCode.TIMEOUT: (504, "Mô hình AI phản hồi quá lâu. Hãy thử lại."),
    RagErrorCode.BLOCKED: (422, "Yêu cầu bị bộ lọc an toàn của mô hình AI chặn. Hãy thử diễn đạt khác."),
    RagErrorCode.EMPTY_RESPONSE: (502, "Mô hình AI không trả về câu trả lời. Hãy thử lại."),
    RagErrorCode.TRUNCATED: (502, "Câu trả lời bị cắt vì vượt giới hạn độ dài. Hãy thử lại hoặc hỏi ngắn gọn hơn."),
}


_GENERATION_ERRORS: dict[GenerationErrorCode, tuple[int, str]] = {
    GenerationErrorCode.NO_CONTENT: (409, "Không đọc được nội dung của tài liệu trong chỉ mục. Hãy xóa tài liệu và tải lên lại."),
    GenerationErrorCode.EMPTY_SELECTION: (422, "Không có nội dung nào khớp với khoảng trang hoặc chủ đề đã chọn."),
    GenerationErrorCode.MALFORMED_OUTPUT: (502, "Mô hình AI trả về kết quả không đúng định dạng. Hãy thử lại."),
}


def api_error_from(exc: BaseException) -> ApiError:
    """Translate a domain exception into an ApiError. Anything unknown becomes a generic 500."""
    tables: tuple[tuple[type, str, dict], ...] = (
        (IngestionError, "ingestion", _INGESTION_ERRORS),
        (EmbeddingError, "embedding", _EMBEDDING_ERRORS),
        (VectorStoreError, "store", _STORE_ERRORS),
        (RagError, "rag", _RAG_ERRORS),
        (GenerationError, "generation", _GENERATION_ERRORS),
    )
    for exc_type, prefix, table in tables:
        if isinstance(exc, exc_type):
            status, message = table.get(exc.code, (500, INTERNAL_ERROR[1]))  # type: ignore[attr-defined]
            headers = {"Retry-After": "60"} if status == 429 else None
            return ApiError(status, f"{prefix}.{exc.code.value}", message, headers=headers)  # type: ignore[attr-defined]
    return ApiError(500, *INTERNAL_ERROR)


def _error_response(status: int, code: str, message: str, details: Any = None,
                    headers: dict[str, str] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


# ====================================================================== request / response models


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable code, e.g. 'ingestion.scanned_pdf'")
    message: str = Field(description="Message for the end user (Vietnamese)")
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthOut(BaseModel):
    status: str = Field(description="'ok', or 'degraded' when Qdrant is unreachable")
    qdrant_reachable: bool


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    status: DocStatus
    error_code: str | None = None
    error_message: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None
    created_at: str


class DeleteOut(BaseModel):
    doc_id: str
    deleted_chunks: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    doc_ids: list[DocIdStr] | None = Field(
        default=None, min_length=1, max_length=MAX_DOC_IDS,
        description="Restrict the search to these documents (all must be 'ready'). Omit = all ready documents.",
    )
    top_k: int | None = Field(default=None, ge=1, le=TOP_K_MAX, description="Default: TOP_K from settings")


class SourceOut(BaseModel):
    number: int = Field(description="The [n] used in the answer")
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float
    excerpt: str
    cited: bool = Field(description="True if the answer cites this source")


class LatencyOut(BaseModel):
    embed_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    used_fallback: bool = Field(description="True = nothing passed the score threshold; the LLM was not called")
    latency: LatencyOut
    best_score: float | None
    truncated: bool


class SummaryRequest(BaseModel):
    style: SummaryStyle = Field(default=SummaryStyle.SHORT, description="'short' or 'detailed'")
    refresh: bool = Field(default=False, description="True = ignore the stored result and ask the model again (costs quota)")


class SummaryOut(BaseModel):
    summary: str
    style: SummaryStyle
    num_chunks_used: int
    num_chunks_total: int
    truncated: bool = Field(description="True = the document was longer than the hard cap and only its beginning was summarised")
    warning: str | None = None
    llm_calls: int = Field(description="LLM calls made by THIS request (0 = answered from the stored result)")
    cached: bool
    latency_ms: float
    llm_ms: float


class QuizRequest(BaseModel):
    num_questions: int = Field(default=5, ge=1, le=10)
    topic: str | None = Field(default=None, max_length=200, description="Optional topic; chunks are chosen by semantic search")
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    verify: bool = Field(default=True, description="Run the verifier call (one extra LLM call)")
    refresh: bool = Field(default=False, description="True = ignore the stored quiz and generate a new one (costs quota)")

    @model_validator(mode="after")
    def _page_range(self) -> "QuizRequest":
        if self.page_from is not None and self.page_to is not None and self.page_from > self.page_to:
            raise ValueError("page_from must not be greater than page_to")
        return self


class QuizSourceOut(BaseModel):
    chunk_index: int
    page: int | None
    filename: str
    quote: str = Field(description="Verbatim passage of the document that supports the answer")


class QuizQuestionOut(BaseModel):
    id: int
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str
    source: QuizSourceOut
    verified: bool | None = Field(description="True = the verifier confirmed it; null = not verified")


class VerifierDecisionOut(BaseModel):
    question: str
    key_supported: bool
    another_option_correct: bool
    reason: str
    kept: bool = Field(description="True = the verifier accepted it")
    in_quiz: bool = Field(default=True, description="False = accepted but cut because more than num_questions survived")


class QuizOut(BaseModel):
    questions: list[QuizQuestionOut]
    requested: int
    returned: int
    dropped: dict[str, int] = Field(description="Why questions were discarded, by reason code")
    message: str | None = Field(default=None, description="Set when fewer questions than requested survived")
    warnings: list[str] = []
    verify_requested: bool
    verified: bool
    verifier_decisions: list[VerifierDecisionOut]
    llm_calls: int
    cached: bool
    latency_ms: float
    llm_ms: float


def _cost(result: GenResult) -> dict[str, Any]:
    return {"llm_calls": result.llm_calls, "cached": result.cached, "latency_ms": result.total_ms, "llm_ms": result.llm_ms}


def _document_out(doc: Document) -> DocumentOut:
    return DocumentOut(**{name: getattr(doc, name) for name in DocumentOut.model_fields})


def _chat_response(result: RagAnswer) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(number=s.number, doc_id=s.doc_id, filename=s.filename, page=s.page,
                      chunk_index=s.chunk_index, score=s.score, excerpt=s.excerpt, cited=s.cited)
            for s in result.sources
        ],
        used_fallback=result.used_fallback,
        latency=LatencyOut(embed_ms=result.latency.embed_ms, search_ms=result.latency.search_ms,
                           llm_ms=result.latency.llm_ms, total_ms=result.latency.total_ms),
        best_score=result.best_score,
        truncated=result.truncated,
    )


# ====================================================================== dependencies
# Tests replace any of these with fakes: app.dependency_overrides[get_store] = lambda: FakeStore().


@lru_cache
def _embedder_singleton() -> Embedder:
    return GeminiEmbedder(get_settings())


@lru_cache
def _store_singleton() -> QdrantStore:
    return QdrantStore(get_settings())


@lru_cache
def _llm_singleton() -> LLM:
    return GeminiLLM(get_settings())


def get_embedder() -> Embedder:
    return _embedder_singleton()


def get_store() -> QdrantStore:
    return _store_singleton()


def get_llm() -> LLM:
    return _llm_singleton()


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Database:
    return Database(settings.sqlite_path)  # just a path; every operation opens its own connection


def get_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[QdrantStore, Depends(get_store)],
    llm: Annotated[LLM, Depends(get_llm)],
) -> RagPipeline:
    return RagPipeline(settings, embedder=embedder, store=store, llm=llm)


def _resolve(app: FastAPI, dependency: Callable[..., Any]) -> Any:
    """For code outside request handling (startup): honour dependency_overrides too."""
    return app.dependency_overrides.get(dependency, dependency)()


# ====================================================================== body size limit (KI-09)


class BodySizeLimitMiddleware:
    """Stop reading a request body once it exceeds a limit, before it is spooled anywhere.

    Starlette's multipart parser writes the whole upload to a temporary file before the endpoint
    runs, so an endpoint-level check alone would let a client push gigabytes onto the disk. Here:
      1. a declared Content-Length above the limit is refused without reading anything;
      2. otherwise the bytes are counted as they arrive (covers chunked uploads) and, past the
         limit, the app is told the client disconnected. FastAPI turns that into its own 400, so
         whatever response the app then tries to send is replaced by our 413.
    """

    def __init__(self, app: Any, limit_for: Callable[[MutableMapping[str, Any]], int]) -> None:
        self.app = app
        self.limit_for = limit_for

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            await self._too_large(scope, receive, send)
            return

        received = 0
        exceeded = False
        body_sent = False

        async def counting_receive() -> MutableMapping[str, Any]:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def replacing_send(message: MutableMapping[str, Any]) -> None:
            nonlocal body_sent
            if not exceeded:
                await send(message)
                return
            if message["type"] == "http.response.start":
                payload = self._payload()
                await send({"type": "http.response.start", "status": 413, "headers": [
                    (b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
            elif message["type"] == "http.response.body" and not body_sent:
                body_sent = True
                await send({"type": "http.response.body", "body": self._payload()})

        await self.app(scope, counting_receive, replacing_send)

    @staticmethod
    def _payload() -> bytes:
        status, message = _INGESTION_ERRORS[IngestionCode.FILE_TOO_LARGE]
        return _error_response(status, f"ingestion.{IngestionCode.FILE_TOO_LARGE.value}", message).body

    async def _too_large(self, scope: Any, receive: Any, send: Any) -> None:
        status, message = _INGESTION_ERRORS[IngestionCode.FILE_TOO_LARGE]
        response = _error_response(status, f"ingestion.{IngestionCode.FILE_TOO_LARGE.value}", message)
        await response(scope, receive, send)


# ====================================================================== indexing job


def index_document_job(db: Database, store: QdrantStore, embedder: Embedder, result: IngestResult) -> None:
    """Embed and index one document, then record the outcome. Runs after the 202 response.

    It must never leave the row in 'processing', so unexpected exceptions are caught too (logged
    with a traceback on the server, a generic message for the user).
    """
    doc_id = result.doc_id
    started = time.perf_counter()
    try:
        store.ensure_collection()
        vectors = embedder.embed_documents([chunk.text for chunk in result.chunks])
        written = store.index_document(doc_id, result.chunks, vectors)
    except (EmbeddingError, VectorStoreError) as exc:
        error = api_error_from(exc)
        logger.warning("Indexing failed doc_id=%s code=%s: %s", doc_id, error.code, exc.message)
        db.mark_failed(doc_id, error.code, error.message)
    except Exception:  # noqa: BLE001 - a background job has no caller; it must record any failure
        logger.exception("Indexing crashed doc_id=%s", doc_id)
        db.mark_failed(doc_id, *INTERNAL_ERROR)
    else:
        db.mark_ready(doc_id, written)
        logger.info("Indexed doc_id=%s: %d chunks in %.1fs", doc_id, written, time.perf_counter() - started)


# ====================================================================== helpers


def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    """Read the upload in pieces and stop as soon as it is larger than `limit` bytes."""
    buffer = bytearray()
    while piece := upload.file.read(READ_CHUNK_BYTES):
        buffer += piece
        if len(buffer) > limit:
            raise IngestionError(IngestionCode.FILE_TOO_LARGE, "The file is larger than the upload limit.")
    return bytes(buffer)


def _stored_path(settings: Settings, doc_id: str, extension: str) -> Path:
    """Generated name: the hash id plus an allow-listed extension. The user's name never touches disk."""
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("extension is not allow-listed")
    return settings.upload_dir / f"{doc_id}{extension}"


def _save_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _require_documents(db: Database, doc_ids: list[str] | None) -> list[str]:
    """The ids chat may search: the requested ones (all must be ready) or every ready document."""
    if doc_ids is None:
        ready = db.ready_doc_ids()
        if not ready:
            raise ApiError(409, "no_documents_ready",
                           "Chưa có tài liệu nào sẵn sàng. Hãy tải tài liệu lên và đợi xử lý xong.")
        return ready
    ids = list(dict.fromkeys(doc_ids))
    found = db.get_documents(ids)
    unknown = [d for d in ids if d not in found]
    if unknown:
        raise ApiError(404, "document_not_found", "Không tìm thấy một hoặc nhiều tài liệu.", details={"doc_ids": unknown})
    not_ready = [{"doc_id": d.doc_id, "status": d.status.value} for d in found.values() if d.status != DocStatus.READY]
    if not_ready:
        raise ApiError(409, "document_not_ready",
                       "Một hoặc nhiều tài liệu chưa sẵn sàng (đang xử lý hoặc xử lý thất bại).",
                       details={"documents": not_ready})
    return ids


# ====================================================================== application


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_logging()
        settings: Settings = _resolve(app, get_settings)
        database = Database(settings.sqlite_path)
        database.init_db()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        database.fail_stuck_processing(
            "interrupted", "Máy chủ đã khởi động lại khi tài liệu đang được xử lý. Hãy tải tệp lên lại."
        )
        try:  # best effort: the server must start (and /health must answer) even if Qdrant is down
            _resolve(app, get_store).ensure_collection()
        except VectorStoreError as exc:
            logger.warning("Qdrant is not ready at startup: %s", exc.message)
        yield

    app = FastAPI(
        title="AI Learning Assistant API",
        description="Upload documents, then ask questions answered only from them, with citations.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        limit_for=lambda scope: (
            _resolve(scope["app"], get_settings).max_upload_bytes + MULTIPART_OVERHEAD_BYTES
            if scope["method"] == "POST" and scope["path"] == "/documents"
            else DEFAULT_BODY_LIMIT_BYTES
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=STREAMLIT_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # ------------------------------------------------------------ error handlers

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)

    async def _domain_error(_: Request, exc: Exception) -> JSONResponse:
        error = api_error_from(exc)
        if error.status >= 500:
            logger.error("%s -> %s (%d): %s", type(exc).__name__, error.code, error.status, exc)
        return _error_response(error.status, error.code, error.message, headers=error.headers)

    for domain_exception in (IngestionError, EmbeddingError, VectorStoreError, RagError, GenerationError):
        app.add_exception_handler(domain_exception, _domain_error)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Only field names and error types: pydantic's own text can echo the submitted value.
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "type": e["type"]}
                   for e in exc.errors()]
        return _error_response(422, "invalid_request", "Yêu cầu không hợp lệ. Hãy kiểm tra lại các trường được nêu.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        known = {400: ("bad_request", "Không đọc được nội dung yêu cầu. Hãy gửi JSON hợp lệ, mã hóa UTF-8."),
                 404: ("not_found", "Không tìm thấy đường dẫn yêu cầu."),
                 405: ("method_not_allowed", "Phương thức HTTP không được hỗ trợ cho đường dẫn này.")}
        code, message = known.get(exc.status_code, (f"http_{exc.status_code}", "Yêu cầu không hợp lệ."))
        return _error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled %s on a request", type(exc).__name__, exc_info=exc)
        return _error_response(500, *INTERNAL_ERROR)

    # ------------------------------------------------------------ routes

    @app.get("/health", response_model=HealthOut)
    def health(store: Annotated[QdrantStore, Depends(get_store)]) -> HealthOut:
        reachable = store.is_reachable()
        return HealthOut(status="ok" if reachable else "degraded", qdrant_reachable=reachable)

    @app.post(
        "/documents", status_code=202, response_model=DocumentOut,
        summary="Upload a document (indexed in the background)",
        responses={200: {"description": "The same file is already indexed (status 'ready')"},
                   **{code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 429, 500, 503)}},
    )
    def upload_document(
        response: Response,
        background: BackgroundTasks,
        file: Annotated[UploadFile, File(description="PDF, DOCX or TXT")],
        settings: Annotated[Settings, Depends(get_settings)],
        db: Annotated[Database, Depends(get_db)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
        store: Annotated[QdrantStore, Depends(get_store)],
    ) -> DocumentOut:
        data = _read_bounded(file, settings.max_upload_bytes)
        # Cheap checks first (extension, size, magic bytes). ingest_file repeats them; that is
        # intentional: the duplicate lookup below must only happen for a valid file.
        upload = validate_upload(file.filename or "", data, settings.max_upload_mb)
        doc_id = compute_doc_id(data)

        existing = db.get_document(doc_id)
        if existing and existing.status in (DocStatus.READY, DocStatus.PROCESSING):
            response.status_code = 200 if existing.status == DocStatus.READY else 202
            logger.info("Upload of doc_id=%s ignored: already %s", doc_id, existing.status.value)
            return _document_out(existing)

        result = ingest_file(upload.filename, data, settings)  # extract + chunk: fast, so errors are immediate 4xx
        try:
            _save_atomically(_stored_path(settings, doc_id, upload.extension), data)
        except OSError as exc:
            logger.error("Cannot store upload doc_id=%s: %s", doc_id, type(exc).__name__)
            raise ApiError(500, "storage_error", "Không lưu được tệp trên máy chủ.") from exc

        document, claimed = db.claim_document(doc_id, result.filename, result.num_pages, len(result.chunks))
        if claimed:
            background.add_task(index_document_job, db, store, embedder, result)
            logger.info("Accepted %s doc_id=%s: %d bytes, %d chunks", result.filename, doc_id, len(data), len(result.chunks))
        else:  # another request claimed it between our lookup and now
            response.status_code = 200 if document.status == DocStatus.READY else 202
        return _document_out(document)

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(db: Annotated[Database, Depends(get_db)]) -> list[DocumentOut]:
        return [_document_out(d) for d in db.list_documents()]

    @app.get("/documents/{doc_id}", response_model=DocumentOut,
             responses={404: {"model": ErrorResponse}})
    def get_document(doc_id: DocIdPath, db: Annotated[Database, Depends(get_db)]) -> DocumentOut:
        document = db.get_document(doc_id)
        if document is None:
            raise ApiError(404, "document_not_found", "Không tìm thấy tài liệu.")
        return _document_out(document)

    @app.delete("/documents/{doc_id}", response_model=DeleteOut,
                responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
    def delete_document(
        doc_id: DocIdPath,
        settings: Annotated[Settings, Depends(get_settings)],
        db: Annotated[Database, Depends(get_db)],
        store: Annotated[QdrantStore, Depends(get_store)],
    ) -> DeleteOut:
        document = db.get_document(doc_id)
        if document is None:
            raise ApiError(404, "document_not_found", "Không tìm thấy tài liệu.")
        busy = ApiError(409, "document_processing", "Tài liệu đang được xử lý; hãy đợi xử lý xong rồi xóa.")
        if document.status == DocStatus.PROCESSING:
            raise busy
        # Vectors first: if Qdrant is down nothing was changed and the request can simply be retried.
        deleted_chunks = store.delete_document(doc_id)
        if not db.delete_document(doc_id):  # a re-upload started in between
            raise busy
        try:
            _stored_path(settings, doc_id, Path(document.filename).suffix.lower()).unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            logger.warning("Could not remove the stored file of doc_id=%s: %s", doc_id, type(exc).__name__)
        logger.info("Deleted doc_id=%s (%d chunks)", doc_id, deleted_chunks)
        return DeleteOut(doc_id=doc_id, deleted_chunks=deleted_chunks)

    @app.post("/chat", response_model=ChatResponse,
              responses={code: {"model": ErrorResponse} for code in (404, 409, 422, 429, 502, 503, 504)})
    def chat(
        body: ChatRequest,
        db: Annotated[Database, Depends(get_db)],
        pipeline: Annotated[RagPipeline, Depends(get_pipeline)],
    ) -> ChatResponse:
        doc_ids = _require_documents(db, body.doc_ids)
        return _chat_response(pipeline.answer(body.question, doc_ids=doc_ids, top_k=body.top_k))

    @app.post("/documents/{doc_id}/summary", response_model=SummaryOut,
              summary="Summary of one ready document (1 LLM call, or map-reduce for a very long one)",
              responses={code: {"model": ErrorResponse} for code in (404, 409, 422, 429, 502, 503, 504)})
    def summarize(
        doc_id: DocIdPath,
        db: Annotated[Database, Depends(get_db)],
        store: Annotated[QdrantStore, Depends(get_store)],
        llm: Annotated[LLM, Depends(get_llm)],
        settings: Annotated[Settings, Depends(get_settings)],
        body: SummaryRequest | None = None,
    ) -> SummaryOut:
        request = body or SummaryRequest()
        _require_documents(db, [doc_id])
        result = summarize_document(db, store, llm, settings, doc_id, request.style, refresh=request.refresh)
        return SummaryOut(**result.payload, **_cost(result))

    @app.post("/documents/{doc_id}/quiz", response_model=QuizOut,
              summary="Multiple-choice quiz grounded in one ready document (1-2 LLM calls)",
              responses={code: {"model": ErrorResponse} for code in (404, 409, 422, 429, 502, 503, 504)})
    def quiz(
        doc_id: DocIdPath,
        db: Annotated[Database, Depends(get_db)],
        store: Annotated[QdrantStore, Depends(get_store)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
        llm: Annotated[LLM, Depends(get_llm)],
        settings: Annotated[Settings, Depends(get_settings)],
        body: QuizRequest | None = None,
    ) -> QuizOut:
        request = body or QuizRequest()
        _require_documents(db, [doc_id])
        result = generate_quiz(
            db, store, embedder, llm, settings, doc_id, num_questions=request.num_questions, topic=request.topic,
            page_from=request.page_from, page_to=request.page_to, verify=request.verify, refresh=request.refresh,
        )
        p = result.payload
        return QuizOut(
            questions=p["questions"], requested=p["requested"], returned=len(p["questions"]), dropped=p["dropped"],
            message=p["message"], warnings=p["warnings"], verify_requested=p["verify_requested"], verified=p["verified"],
            verifier_decisions=p["verifier_decisions"], **_cost(result),
        )

    return app


app = create_app()
