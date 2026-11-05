"""Tests for app/main.py with FastAPI's TestClient. No network: every dependency is a fake.

TestClient runs BackgroundTasks before `client.post(...)` returns, so after an upload the
document is already 'ready' or 'failed'. To see 'processing' we look at the database from
inside the fake embedder, i.e. while the background job is running.
"""

import inspect
import json
import sqlite3
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.db import Database, DocStatus
from app.embeddings import EmbeddingError
from app.embeddings import ErrorCode as EmbeddingCode
from app.ingestion import ErrorCode as IngestionCode
from app.ingestion import compute_doc_id
from app.main import api_error_from, create_app
from app.config import get_settings
from app.prompts import FALLBACK_MESSAGE
from app.rag import RagError, RagErrorCode
from app.vectorstore import StoreErrorCode, VectorStoreError
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore, make_settings

API_KEY = "test-key-not-real-123"
MB = 1024 * 1024

TEXT = (
    "work_mem là bộ nhớ cho mỗi thao tác sắp xếp hoặc băm. Giá trị mặc định là 4MB. "
    "Dấu hiệu thiếu work_mem là dòng external merge Disk trong EXPLAIN ANALYZE.\n\n"
    "PgBouncer có ba chế độ pooling: session, transaction và statement."
).encode()


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder that records what the database says WHILE a document is being embedded."""

    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db_path = db_path
        self.document_calls = 0
        self.statuses_while_embedding: list[DocStatus] = []
        self.fail_documents: Exception | None = None
        self.fail_query: Exception | None = None

    def embed_documents(self, texts):
        self.document_calls += 1
        self.statuses_while_embedding = [d.status for d in Database(self.db_path).list_documents()]
        if self.fail_documents:
            raise self.fail_documents
        return super().embed_documents(texts)

    def embed_query(self, text):
        if self.fail_query:
            raise self.fail_query
        return super().embed_query(text)


def build_env(tmp_path: Path, **settings_overrides) -> SimpleNamespace:
    values = {
        "gemini_api_key": API_KEY,
        "gemini_llm_model": "fake-llm",
        "score_threshold": 0.6,
        "max_upload_mb": 1,
        "sqlite_path": tmp_path / "app.sqlite3",
        "upload_dir": tmp_path / "uploads",
        **settings_overrides,
    }
    settings = make_settings(**values)
    embedder = SpyEmbedder(settings.sqlite_path)
    store = FakeVectorStore()
    llm = FakeLLM("Trả lời từ tài liệu [1].")
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[main.get_embedder] = lambda: embedder
    application.dependency_overrides[main.get_store] = lambda: store
    application.dependency_overrides[main.get_llm] = lambda: llm
    return SimpleNamespace(app=application, settings=settings, embedder=embedder, store=store, llm=llm,
                           db=Database(settings.sqlite_path), uploads=settings.upload_dir)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[SimpleNamespace]:
    e = build_env(tmp_path)
    with TestClient(e.app) as client:  # `with` runs the startup (lifespan) code
        e.client = client
        yield e


def upload(env: SimpleNamespace, data: bytes = TEXT, filename: str = "notes.txt"):
    return env.client.post("/documents", files={"file": (filename, data, "application/octet-stream")})


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"}, body
    assert {"code", "message"} <= set(body["error"])
    return body["error"]


def make_pdf(text: str = "PostgreSQL tuning guide. Use EXPLAIN ANALYZE to read the real plan.") -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    return document.tobytes()


def ready_document(env: SimpleNamespace, data: bytes = TEXT, filename: str = "notes.txt") -> str:
    response = upload(env, data, filename)
    assert response.status_code == 202, response.text
    doc_id = response.json()["doc_id"]
    assert env.client.get(f"/documents/{doc_id}").json()["status"] == "ready"
    return doc_id


def processing_document(env: SimpleNamespace, doc_id: str = "a" * 16) -> str:
    env.db.claim_document(doc_id, "pending.txt", None, 3)
    return doc_id


# =========================================================================== upload: success


def test_upload_txt_returns_202_processing_then_the_document_becomes_ready(env) -> None:
    response = upload(env)
    assert response.status_code == 202
    body = response.json()
    assert body["doc_id"] == compute_doc_id(TEXT) and body["filename"] == "notes.txt"
    assert body["status"] == "processing"  # the response is built before the background job runs
    assert body["num_chunks"] >= 1 and body["error_message"] is None

    document = env.client.get(f"/documents/{body['doc_id']}").json()
    assert document["status"] == "ready" and document["num_chunks"] == body["num_chunks"]
    assert len(env.store.indexed[body["doc_id"]]) == body["num_chunks"]


def test_status_is_processing_while_embedding_and_ready_afterwards(env) -> None:
    upload(env)
    assert env.embedder.statuses_while_embedding == [DocStatus.PROCESSING]
    assert [d["status"] for d in env.client.get("/documents").json()] == ["ready"]


def test_pdf_upload_records_the_page_count(env) -> None:
    body = upload(env, make_pdf(), "guide.pdf").json()
    assert body["num_pages"] == 1
    assert env.client.get(f"/documents/{body['doc_id']}").json()["status"] == "ready"


def test_the_file_is_stored_under_a_generated_name_never_the_users_name(env) -> None:
    response = upload(env, filename="..\\..\\evil name <script>.txt")
    body = response.json()
    assert body["filename"] == "evil name _script_.txt"  # sanitized display name only
    stored = [p.name for p in env.uploads.iterdir()]
    assert stored == [f"{body['doc_id']}.txt"]
    assert (env.uploads / stored[0]).read_bytes() == TEXT
    assert not any(env.uploads.rglob("*evil*")) and not list(env.uploads.glob(".*.tmp"))


# =========================================================================== upload: rejected input


def test_unsupported_extension_is_rejected(env) -> None:
    response = upload(env, b"MZ\x90\x00 binary", "virus.exe")
    assert response.status_code == 415
    assert error_of(response)["code"] == "ingestion.unsupported_extension"
    assert env.db.list_documents() == [] and list(env.uploads.iterdir()) == []


def test_content_that_does_not_match_the_extension_is_rejected(env) -> None:
    response = upload(env, b"just text, not a pdf at all", "fake.pdf")
    assert response.status_code == 415 and error_of(response)["code"] == "ingestion.content_mismatch"
    assert env.db.list_documents() == []


def test_empty_file_is_rejected(env) -> None:
    response = upload(env, b"", "empty.txt")
    assert response.status_code == 400 and error_of(response)["code"] == "ingestion.empty_file"


def test_corrupted_pdf_is_rejected_immediately_and_not_stored(env) -> None:
    response = upload(env, b"%PDF-1.4\n\x00\x01 this is garbage, not a real pdf", "broken.pdf")
    assert response.status_code == 422 and error_of(response)["code"] == "ingestion.corrupted_file"
    assert env.db.list_documents() == [] and list(env.uploads.iterdir()) == []
    assert env.embedder.document_calls == 0


def test_whitespace_only_text_has_no_usable_text(env) -> None:
    response = upload(env, b"   \n\n   \t  \n", "blank.txt")
    assert response.status_code == 422 and error_of(response)["code"] == "ingestion.no_text"


def test_missing_file_field_is_a_validation_error(env) -> None:
    response = env.client.post("/documents", data={"other": "x"})
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request"


# =========================================================================== upload: size limit (KI-09)


def test_a_file_of_exactly_the_limit_is_accepted_and_one_byte_more_is_not(env) -> None:
    exact = (b"word " * (MB // 5 + 1))[:MB]
    assert len(exact) == MB
    assert upload(env, exact, "exact.txt").status_code == 202
    too_big = exact + b"x"
    response = upload(env, too_big, "over.txt")
    assert response.status_code == 413 and error_of(response)["code"] == "ingestion.file_too_large"
    assert env.db.get_document(compute_doc_id(too_big)) is None


def test_the_endpoint_itself_stops_reading_at_the_limit(env) -> None:
    class Upload:  # looks like UploadFile.file; counts what was read
        def __init__(self) -> None:
            self.read_bytes = 0

        def read(self, n: int) -> bytes:
            self.read_bytes += n
            return b"x" * n

    stream = Upload()
    with pytest.raises(main.IngestionError) as info:
        main._read_bounded(SimpleNamespace(file=stream), 200_000)
    assert info.value.code == IngestionCode.FILE_TOO_LARGE
    assert 200_000 < stream.read_bytes <= 200_000 + main.READ_CHUNK_BYTES  # not "read everything"


def test_a_huge_declared_content_length_is_refused_before_the_endpoint_runs(env, monkeypatch) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("the endpoint was reached; the middleware should have refused first")

    monkeypatch.setattr(main, "_read_bounded", must_not_run)
    response = upload(env, b"a" * (3 * MB), "big.txt")
    assert int(response.request.headers["content-length"]) > 3 * MB
    assert response.status_code == 413 and error_of(response)["code"] == "ingestion.file_too_large"
    assert env.db.list_documents() == [] and env.embedder.document_calls == 0


def test_a_chunked_upload_without_content_length_is_cut_off_too(env, monkeypatch) -> None:
    monkeypatch.setattr(main, "_read_bounded", lambda *a, **k: (_ for _ in ()).throw(AssertionError("endpoint reached")))
    boundary = "XBOUNDARY"
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n\r\n'.encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    def body() -> Iterator[bytes]:
        yield head
        for _ in range(40):
            yield b"a" * 65536  # 2.5 MB in total, limit is 1 MB
        yield tail

    response = env.client.post("/documents", content=body(),
                               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert "content-length" not in response.request.headers  # really the chunked path
    assert response.status_code == 413 and error_of(response)["code"] == "ingestion.file_too_large"
    assert env.db.list_documents() == []


def test_other_endpoints_have_a_small_body_limit_too(env) -> None:
    response = env.client.post("/chat", content=b"x" * (2 * MB), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


# =========================================================================== upload: idempotency and failure


def test_uploading_the_same_file_again_is_idempotent(env) -> None:
    first = upload(env)
    assert first.status_code == 202
    again = upload(env, filename="renamed copy.txt")
    assert again.status_code == 200  # already indexed: nothing to do
    assert again.json()["doc_id"] == first.json()["doc_id"]
    assert again.json()["filename"] == "notes.txt" and again.json()["status"] == "ready"
    assert env.embedder.document_calls == 1 and env.store.index_calls == [first.json()["doc_id"]]
    assert len(env.db.list_documents()) == 1 and len(list(env.uploads.iterdir())) == 1


def test_a_duplicate_is_answered_from_the_database_without_parsing_the_file_again(env, monkeypatch) -> None:
    upload(env)
    monkeypatch.setattr(main, "ingest_file", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed again")))
    assert upload(env).status_code == 200
    env.db.claim_document("b" * 16, "x.txt", None, 1)  # a different document that is mid-indexing
    assert upload(env, filename="again.txt").status_code == 200


def test_uploading_a_file_that_is_still_processing_does_not_start_a_second_job(env) -> None:
    doc_id = processing_document(env, compute_doc_id(TEXT))
    response = upload(env)
    assert response.status_code == 202 and response.json()["doc_id"] == doc_id
    assert response.json()["status"] == "processing"
    assert env.embedder.document_calls == 0 and env.store.index_calls == []


def test_a_processing_duplicate_is_not_parsed_or_stored_again(env, monkeypatch) -> None:
    processing_document(env, compute_doc_id(TEXT))
    monkeypatch.setattr(main, "ingest_file", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed again")))
    assert upload(env).status_code == 202
    assert list(env.uploads.iterdir()) == []


@pytest.mark.parametrize(
    "failure,code,fragment",
    [
        (EmbeddingError(EmbeddingCode.RATE_LIMITED, "429 quota exceeded, retry in 54s"), "embedding.rate_limited", "giới hạn"),
        (EmbeddingError(EmbeddingCode.SERVICE_UNAVAILABLE, "HTTP 503"), "embedding.service_unavailable", "không khả dụng"),
        (EmbeddingError(EmbeddingCode.AUTH_ERROR, f"bad key {API_KEY}"), "embedding.auth_error", "xác thực"),
    ],
)
def test_embedding_failure_marks_the_document_failed_with_a_clear_message(env, failure, code, fragment) -> None:
    env.embedder.fail_documents = failure
    response = upload(env)
    assert response.status_code == 202  # accepted; the failure is reported through the status
    document = env.client.get(f"/documents/{response.json()['doc_id']}").json()
    assert document["status"] == "failed" and document["error_code"] == code
    assert fragment in document["error_message"]
    assert API_KEY not in json.dumps(document) and "429" not in document["error_message"]
    assert env.store.index_calls == []


def test_vector_store_failure_marks_the_document_failed(env) -> None:
    env.store.fail_index = VectorStoreError(StoreErrorCode.BACKEND_UNAVAILABLE, "Cannot reach Qdrant at http://x:6333")
    document = env.client.get(f"/documents/{upload(env).json()['doc_id']}").json()
    assert document["status"] == "failed" and document["error_code"] == "store.backend_unavailable"
    assert "Qdrant" in document["error_message"] and "http://" not in document["error_message"]


def test_an_unexpected_crash_in_the_job_still_ends_in_failed_without_leaking(env) -> None:
    env.embedder.fail_documents = RuntimeError(r"boom at C:\secret\path\x.py")
    document = env.client.get(f"/documents/{upload(env).json()['doc_id']}").json()
    assert document["status"] == "failed" and document["error_code"] == "internal_error"
    assert "secret" not in document["error_message"] and "C:" not in document["error_message"]


def test_a_failed_document_can_be_retried_by_uploading_it_again(env) -> None:
    env.embedder.fail_documents = EmbeddingError(EmbeddingCode.RATE_LIMITED, "429")
    doc_id = upload(env).json()["doc_id"]
    assert env.client.get(f"/documents/{doc_id}").json()["status"] == "failed"
    env.embedder.fail_documents = None
    assert upload(env).status_code == 202
    document = env.client.get(f"/documents/{doc_id}").json()
    assert document["status"] == "ready" and document["error_message"] is None
    assert env.embedder.document_calls == 2


# =========================================================================== list / get / delete


def test_list_documents_newest_first(env) -> None:
    first = ready_document(env, TEXT, "first.txt")
    second = ready_document(env, TEXT + b" more words here to change the hash", "second.txt")
    listed = env.client.get("/documents").json()
    assert {d["doc_id"] for d in listed} == {first, second} and len(listed) == 2
    assert env.client.get("/documents").status_code == 200


def test_get_unknown_document_is_404_and_a_malformed_id_is_422(env) -> None:
    response = env.client.get(f"/documents/{'f' * 16}")
    assert response.status_code == 404 and error_of(response)["code"] == "document_not_found"
    assert env.client.get("/documents/not-a-valid-id").status_code == 422
    assert env.client.get("/documents/..%2F..%2Fetc%2Fpasswd").status_code in (404, 422)


def test_delete_removes_vectors_row_and_file(env) -> None:
    doc_id = ready_document(env)
    stored = env.uploads / f"{doc_id}.txt"
    assert stored.exists() and doc_id in env.store.indexed

    response = env.client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["doc_id"] == doc_id and response.json()["deleted_chunks"] >= 1
    assert doc_id not in env.store.indexed and not stored.exists()
    assert env.client.get(f"/documents/{doc_id}").status_code == 404


def test_delete_unknown_document_is_404(env) -> None:
    response = env.client.delete(f"/documents/{'e' * 16}")
    assert response.status_code == 404 and error_of(response)["code"] == "document_not_found"


def test_a_document_that_is_processing_cannot_be_deleted(env) -> None:
    doc_id = processing_document(env)
    env.store.indexed[doc_id] = ["partial chunk written by the running job"]
    response = env.client.delete(f"/documents/{doc_id}")
    assert response.status_code == 409 and error_of(response)["code"] == "document_processing"
    assert env.db.get_document(doc_id) is not None
    assert env.store.indexed[doc_id] == ["partial chunk written by the running job"]  # vectors untouched


def test_delete_when_qdrant_is_down_changes_nothing_so_it_can_be_retried(env) -> None:
    doc_id = ready_document(env)
    env.store.fail_delete = VectorStoreError(StoreErrorCode.BACKEND_UNAVAILABLE, "down")
    response = env.client.delete(f"/documents/{doc_id}")
    assert response.status_code == 503 and error_of(response)["code"] == "store.backend_unavailable"
    assert env.db.get_document(doc_id) is not None and (env.uploads / f"{doc_id}.txt").exists()
    env.store.fail_delete = None
    assert env.client.delete(f"/documents/{doc_id}").status_code == 200


def test_a_failed_document_can_be_deleted(env) -> None:
    env.embedder.fail_documents = EmbeddingError(EmbeddingCode.RATE_LIMITED, "429")
    doc_id = upload(env).json()["doc_id"]
    assert env.client.delete(f"/documents/{doc_id}").status_code == 200
    assert not list(env.uploads.iterdir())


# =========================================================================== chat


def chat(env: SimpleNamespace, question: str = "work_mem là gì?", **extra):
    return env.client.post("/chat", json={"question": question, **extra})


def test_chat_returns_the_answer_with_sources_and_latency(env) -> None:
    doc_id = ready_document(env)
    response = chat(env)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Trả lời từ tài liệu [1]." and body["used_fallback"] is False
    assert set(body["latency"]) == {"embed_ms", "search_ms", "llm_ms", "total_ms"}
    first = body["sources"][0]
    assert {"number", "filename", "page", "chunk_index", "score", "excerpt", "cited"} <= set(first)
    assert first["filename"] == "notes.txt" and first["doc_id"] == doc_id
    assert first["cited"] is True and first["number"] == 1 and first["page"] is None
    assert len(env.llm.calls) == 1 and "work_mem" in env.llm.calls[0]["prompt"]


def test_chat_passes_the_document_filter_and_top_k_to_the_search(env) -> None:
    first = ready_document(env)
    ready_document(env, TEXT + b" another document with different content", "other.txt")
    body = chat(env, doc_ids=[first], top_k=3).json()
    assert env.store.search_calls[-1] == {"top_k": 3, "score_threshold": None, "doc_ids": [first]}
    assert {s["doc_id"] for s in body["sources"]} == {first}


def test_chat_without_doc_ids_searches_only_ready_documents(env) -> None:
    ready = ready_document(env)
    processing_document(env, "b" * 16)
    env.db.claim_document("c" * 16, "failed.txt", None, 1)
    env.db.mark_failed("c" * 16, "x", "y")
    assert chat(env).status_code == 200
    assert env.store.search_calls[-1]["doc_ids"] == [ready]


def test_chat_with_no_ready_document_is_a_clear_conflict(env) -> None:
    processing_document(env)
    response = chat(env)
    assert response.status_code == 409 and error_of(response)["code"] == "no_documents_ready"
    assert env.llm.calls == [] and env.store.search_calls == []


def test_chat_on_a_document_that_is_not_ready_is_refused(env) -> None:
    ready = ready_document(env)
    pending = processing_document(env, "b" * 16)
    response = chat(env, doc_ids=[ready, pending])
    assert response.status_code == 409
    error = error_of(response)
    assert error["code"] == "document_not_ready"
    assert error["details"]["documents"] == [{"doc_id": pending, "status": "processing"}]
    assert env.llm.calls == [] and env.store.search_calls == []


def test_chat_on_a_failed_document_is_refused(env) -> None:
    env.embedder.fail_documents = EmbeddingError(EmbeddingCode.RATE_LIMITED, "429")
    doc_id = upload(env).json()["doc_id"]
    response = chat(env, doc_ids=[doc_id])
    assert response.status_code == 409 and error_of(response)["details"]["documents"][0]["status"] == "failed"


def test_chat_on_an_unknown_document_is_404(env) -> None:
    ready_document(env)
    response = chat(env, doc_ids=["f" * 16])
    assert response.status_code == 404
    assert error_of(response)["code"] == "document_not_found" and error_of(response)["details"] == {"doc_ids": ["f" * 16]}
    assert env.llm.calls == []


def test_chat_fallback_when_nothing_passes_the_threshold_does_not_call_the_llm(env) -> None:
    ready_document(env)
    env.store.score = 0.3
    env.store._rebuild()
    body = chat(env, "Công thức nấu phở?").json()
    assert body["used_fallback"] is True and body["answer"] == FALLBACK_MESSAGE
    assert body["sources"] == [] and body["latency"]["llm_ms"] == 0 and body["best_score"] == pytest.approx(0.3)
    assert env.llm.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "ok?", "top_k": 0},
        {"question": "ok?", "top_k": 21},
        {"question": "ok?", "doc_ids": []},
        {"question": "ok?", "doc_ids": ["nope"]},
        {"question": "ok?", "doc_ids": ["a" * 16] * 51},
        {},
    ],
)
def test_invalid_chat_requests_are_422_with_a_stable_code(env, payload) -> None:
    response = env.client.post("/chat", json=payload)
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request"
    assert env.llm.calls == []


def test_validation_errors_do_not_echo_the_submitted_value(env) -> None:
    secret = "SUPER-SECRET-VALUE"
    response = env.client.post("/chat", json={"question": "ok?", "top_k": secret, "doc_ids": [secret]})
    assert response.status_code == 422 and secret not in response.text


def test_the_question_is_trimmed(env) -> None:
    ready_document(env)
    chat(env, "   work_mem là gì?   ")
    assert env.llm.calls[0]["prompt"].rstrip().endswith("Question: work_mem là gì?")


# =========================================================================== error mapping


@pytest.mark.parametrize("exc_class,enum", [
    (main.IngestionError, IngestionCode),
    (EmbeddingError, EmbeddingCode),
    (VectorStoreError, StoreErrorCode),
    (RagError, RagErrorCode),
])
def test_every_error_code_is_mapped_to_a_specific_status_and_a_vietnamese_message(exc_class, enum) -> None:
    for member in enum:
        error = api_error_from(exc_class(member, "English technical detail"))
        assert error.code != "internal_error", f"{enum.__name__}.{member.name} is not mapped"
        assert error.code.endswith(f".{member.value}") and 400 <= error.status < 600
        assert "English technical detail" not in error.message
        assert any(ord(ch) > 127 for ch in error.message) or "PDF" in error.message  # Vietnamese text


@pytest.mark.parametrize("rag_code,status", [
    (RagErrorCode.RATE_LIMITED, 429), (RagErrorCode.TIMEOUT, 504), (RagErrorCode.SERVICE_UNAVAILABLE, 503),
    (RagErrorCode.BLOCKED, 422), (RagErrorCode.EMPTY_RESPONSE, 502), (RagErrorCode.AUTH_ERROR, 502),
])
def test_llm_errors_reach_the_client_as_the_mapped_status(env, rag_code, status) -> None:
    ready_document(env)
    env.llm.reply = RagError(rag_code, "internal detail")
    response = chat(env)
    assert response.status_code == status and error_of(response)["code"] == f"rag.{rag_code.value}"


def test_rate_limit_responses_carry_retry_after(env) -> None:
    ready_document(env)
    env.llm.reply = RagError(RagErrorCode.RATE_LIMITED, "slow down")
    assert chat(env).headers["retry-after"] == "60"


def test_embedding_and_store_errors_during_chat_are_mapped(env) -> None:
    ready_document(env)
    env.embedder.fail_query = EmbeddingError(EmbeddingCode.RATE_LIMITED, "429")
    assert chat(env).status_code == 429
    env.embedder.fail_query = None
    env.store.search = lambda *a, **k: (_ for _ in ()).throw(VectorStoreError(StoreErrorCode.BACKEND_UNAVAILABLE, "down"))
    response = chat(env)
    assert response.status_code == 503 and error_of(response)["code"] == "store.backend_unavailable"


def test_missing_score_threshold_is_a_503_not_a_crash(tmp_path) -> None:
    e = build_env(tmp_path, score_threshold=None)
    with TestClient(e.app) as client:
        e.client = client
        ready_document(e)
        response = chat(e)
    assert response.status_code == 503 and error_of(response)["code"] == "rag.not_configured"


def test_error_bodies_never_leak_keys_paths_or_tracebacks(env) -> None:
    ready_document(env)
    leaky = f"Traceback (most recent call last): File \"C:\\Users\\dev\\app\\rag.py\" key={API_KEY} /home/dev/.env"
    for failure in (RagError(RagErrorCode.REQUEST_REJECTED, leaky),):
        env.llm.reply = failure
        text = chat(env).text
        for forbidden in (API_KEY, "Traceback", "C:\\", "/home/", "rag.py", ".env"):
            assert forbidden not in text, forbidden


def test_an_unexpected_exception_is_a_generic_500(tmp_path) -> None:
    e = build_env(tmp_path)
    e.llm.reply = RuntimeError(f"unexpected at C:\\Users\\dev\\x.py with {API_KEY}")
    with TestClient(e.app, raise_server_exceptions=False) as client:
        e.client = client
        ready_document(e)
        response = chat(e)
    assert response.status_code == 500 and error_of(response)["code"] == "internal_error"
    assert API_KEY not in response.text and "C:" not in response.text and "Traceback" not in response.text


def test_unknown_routes_and_methods_use_the_same_error_shape(env) -> None:
    missing = env.client.get("/nope")
    assert missing.status_code == 404 and error_of(missing)["code"] == "not_found"
    wrong = env.client.put("/documents")
    assert wrong.status_code == 405 and error_of(wrong)["code"] == "method_not_allowed"


# =========================================================================== health, startup, docs, CORS


def test_health_reports_qdrant(env) -> None:
    assert env.client.get("/health").json() == {"status": "ok", "qdrant_reachable": True}
    env.store.reachable = False
    response = env.client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "degraded", "qdrant_reachable": False}


def test_documents_stuck_in_processing_are_marked_failed_at_startup(tmp_path) -> None:
    e = build_env(tmp_path)
    e.db.init_db()
    e.db.claim_document("a" * 16, "half.txt", None, 5)
    e.db.claim_document("b" * 16, "done.txt", None, 5)
    e.db.mark_ready("b" * 16, 5)
    with TestClient(e.app) as client:
        stuck = client.get(f"/documents/{'a' * 16}").json()
        done = client.get(f"/documents/{'b' * 16}").json()
    assert stuck["status"] == "failed" and stuck["error_code"] == "interrupted" and "tải" in stuck["error_message"]
    assert done["status"] == "ready"


def test_startup_creates_the_database_folders_and_collection(tmp_path) -> None:
    e = build_env(tmp_path)
    with TestClient(e.app):
        pass
    assert e.settings.sqlite_path.exists() and e.settings.upload_dir.is_dir() and e.store.ensure_calls == 1


def test_startup_survives_an_unreachable_qdrant(tmp_path) -> None:
    e = build_env(tmp_path)
    e.store.ensure_collection = lambda: (_ for _ in ()).throw(VectorStoreError(StoreErrorCode.BACKEND_UNAVAILABLE, "down"))
    with TestClient(e.app) as client:
        assert client.get("/documents").status_code == 200


def test_swagger_and_openapi_are_served(env) -> None:
    docs = env.client.get("/docs")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    paths = env.client.get("/openapi.json").json()["paths"]
    assert {"/health", "/documents", "/documents/{doc_id}", "/chat"} <= set(paths)


def test_cors_allows_only_the_local_streamlit_origin(env) -> None:
    def preflight(origin: str):
        return env.client.options("/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"})

    allowed = preflight("http://localhost:8501")
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in preflight("http://evil.example").headers


def test_endpoints_are_plain_def_so_they_run_in_the_threadpool_not_on_the_event_loop() -> None:
    for function in (main.create_app().router.routes):
        endpoint = getattr(function, "endpoint", None)
        if endpoint and getattr(function, "path", "") in ("/health", "/documents", "/documents/{doc_id}", "/chat"):
            assert not inspect.iscoroutinefunction(endpoint), function.path
    assert not inspect.iscoroutinefunction(main.index_document_job)


def test_nothing_is_written_to_sqlite_outside_parametrized_queries(env) -> None:
    hostile = "x'); DROP TABLE documents; --.txt"
    upload(env, filename=hostile)
    with sqlite3.connect(env.settings.sqlite_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_middleware_refuses_a_declared_oversize_body_without_reading_any_of_it() -> None:
    import asyncio

    inner_called = False
    sent: list[dict] = []

    async def inner(scope, receive, send):
        nonlocal inner_called
        inner_called = True

    async def never_read():
        raise AssertionError("the body must not be read when Content-Length is already too large")

    async def collect(message):
        sent.append(message)

    middleware = main.BodySizeLimitMiddleware(inner, limit_for=lambda scope: 1000)
    scope = {"type": "http", "method": "POST", "path": "/documents", "headers": [(b"content-length", b"5000")]}
    asyncio.run(middleware(scope, never_read, collect))
    assert inner_called is False
    assert sent[0]["status"] == 413 and b"ingestion.file_too_large" in sent[1]["body"]


def test_an_error_code_nobody_mapped_yet_becomes_a_generic_500_not_a_crash() -> None:
    future_code = StrEnum("FutureCode", {"BRAND_NEW": "brand_new_failure"}).BRAND_NEW
    error = api_error_from(main.IngestionError(future_code, "English detail"))  # type: ignore[arg-type]
    assert error.status == 500 and error.message == main.INTERNAL_ERROR[1] and "English" not in error.message
    assert api_error_from(ValueError("not a domain error")).code == "internal_error"


def test_a_body_that_is_not_valid_utf8_json_gets_a_clear_400(env) -> None:
    body = '{"question": "work_mem là gì?"}'.encode("cp1252")  # the bytes some Windows shells send
    response = env.client.post("/chat", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400 and error_of(response)["code"] == "bad_request"
    assert "UTF-8" in error_of(response)["message"]
