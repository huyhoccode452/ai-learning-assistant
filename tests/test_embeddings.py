"""Unit tests for app/embeddings.py. No network: the Gemini client is a fake."""

import logging
import math

import httpx
import pytest
from google.genai import errors as genai_errors

from app.embeddings import (
    MAX_BATCH_SIZE,
    TASK_DOCUMENT,
    TASK_QUERY,
    EmbeddingError,
    ErrorCode,
    GeminiEmbedder,
)
from tests.fakes import FakeGenaiClient, FakeModels, make_settings


def api_error(code: int, status: str, message: str = "boom", retry_delay: str | None = None):
    body: dict = {"code": code, "message": message, "status": status}
    if retry_delay:
        body["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return genai_errors.APIError(code, {"error": body})


def build(models: FakeModels, **kwargs):
    """Embedder wired to a fake client, with a recording sleep and max-jitter (deterministic)."""
    sleeps: list[float] = []
    embedder = GeminiEmbedder(
        make_settings(embedding_dim=models.dim),
        client=FakeGenaiClient(models),
        sleep=sleeps.append,
        jitter=lambda low, high: high,  # always the upper bound => delay == full backoff
        **kwargs,
    )
    return embedder, sleeps


def texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def assert_error(code: ErrorCode, fn, *args) -> EmbeddingError:
    with pytest.raises(EmbeddingError) as info:
        fn(*args)
    assert info.value.code == code, f"expected {code}, got {info.value.code}: {info.value.message}"
    assert info.value.message
    return info.value


# --------------------------------------------------------------------------- batching / order


def test_order_is_preserved_across_batches() -> None:
    models = FakeModels(dim=300)
    embedder, _ = build(models)
    vectors = embedder.embed_documents(texts(250))
    assert len(vectors) == 250
    assert [v.index(max(v)) for v in vectors] == list(range(250))  # vector i belongs to text i


def test_requests_are_split_into_batches_of_at_most_100() -> None:
    models = FakeModels(dim=300)
    embedder, _ = build(models)
    embedder.embed_documents(texts(250))
    assert [len(c["contents"]) for c in models.embed_calls] == [100, 100, 50]
    assert MAX_BATCH_SIZE == 100


def test_custom_batch_size_and_its_limits() -> None:
    models = FakeModels(dim=30)
    embedder, _ = build(models, batch_size=7)
    embedder.embed_documents(texts(20))
    assert [len(c["contents"]) for c in models.embed_calls] == [7, 7, 6]
    for bad in (0, 101):
        with pytest.raises(ValueError):
            build(FakeModels(), batch_size=bad)


def test_documents_and_queries_use_different_task_types_and_the_configured_dimension() -> None:
    models = FakeModels(dim=8)
    embedder, _ = build(models)
    embedder.embed_documents(["t1", "t2"])
    embedder.embed_query("t3")
    doc_call, query_call = models.embed_calls
    assert doc_call["task_type"] == TASK_DOCUMENT == "RETRIEVAL_DOCUMENT"
    assert query_call["task_type"] == TASK_QUERY == "RETRIEVAL_QUERY"
    assert doc_call["dim"] == query_call["dim"] == 8
    assert doc_call["model"] == "fake-embedding-model"


def test_empty_input_list_makes_no_call() -> None:
    models = FakeModels()
    embedder, _ = build(models)
    assert embedder.embed_documents([]) == []
    assert models.embed_calls == []


def test_vectors_are_l2_normalized() -> None:
    # The fake returns vectors of length 5 (like truncated Matryoshka vectors, not unit length).
    embedder, _ = build(FakeModels(dim=8))
    vector = embedder.embed_query("t2")
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- response checks


def test_wrong_dimension_from_the_api_is_detected() -> None:
    models = FakeModels(dim=10)  # the API "returns" 10 dims...
    embedder = GeminiEmbedder(make_settings(embedding_dim=8), client=FakeGenaiClient(models))  # ...we expect 8
    err = assert_error(ErrorCode.DIMENSION_MISMATCH, embedder.embed_documents, ["t1"])
    assert "10" in err.message and "8" in err.message and "re-index" in err.message


def test_wrong_number_of_embeddings_is_detected() -> None:
    embedder, _ = build(FakeModels(dim=8, drop_last=True))
    err = assert_error(ErrorCode.BAD_RESPONSE, embedder.embed_documents, ["t1", "t2"])
    assert "2 texts" in err.message and "1 embeddings" in err.message


def test_missing_configuration_is_reported() -> None:
    with pytest.raises(EmbeddingError) as info:
        GeminiEmbedder(make_settings(gemini_embedding_model=None), client=FakeGenaiClient(FakeModels()))
    assert info.value.code == ErrorCode.NOT_CONFIGURED


# --------------------------------------------------------------------------- retry


def test_retries_on_429_then_succeeds_with_exponential_backoff() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED"), api_error(429, "RESOURCE_EXHAUSTED")])
    embedder, sleeps = build(models)
    vectors = embedder.embed_documents(["t1"])
    assert len(vectors) == 1
    assert len(models.embed_calls) == 3  # 2 failures + 1 success
    assert sleeps == [2.0, 4.0]  # base 2s, doubling


def test_backoff_is_capped_by_max_delay() -> None:
    errors = [api_error(503, "UNAVAILABLE")] * 4
    models = FakeModels(script=list(errors))
    embedder, sleeps = build(models, max_retries=4, base_delay=10.0, max_delay=25.0)
    embedder.embed_documents(["t1"])
    assert sleeps == [10.0, 20.0, 25.0, 25.0]


def test_jitter_is_applied_between_half_and_full_backoff() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED")])
    sleeps: list[float] = []
    embedder = GeminiEmbedder(
        make_settings(embedding_dim=8), client=FakeGenaiClient(models), sleep=sleeps.append,
        jitter=lambda low, high: low,  # lower bound
    )
    embedder.embed_documents(["t1"])
    assert sleeps == [1.0]  # backoff 2s -> "equal jitter" lower bound is 1s


def test_server_retry_delay_is_obeyed() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED", retry_delay="54s")])
    embedder, sleeps = build(models)
    embedder.embed_documents(["t1"])
    assert sleeps[0] >= 54.0  # backoff alone would have been 2s


def test_server_asking_for_a_very_long_wait_fails_fast() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED", retry_delay="3600s")])
    embedder, sleeps = build(models)
    err = assert_error(ErrorCode.RATE_LIMITED, embedder.embed_documents, ["t1"])
    assert sleeps == [] and len(models.embed_calls) == 1
    assert "3600" in err.message


def test_gives_up_after_max_retries_with_a_clear_error() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED")] * 10)
    embedder, sleeps = build(models, max_retries=3)
    err = assert_error(ErrorCode.RATE_LIMITED, embedder.embed_documents, ["t1"])
    assert len(models.embed_calls) == 4  # first try + 3 retries
    assert len(sleeps) == 3
    assert "3 retries" in err.message and "429" in err.message
    assert isinstance(err.__cause__, genai_errors.APIError)


def test_max_retries_zero_means_no_retry() -> None:
    models = FakeModels(script=[api_error(503, "UNAVAILABLE")])
    embedder, sleeps = build(models, max_retries=0)
    assert_error(ErrorCode.SERVICE_UNAVAILABLE, embedder.embed_documents, ["t1"])
    assert len(models.embed_calls) == 1 and sleeps == []


def test_max_retries_defaults_to_the_setting() -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED")] * 10)
    sleeps: list[float] = []
    embedder = GeminiEmbedder(
        make_settings(embedding_dim=8, embed_max_retries=1), client=FakeGenaiClient(models), sleep=sleeps.append
    )
    assert_error(ErrorCode.RATE_LIMITED, embedder.embed_documents, ["t1"])
    assert len(models.embed_calls) == 2


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_transient_5xx_is_retried(code: int) -> None:
    models = FakeModels(script=[api_error(code, "UNAVAILABLE")])
    embedder, sleeps = build(models)
    embedder.embed_documents(["t1"])
    assert len(models.embed_calls) == 2 and len(sleeps) == 1


def test_network_errors_are_retried() -> None:
    models = FakeModels(script=[httpx.ReadTimeout("slow"), httpx.ConnectError("reset")])
    embedder, sleeps = build(models)
    embedder.embed_documents(["t1"])
    assert len(models.embed_calls) == 3 and len(sleeps) == 2


@pytest.mark.parametrize(
    ("code", "status", "expected"),
    [
        (400, "INVALID_ARGUMENT", ErrorCode.REQUEST_REJECTED),
        (404, "NOT_FOUND", ErrorCode.REQUEST_REJECTED),
        (401, "UNAUTHENTICATED", ErrorCode.AUTH_ERROR),
        (403, "PERMISSION_DENIED", ErrorCode.AUTH_ERROR),
    ],
)
def test_client_errors_are_not_retried(code: int, status: str, expected: ErrorCode) -> None:
    models = FakeModels(script=[api_error(code, status, message="bad thing")])
    embedder, sleeps = build(models)
    err = assert_error(expected, embedder.embed_documents, ["t1"])
    assert len(models.embed_calls) == 1  # exactly one attempt
    assert sleeps == []
    assert str(code) in err.message and "bad thing" in err.message


def test_api_key_is_redacted_from_error_messages() -> None:
    key = make_settings().gemini_api_key.get_secret_value()
    models = FakeModels(script=[api_error(400, "INVALID_ARGUMENT", message=f"key {key} is invalid")])
    embedder, _ = build(models)
    err = assert_error(ErrorCode.REQUEST_REJECTED, embedder.embed_documents, ["t1"])
    assert key not in err.message


# --------------------------------------------------------------------------- empty / too long


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_or_whitespace_text_is_rejected_before_any_api_call(bad: str) -> None:
    models = FakeModels()
    embedder, _ = build(models)
    err = assert_error(ErrorCode.EMPTY_TEXT, embedder.embed_documents, ["t1", bad, "t2"])
    assert "index 1" in err.message
    assert models.embed_calls == []  # nothing was sent, not even the valid texts
    assert_error(ErrorCode.EMPTY_TEXT, embedder.embed_query, bad)


def test_all_empty_indexes_are_reported() -> None:
    embedder, _ = build(FakeModels())
    err = assert_error(ErrorCode.EMPTY_TEXT, embedder.embed_documents, ["", "t1", " ", ""])
    assert "3 text(s)" in err.message and "0, 2, 3" in err.message


def test_text_over_the_token_limit_is_reported_not_truncated() -> None:
    long_text = "x" * 4000
    models = FakeModels(token_counts={4000: 3002})
    embedder, _ = build(models)
    err = assert_error(ErrorCode.TEXT_TOO_LONG, embedder.embed_documents, ["t1", long_text, "t2"])
    assert "index 1" in err.message and "3002" in err.message and "2048" in err.message
    assert models.embed_calls == []  # rejected up front; nothing embedded


def test_long_text_within_the_token_limit_is_accepted() -> None:
    text = "t5" + "x" * 1500  # over the char threshold, but only ~1500 tokens
    models = FakeModels(token_counts={len(text): 1500})
    embedder, _ = build(models)
    assert len(embedder.embed_documents([text])) == 1
    assert models.count_calls == [text]


def test_short_texts_are_never_token_counted() -> None:
    models = FakeModels(dim=300)
    embedder, _ = build(models)
    embedder.embed_documents(texts(250))
    embedder.embed_query("t1")
    assert models.count_calls == []  # chunk-sized texts cost no extra API calls


def test_too_long_query_is_rejected() -> None:
    models = FakeModels(token_counts={5000: 4000})
    embedder, _ = build(models)
    assert_error(ErrorCode.TEXT_TOO_LONG, embedder.embed_query, "y" * 5000)


# --------------------------------------------------------------------------- logging


def test_logs_batch_size_and_latency_but_never_the_key_or_the_texts(caplog: pytest.LogCaptureFixture) -> None:
    models = FakeModels(script=[api_error(429, "RESOURCE_EXHAUSTED")])
    embedder, _ = build(models)
    secret_text = "t7"
    key = make_settings().gemini_api_key.get_secret_value()
    with caplog.at_level(logging.DEBUG, logger="app.embeddings"):
        embedder.embed_documents([secret_text, "t8"])
    log = caplog.text
    assert "size=2" in log and "latency=" in log and "retry 1/3" in log
    assert key not in log
    assert f"'{secret_text}'" not in log and f"[{secret_text}" not in log
