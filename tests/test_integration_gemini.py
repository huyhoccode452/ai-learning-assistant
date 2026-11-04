"""Integration tests that call the REAL Gemini API. They use quota, so they are skipped by default.

Run them explicitly:   pytest -m integration -v
Total cost: about 5 embedded texts (free tier allows ~100 per minute) plus a few count_tokens calls.
"""

import math

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.embeddings import EmbeddingError, ErrorCode, GeminiEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    try:
        settings = Settings()  # reads the real .env
    except ValidationError:
        pytest.skip("no GEMINI_API_KEY configured")
    if not settings.gemini_embedding_model or not settings.embedding_dim:
        pytest.skip("GEMINI_EMBEDDING_MODEL / EMBEDDING_DIM not set in .env")
    return settings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_real_embeddings_have_the_right_shape_order_and_task_asymmetry(real_settings: Settings) -> None:
    embedder = GeminiEmbedder(real_settings)
    texts = ["work_mem là bộ nhớ cho mỗi thao tác sắp xếp.", "Phân vùng bảng giúp xóa dữ liệu cũ.", "Công thức nấu phở bò."]
    batch = embedder.embed_documents(texts)

    assert len(batch) == 3
    assert all(len(v) == real_settings.embedding_dim for v in batch)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6) for v in batch)  # normalized

    single = embedder.embed_documents([texts[1]])[0]
    assert cosine(batch[1], single) > 0.999  # same text -> same vector: order preserved in the batch
    assert cosine(batch[0], single) < 0.95  # a different text is clearly different

    query_vec = embedder.embed_query(texts[1])
    assert cosine(single, query_vec) < 0.99  # RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY really differ


def test_real_api_over_limit_text_is_reported_instead_of_silently_truncated(real_settings: Settings) -> None:
    embedder = GeminiEmbedder(real_settings)
    too_long = " ".join(["chỉ mục"] * 1500)  # measured earlier: ~3000 tokens, limit is 2048
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_documents(["câu ngắn", too_long])
    assert info.value.code == ErrorCode.TEXT_TOO_LONG
    assert "index 1" in info.value.message and "2048" in info.value.message


def test_real_api_invalid_request_is_not_retried(real_settings: Settings) -> None:
    sleeps: list[float] = []
    bad = real_settings.model_copy(update={"gemini_embedding_model": "gemini-embedding-does-not-exist"})
    embedder = GeminiEmbedder(bad, sleep=sleeps.append)
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_documents(["xin chào"])
    assert info.value.code == ErrorCode.REQUEST_REJECTED
    assert sleeps == []  # failed immediately, never entered the retry loop
