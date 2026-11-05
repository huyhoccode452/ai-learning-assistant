"""Tests for app/vectorstore.py against a real local Qdrant (docker compose up -d).

Every test uses its own collection with a random name that is always deleted afterwards.
If no Qdrant server is reachable the whole module is skipped (not failed).
"""

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.ingestion import Chunk, make_chunk_id
from app.vectorstore import (
    DocumentInfo,
    QdrantStore,
    StoreErrorCode,
    VectorStoreError,
)
from tests.fakes import FakeEmbedder, make_settings

pytestmark = pytest.mark.qdrant

DIM = 64


@pytest.fixture(scope="module")
def settings():
    return make_settings(embedding_dim=DIM)


@pytest.fixture(scope="module", autouse=True)
def _require_qdrant(settings) -> None:
    try:
        QdrantClient(url=settings.qdrant_url, timeout=3).get_collections()
    except Exception as exc:  # any failure to connect means: skip, do not fail
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}: {type(exc).__name__}")


@pytest.fixture
def raw() -> Iterator[QdrantClient]:
    from tests.fakes import make_settings as _ms

    client = QdrantClient(url=_ms().qdrant_url, timeout=10)
    yield client
    client.close()


@pytest.fixture
def store(settings, raw: QdrantClient) -> Iterator[QdrantStore]:
    """A store bound to a fresh random collection name. NOT created yet; always cleaned up."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    s = QdrantStore(settings, collection=name)
    try:
        yield s
    finally:
        if raw.collection_exists(name):
            raw.delete_collection(name)
        s.close()


@pytest.fixture
def ready(store: QdrantStore) -> QdrantStore:
    store.ensure_collection()
    return store


embedder = FakeEmbedder(DIM)


def make_chunks(doc_id: str, texts: list[str], filename: str | None = None,
                pages: list[int | None] | None = None) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=make_chunk_id(doc_id, i),
            doc_id=doc_id,
            filename=filename or f"{doc_id}.pdf",
            page=pages[i] if pages else i + 1,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def index(store: QdrantStore, doc_id: str, texts: list[str], **kwargs) -> list[Chunk]:
    chunks = make_chunks(doc_id, texts, **kwargs)
    store.index_document(doc_id, chunks, embedder.embed_documents(texts))
    return chunks


def search(store: QdrantStore, query: str, **kwargs):
    return store.search(embedder.embed_query(query), **{"top_k": 10, **kwargs})


def total_points(raw: QdrantClient, store: QdrantStore) -> int:
    return raw.count(store.collection, exact=True).count


DOC_A = "aaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbb"
TEXTS_A = [
    "work_mem là bộ nhớ cho mỗi thao tác sắp xếp",
    "autovacuum dọn các dòng chết trong bảng",
    "chỉ mục B-tree phù hợp cho so sánh bằng",
]
TEXTS_B = [
    "phân vùng bảng giúp xóa dữ liệu cũ nhanh hơn",
    "replication slot bị bỏ quên làm đầy ổ đĩa",
]


# --------------------------------------------------------------------------- collection


def test_ensure_collection_creates_it_with_cosine_dim_and_a_doc_id_index(store, raw) -> None:
    assert store.ensure_collection() is True
    info = raw.get_collection(store.collection)
    assert info.config.params.vectors.size == DIM
    assert info.config.params.vectors.distance == Distance.COSINE
    assert str(info.payload_schema["doc_id"].data_type).endswith("keyword")


def test_ensure_collection_is_idempotent(store) -> None:
    assert store.ensure_collection() is True
    assert store.ensure_collection() is False  # second call: exists and matches


def test_ensure_collection_adds_a_missing_doc_id_index(store, raw) -> None:
    raw.create_collection(store.collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    assert "doc_id" not in raw.get_collection(store.collection).payload_schema
    store.ensure_collection()
    assert "doc_id" in raw.get_collection(store.collection).payload_schema


def test_dimension_mismatch_raises_and_never_touches_the_collection(store, raw) -> None:
    raw.create_collection(store.collection, vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
    with pytest.raises(VectorStoreError) as info:
        store.ensure_collection()
    assert info.value.code == StoreErrorCode.COLLECTION_MISMATCH
    assert f"size={DIM * 2}" in info.value.message and f"size={DIM}" in info.value.message
    assert "re-index" in info.value.message
    still = raw.get_collection(store.collection)  # not deleted, not altered
    assert still.config.params.vectors.size == DIM * 2


def test_distance_mismatch_raises(store, raw) -> None:
    raw.create_collection(store.collection, vectors_config=VectorParams(size=DIM, distance=Distance.DOT))
    with pytest.raises(VectorStoreError) as info:
        store.ensure_collection()
    assert info.value.code == StoreErrorCode.COLLECTION_MISMATCH
    assert "Dot" in info.value.message and "Cosine" in info.value.message
    assert raw.collection_exists(store.collection)


def test_unreachable_qdrant_gives_a_clear_error(settings) -> None:
    dead_url = "http://127.0.0.1:1"  # nothing listens on port 1 (127.0.0.1 avoids slow IPv6 fallback)
    dead = QdrantStore(make_settings(embedding_dim=DIM, qdrant_url=dead_url),
                       client=QdrantClient(url=dead_url, timeout=2, check_compatibility=False))
    with pytest.raises(VectorStoreError) as info:
        dead.ensure_collection()
    assert info.value.code == StoreErrorCode.BACKEND_UNAVAILABLE
    assert "docker compose up" in info.value.message


def test_using_a_missing_collection_explains_how_to_fix_it(store) -> None:
    with pytest.raises(VectorStoreError) as info:
        search(store, "work_mem")
    assert info.value.code == StoreErrorCode.BACKEND_ERROR
    assert "ensure_collection" in info.value.message


# --------------------------------------------------------------------------- indexing


def test_index_then_search_returns_the_right_chunk_with_payload(ready) -> None:
    index(ready, DOC_A, TEXTS_A, filename="guide.pdf", pages=[5, None, 7])
    hits = search(ready, "work_mem sắp xếp", top_k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.text == TEXTS_A[0]
    assert (hit.doc_id, hit.filename, hit.page, hit.chunk_index) == (DOC_A, "guide.pdf", 5, 0)
    assert 0.0 < hit.score <= 1.0 + 1e-6
    by_text = {h.text: h for h in search(ready, "autovacuum chỉ mục work_mem")}
    assert by_text[TEXTS_A[1]].page is None  # page=None survives the round trip


def test_point_id_is_the_chunk_id(ready, raw) -> None:
    chunks = index(ready, DOC_A, TEXTS_A)
    stored = {str(p.id) for p in raw.scroll(ready.collection, limit=10)[0]}
    assert stored == {c.chunk_id for c in chunks}


def test_indexing_the_same_document_twice_keeps_the_same_point_count(ready, raw) -> None:
    index(ready, DOC_A, TEXTS_A)
    first = total_points(raw, ready)
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_A, TEXTS_A)
    assert first == len(TEXTS_A)
    assert total_points(raw, ready) == len(TEXTS_A)


def test_reindexing_with_fewer_chunks_removes_the_stale_ones(ready, raw) -> None:
    index(ready, DOC_A, TEXTS_A)  # 3 chunks
    index(ready, DOC_A, TEXTS_A[:1])  # same document, now only 1 chunk
    assert total_points(raw, ready) == 1
    assert ready.count_chunks(DOC_A) == 1
    leftovers = [h for h in search(ready, "autovacuum chỉ mục B-tree") if h.chunk_index > 0]
    assert leftovers == []  # chunk_index 1 and 2 are really gone, not just hidden


def test_reindexing_with_different_text_replaces_the_content(ready) -> None:
    index(ready, DOC_A, ["văn bản cũ về replication"])
    index(ready, DOC_A, ["văn bản mới về autovacuum"])
    texts = [h.text for h in search(ready, "văn bản replication autovacuum")]
    assert texts == ["văn bản mới về autovacuum"]


def test_indexing_does_not_touch_other_documents(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_B, TEXTS_B)
    index(ready, DOC_A, TEXTS_A[:1])
    assert ready.count_chunks(DOC_B) == len(TEXTS_B)


def test_large_documents_are_upserted_in_batches(settings, raw) -> None:
    name = f"test_{uuid.uuid4().hex[:12]}"
    store = QdrantStore(settings, collection=name, upsert_batch_size=7)
    try:
        store.ensure_collection()
        texts = [f"đoạn số {i} nói về chỉ mục" for i in range(20)]
        assert store.index_document(DOC_A, make_chunks(DOC_A, texts), embedder.embed_documents(texts)) == 20
        assert total_points(raw, store) == 20
    finally:
        raw.delete_collection(name)


def test_index_document_validates_its_input(ready) -> None:
    good = make_chunks(DOC_A, TEXTS_A)
    vectors = embedder.embed_documents(TEXTS_A)
    cases = {
        "empty": ([], []),
        "count mismatch": (good, vectors[:2]),
        "wrong dimension": (good, [v[:10] for v in vectors]),
        "foreign chunk": (good + make_chunks(DOC_B, ["văn bản khác hẳn"]), vectors + embedder.embed_documents(["văn bản khác hẳn"])),
    }
    for label, (chunks, vecs) in cases.items():
        with pytest.raises(VectorStoreError) as info:
            ready.index_document(DOC_A, chunks, vecs)
        assert info.value.code == StoreErrorCode.INVALID_INPUT, label
    assert ready.count_chunks(DOC_A) == 0  # nothing was written by the failed calls


def test_invalid_input_does_not_delete_existing_data(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    with pytest.raises(VectorStoreError):
        ready.index_document(DOC_A, make_chunks(DOC_A, TEXTS_A), [[0.0] * 3] * 3)  # wrong dim
    assert ready.count_chunks(DOC_A) == 3  # validation happens BEFORE the delete


# --------------------------------------------------------------------------- search


def test_doc_id_filter_returns_only_that_document(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_B, TEXTS_B)
    query = "bảng dữ liệu autovacuum phân vùng replication work_mem"
    assert {h.doc_id for h in search(ready, query)} == {DOC_A, DOC_B}  # no filter: both
    only_a = search(ready, query, doc_ids=[DOC_A])
    assert only_a and {h.doc_id for h in only_a} == {DOC_A}
    only_b = search(ready, query, doc_ids=[DOC_B])
    assert only_b and {h.doc_id for h in only_b} == {DOC_B}
    both = search(ready, query, doc_ids=[DOC_A, DOC_B])
    assert {h.doc_id for h in both} == {DOC_A, DOC_B}


def test_empty_doc_id_list_means_no_documents_not_all_documents(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    assert search(ready, "work_mem", doc_ids=[]) == []


def test_unknown_doc_id_returns_nothing(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    assert search(ready, "work_mem", doc_ids=["does-not-exist"]) == []


def test_results_are_sorted_best_first_and_top_k_is_respected(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_B, TEXTS_B)
    hits = search(ready, "work_mem sắp xếp bộ nhớ thao tác", top_k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].text == TEXTS_A[0]
    assert len(search(ready, "work_mem", top_k=1)) == 1


def test_score_threshold_filters_low_scoring_chunks(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_B, TEXTS_B)
    query = "work_mem là bộ nhớ cho mỗi thao tác sắp xếp"  # identical to TEXTS_A[0]
    everything = search(ready, query)
    scores = [h.score for h in everything]
    assert scores[0] == pytest.approx(1.0, abs=1e-5) and scores[-1] < 0.5  # a real spread
    cutoff = (scores[0] + scores[1]) / 2
    kept = search(ready, query, score_threshold=cutoff)
    assert [h.text for h in kept] == [TEXTS_A[0]]
    assert all(h.score >= cutoff for h in kept)
    assert search(ready, query, score_threshold=1.01) == []  # nothing can pass


def test_search_validates_its_arguments(ready) -> None:
    vector = embedder.embed_query("work_mem")
    with pytest.raises(VectorStoreError) as info:
        ready.search(vector, top_k=0)
    assert info.value.code == StoreErrorCode.INVALID_INPUT
    with pytest.raises(VectorStoreError) as info:
        ready.search(vector[:10], top_k=3)
    assert info.value.code == StoreErrorCode.INVALID_INPUT and "64" in info.value.message


# --------------------------------------------------------------------------- delete / list / exists


def test_delete_document_removes_all_its_points_and_only_those(ready, raw) -> None:
    index(ready, DOC_A, TEXTS_A)
    index(ready, DOC_B, TEXTS_B)
    assert ready.delete_document(DOC_A) == len(TEXTS_A)
    assert ready.count_chunks(DOC_A) == 0
    assert ready.count_chunks(DOC_B) == len(TEXTS_B)
    assert total_points(raw, ready) == len(TEXTS_B)


def test_deleting_an_unknown_document_is_a_harmless_zero(ready) -> None:
    index(ready, DOC_A, TEXTS_A)
    assert ready.delete_document("nope") == 0
    assert ready.count_chunks(DOC_A) == 3


def test_is_indexed(ready) -> None:
    assert ready.is_indexed(DOC_A) is False
    index(ready, DOC_A, TEXTS_A)
    assert ready.is_indexed(DOC_A) is True
    assert ready.is_indexed(DOC_B) is False
    ready.delete_document(DOC_A)
    assert ready.is_indexed(DOC_A) is False


def test_list_documents_reports_filenames_and_exact_chunk_counts(ready) -> None:
    assert ready.list_documents() == []
    index(ready, DOC_B, TEXTS_B, filename="replication.pdf")
    index(ready, DOC_A, TEXTS_A, filename="autovacuum.pdf")
    assert ready.list_documents() == [
        DocumentInfo(DOC_A, "autovacuum.pdf", 3),
        DocumentInfo(DOC_B, "replication.pdf", 2),
    ]
    index(ready, DOC_A, TEXTS_A[:2], filename="autovacuum.pdf")  # re-index shrinks the count
    assert [d.chunk_count for d in ready.list_documents()] == [2, 2]


def test_is_reachable_is_true_for_a_live_server_and_false_for_a_dead_one(store: QdrantStore) -> None:
    assert store.is_reachable() is True
    dead = QdrantStore(make_settings(embedding_dim=DIM, qdrant_url="http://127.0.0.1:9"), collection="unused")
    try:
        assert dead.is_reachable() is False  # must return False, never raise
    finally:
        dead.close()
