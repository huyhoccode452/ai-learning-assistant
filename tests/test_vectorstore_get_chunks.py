"""QdrantStore.get_chunks with a fake client: pagination, ordering and error mapping without a server."""

from types import SimpleNamespace

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from app.vectorstore import QdrantStore, StoreErrorCode, VectorStoreError
from tests.fakes import make_settings


def point(index: int, doc_id: str = "d" * 16):
    return SimpleNamespace(payload={"doc_id": doc_id, "filename": "a.pdf", "page": index // 2 + 1, "chunk_index": index, "text": f"t{index}"})


class FakeScrollClient:
    """Serves the points in the given pages; `offset` is the number of the next page, None after the last."""

    def __init__(self, pages: list[list]) -> None:
        self.pages, self.calls = pages, []

    def scroll(self, collection, *, scroll_filter, limit, offset, with_payload, with_vectors):
        self.calls.append({"collection": collection, "filter": scroll_filter, "limit": limit, "offset": offset,
                           "with_payload": with_payload, "with_vectors": with_vectors})
        page = 0 if offset is None else offset
        return self.pages[page], (page + 1 if page + 1 < len(self.pages) else None)


def store_with(client) -> QdrantStore:
    return QdrantStore(make_settings(embedding_dim=8), collection="c", client=client)


def test_every_page_is_read_and_the_order_is_restored() -> None:
    client = FakeScrollClient([[point(4), point(1)], [point(0), point(3)], [point(2)]])  # Qdrant promises no order
    chunks = store_with(client).get_chunks("d" * 16)
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4] and [c.text for c in chunks] == ["t0", "t1", "t2", "t3", "t4"]
    assert [call["offset"] for call in client.calls] == [None, 1, 2]


def test_it_reads_payloads_only_with_a_page_sized_limit_and_a_doc_filter() -> None:
    client = FakeScrollClient([[point(0)]])
    store_with(client).get_chunks("d" * 16)
    call = client.calls[0]
    assert call["collection"] == "c" and call["limit"] == 256 and call["with_payload"] is True and call["with_vectors"] is False
    assert call["filter"].must[0].key == "doc_id" and call["filter"].must[0].match.value == "d" * 16


def test_a_document_with_no_points_is_an_empty_list() -> None:
    assert store_with(FakeScrollClient([[]])).get_chunks("d" * 16) == []


def test_a_server_that_cannot_be_reached_becomes_a_store_error() -> None:
    class Down:
        def scroll(self, *args, **kwargs):
            raise ResponseHandlingException("connection refused")

    with pytest.raises(VectorStoreError) as info:
        store_with(Down()).get_chunks("d" * 16)
    assert info.value.code is StoreErrorCode.BACKEND_UNAVAILABLE
