"""Qdrant vector store wrapper: one collection, Cosine distance, one point per chunk.

Point id = chunk_id (a stable uuid5), payload = doc_id, filename, page, chunk_index, text.
Indexing a document first deletes every point of that doc_id, so re-indexing with a
different chunking can never leave stale chunks behind (known pitfall, docs/known_issues.md).
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import Settings
from app.ingestion import Chunk

logger = logging.getLogger(__name__)

DISTANCE = Distance.COSINE
DOC_ID_FIELD = "doc_id"
UPSERT_BATCH_SIZE = 100
LIST_DOCUMENTS_LIMIT = 10_000  # upper bound on distinct documents returned by list_documents


class StoreErrorCode(StrEnum):
    NOT_CONFIGURED = "not_configured"
    COLLECTION_MISMATCH = "collection_mismatch"
    INVALID_INPUT = "invalid_input"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_ERROR = "backend_error"


class VectorStoreError(Exception):
    """Expected vector-store failure with a stable `code` and a readable `message`."""

    def __init__(self, code: StoreErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    score: float  # cosine similarity, higher = more similar
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    doc_id: str
    filename: str
    chunk_count: int


class QdrantStore:
    def __init__(
        self,
        settings: Settings,
        *,
        collection: str | None = None,
        client: QdrantClient | None = None,
        upsert_batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        if not settings.embedding_dim:
            raise VectorStoreError(StoreErrorCode.NOT_CONFIGURED, "EMBEDDING_DIM must be set in .env.")
        self.collection = collection or settings.qdrant_collection
        self.dim: int = settings.embedding_dim
        self._url = settings.qdrant_url
        self._batch_size = upsert_batch_size
        self._client = client or QdrantClient(url=settings.qdrant_url, timeout=30)

    def close(self) -> None:
        self._client.close()

    def is_reachable(self) -> bool:
        """True if the Qdrant server answers. Never raises (used by the /health endpoint)."""
        try:
            self._client.get_collections()
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            logger.warning("Qdrant is not reachable at %s: %s", self._url, type(exc).__name__)
            return False
        return True

    # ------------------------------------------------------------------ errors

    @contextmanager
    def _backend(self, action: str) -> Iterator[None]:
        """Turn Qdrant client exceptions into VectorStoreError with an actionable message."""
        try:
            yield
        except ResponseHandlingException as exc:
            raise VectorStoreError(
                StoreErrorCode.BACKEND_UNAVAILABLE,
                f"Cannot reach Qdrant at {self._url} while {action}: {exc}. "
                "Is it running? (docker compose up -d)",
            ) from exc
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                detail = f"collection '{self.collection}' not found; call ensure_collection() first"
            else:
                detail = f"HTTP {exc.status_code}: {exc.content.decode('utf-8', 'replace')[:300]}"
            raise VectorStoreError(
                StoreErrorCode.BACKEND_ERROR, f"Qdrant rejected the request while {action}: {detail}"
            ) from exc

    # ------------------------------------------------------------------ collection

    def ensure_collection(self) -> bool:
        """Create the collection if missing; verify it matches otherwise. True if created.

        Never deletes or modifies an existing collection: on a size/distance mismatch it
        raises and leaves the decision (recreate + re-index) to the human.
        """
        with self._backend("ensuring the collection"):
            created = False
            if not self._client.collection_exists(self.collection):
                try:
                    self._client.create_collection(
                        self.collection,
                        vectors_config=VectorParams(size=self.dim, distance=DISTANCE),
                    )
                    created = True
                except UnexpectedResponse as exc:
                    if exc.status_code != 409:  # 409 = someone created it between check and create
                        raise
            info = self._client.get_collection(self.collection)
            self._verify_config(info)
            if DOC_ID_FIELD not in info.payload_schema:
                self._client.create_payload_index(
                    self.collection, DOC_ID_FIELD, PayloadSchemaType.KEYWORD
                )
        if created:
            logger.info("Created collection %s (dim=%d, %s)", self.collection, self.dim, DISTANCE.value)
        return created

    def _verify_config(self, info: CollectionInfo) -> None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise VectorStoreError(
                StoreErrorCode.COLLECTION_MISMATCH,
                f"Collection '{self.collection}' uses named vectors, but this app expects a single "
                "unnamed vector. Delete the collection manually and re-index.",
            )
        if vectors.size != self.dim or vectors.distance != DISTANCE:
            raise VectorStoreError(
                StoreErrorCode.COLLECTION_MISMATCH,
                f"Collection '{self.collection}' has size={vectors.size}, "
                f"distance={vectors.distance.value}, but settings require size={self.dim}, "
                f"distance={DISTANCE.value}. Delete the collection yourself and re-index all "
                "documents (this code never deletes a collection).",
            )

    # ------------------------------------------------------------------ write

    def index_document(
        self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Replace everything stored for `doc_id` with these chunks. Returns points written.

        Delete-then-upsert is idempotent but not atomic: if the upsert fails midway the document
        is incomplete until index_document is called again (see docs/known_issues.md).
        """
        self._validate_for_indexing(doc_id, chunks, vectors)
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=list(vector),
                payload={
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        with self._backend("indexing a document"):
            self._delete_points(doc_id)
            for start in range(0, len(points), self._batch_size):
                self._client.upsert(
                    self.collection, points=points[start : start + self._batch_size], wait=True
                )
        logger.info("Indexed doc_id=%s: %d chunks into %s", doc_id, len(points), self.collection)
        return len(points)

    def _validate_for_indexing(
        self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        def invalid(message: str) -> VectorStoreError:
            return VectorStoreError(StoreErrorCode.INVALID_INPUT, message)

        if not chunks:
            raise invalid("Nothing to index: chunks is empty.")
        if len(chunks) != len(vectors):
            raise invalid(f"{len(chunks)} chunks but {len(vectors)} vectors; they must match 1:1.")
        foreign = [c.chunk_index for c in chunks if c.doc_id != doc_id]
        if foreign:
            raise invalid(f"chunks {foreign[:5]} do not belong to doc_id={doc_id}.")
        wrong = [i for i, v in enumerate(vectors) if len(v) != self.dim]
        if wrong:
            raise invalid(
                f"vector(s) at position {wrong[:5]} do not have {self.dim} dimensions "
                f"(got {len(vectors[wrong[0]])})."
            )

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of a document. Returns how many chunks it had."""
        with self._backend("deleting a document"):
            existing = self.count_chunks(doc_id)
            if existing:
                self._delete_points(doc_id)
        logger.info("Deleted doc_id=%s (%d chunks) from %s", doc_id, existing, self.collection)
        return existing

    def _delete_points(self, doc_id: str) -> None:
        self._client.delete(
            self.collection,
            points_selector=FilterSelector(filter=_doc_filter([doc_id])),
            wait=True,
        )

    # ------------------------------------------------------------------ read

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k most similar chunks, best first.

        doc_ids=None searches everything; an EMPTY list means "no documents" and returns [].
        score_threshold drops chunks whose cosine similarity is below it.
        """
        if top_k < 1:
            raise VectorStoreError(StoreErrorCode.INVALID_INPUT, "top_k must be at least 1.")
        if len(query_vector) != self.dim:
            raise VectorStoreError(
                StoreErrorCode.INVALID_INPUT,
                f"The query vector has {len(query_vector)} dimensions; the collection uses {self.dim}.",
            )
        if doc_ids is not None and len(doc_ids) == 0:
            return []
        with self._backend("searching"):
            response = self._client.query_points(
                self.collection,
                query=list(query_vector),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=_doc_filter(doc_ids) if doc_ids is not None else None,
                with_payload=True,
            )
        return [_to_retrieved(point.score, point.payload or {}) for point in response.points]

    def count_chunks(self, doc_id: str) -> int:
        with self._backend("counting chunks"):
            return self._client.count(
                self.collection, count_filter=_doc_filter([doc_id]), exact=True
            ).count

    def is_indexed(self, doc_id: str) -> bool:
        return self.count_chunks(doc_id) > 0

    def list_documents(self) -> list[DocumentInfo]:
        """Distinct documents with their chunk counts (exact), sorted by filename."""
        with self._backend("listing documents"):
            facet = self._client.facet(
                self.collection, key=DOC_ID_FIELD, limit=LIST_DOCUMENTS_LIMIT, exact=True
            )
            documents = []
            for hit in facet.hits:
                doc_id = str(hit.value)
                points, _ = self._client.scroll(
                    self.collection,
                    scroll_filter=_doc_filter([doc_id]),
                    limit=1,
                    with_payload=["filename"],
                    with_vectors=False,
                )
                filename = (points[0].payload or {}).get("filename", "") if points else ""
                documents.append(DocumentInfo(doc_id, str(filename), hit.count))
        return sorted(documents, key=lambda d: (d.filename, d.doc_id))


def _doc_filter(doc_ids: Sequence[str]) -> Filter:
    condition = (
        FieldCondition(key=DOC_ID_FIELD, match=MatchValue(value=doc_ids[0]))
        if len(doc_ids) == 1
        else FieldCondition(key=DOC_ID_FIELD, match=MatchAny(any=list(doc_ids)))
    )
    return Filter(must=[condition])


def _to_retrieved(score: float, payload: dict) -> RetrievedChunk:
    return RetrievedChunk(
        score=score,
        doc_id=str(payload.get("doc_id", "")),
        filename=str(payload.get("filename", "")),
        page=payload.get("page"),
        chunk_index=int(payload.get("chunk_index", -1)),
        text=str(payload.get("text", "")),
    )
