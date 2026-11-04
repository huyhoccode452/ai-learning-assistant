"""End-to-end retrieval check on the real sample document (real Gemini API + real Qdrant).

    python scripts/try_retrieval.py            # temp collection, deleted at the end
    python scripts/try_retrieval.py --keep     # keep the temp collection for inspection

Flow: ingest the sample PDF -> embed chunks -> index -> run Vietnamese queries and print the
top hits with their scores. It prints score ranges per query category so YOU can reason about
SCORE_THRESHOLD; it deliberately does not choose one.
Costs ~38 embedded texts (free tier: ~100 per minute).
"""

import argparse
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embeddings import EmbeddingError, GeminiEmbedder  # noqa: E402
from app.ingestion import IngestionError, ingest_file  # noqa: E402
from app.vectorstore import QdrantStore, VectorStoreError  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "postgresql_performance_guide_vi.pdf"
report = logging.getLogger("try_retrieval")

RELEVANT, RELATED_ABSENT, UNRELATED = "relevant", "related-but-absent", "unrelated"


@dataclass(frozen=True)
class Query:
    text: str
    category: str
    expected_page: int | None = None  # where the author knows the answer lives (relevant only)


QUERIES = [
    Query("work_mem là gì và dấu hiệu nào cho thấy thiếu work_mem?", RELEVANT, 5),
    Query("Autovacuum chạy khi nào và ngưỡng là bao nhiêu dòng chết?", RELEVANT, 6),
    Query("Khi nào nên dùng chỉ mục một phần (partial index)?", RELEVANT, 3),
    Query("Phân vùng bảng có lợi ích gì so với việc DELETE dữ liệu cũ?", RELEVANT, 7),
    Query("PgBouncer có những chế độ pooling nào?", RELEVANT, 5),
    Query("Vì sao OFFSET lớn làm truy vấn chậm?", RELEVANT, 4),
    # Same domain (databases) but not covered by the document: the hardest case for a threshold.
    Query("Sharding trong MongoDB hoạt động như thế nào?", RELATED_ABSENT),
    Query("Cho tôi công thức nấu phở bò truyền thống.", UNRELATED),
    Query("Đội tuyển nào vô địch World Cup 2022?", UNRELATED),
]


def excerpt(text: str, width: int = 100) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def run(keep: bool, top_k: int) -> int:
    settings = Settings()
    embedder = GeminiEmbedder(settings)
    collection = f"try_retrieval_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(settings, collection=collection)

    report.info("Settings: model=%s dim=%d chunk_size=%d overlap=%d | collection=%s",
                settings.gemini_embedding_model, settings.embedding_dim, settings.chunk_size,
                settings.chunk_overlap, collection)
    try:
        store.ensure_collection()

        started = time.perf_counter()
        result = ingest_file(SAMPLE.name, SAMPLE.read_bytes(), settings)
        report.info("Ingested %s: %d pages -> %d chunks (%.2fs)",
                    result.filename, result.num_pages, len(result.chunks), time.perf_counter() - started)

        started = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in result.chunks])
        embed_s = time.perf_counter() - started
        started = time.perf_counter()
        store.index_document(result.doc_id, result.chunks, vectors)
        report.info("Embedded %d chunks in %.2fs, indexed in %.2fs", len(vectors), embed_s, time.perf_counter() - started)
        report.info("Indexed documents: %s\n", store.list_documents())

        top1: dict[str, list[float]] = {}
        top_all: dict[str, list[float]] = {}
        for number, query in enumerate(QUERIES, start=1):
            started = time.perf_counter()
            query_vector = embedder.embed_query(query.text)
            embed_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            hits = store.search(query_vector, top_k=top_k)
            search_ms = (time.perf_counter() - started) * 1000

            report.info("[%d] (%s) %s", number, query.category, query.text)
            report.info("    latency: embed_query=%.0f ms, search=%.0f ms, embed+search=%.0f ms",
                        embed_ms, search_ms, embed_ms + search_ms)
            for rank, hit in enumerate(hits, start=1):
                page = "-" if hit.page is None else hit.page
                report.info("    #%d score=%.4f page=%s chunk=%-2d | %s",
                            rank, hit.score, page, hit.chunk_index, excerpt(hit.text))
            if query.expected_page is not None:
                pages = [h.page for h in hits]
                report.info("    expected page %d: top-1 %s, within top-%d %s",
                            query.expected_page, "yes" if pages[:1] == [query.expected_page] else "NO",
                            top_k, "yes" if query.expected_page in pages else "NO")
            report.info("")
            top1.setdefault(query.category, []).append(hits[0].score)
            top_all.setdefault(query.category, []).extend(h.score for h in hits)

        report.info("=" * 78)
        report.info("Score ranges (cosine similarity). No SCORE_THRESHOLD is chosen here.")
        report.info("%-20s %8s | %-21s | %-21s", "category", "queries", "top-1 min .. max", f"top-{top_k} (all hits) min .. max")
        for category in (RELEVANT, RELATED_ABSENT, UNRELATED):
            ones, alls = top1[category], top_all[category]
            report.info("%-20s %8d | %.4f .. %.4f       | %.4f .. %.4f",
                        category, len(ones), min(ones), max(ones), min(alls), max(alls))
        report.info("Difference: min(relevant top-1) - max(unrelated top-1) = %.4f",
                    min(top1[RELEVANT]) - max(top1[UNRELATED]))
        report.info("Difference: min(relevant top-1) - max(related-but-absent top-1) = %.4f",
                    min(top1[RELEVANT]) - max(top1[RELATED_ABSENT]))
        return 0
    except (IngestionError, EmbeddingError, VectorStoreError) as exc:
        report.error("FAILED [%s]: %s", getattr(exc, "code", type(exc).__name__), exc)
        return 1
    finally:
        if keep:
            report.info("Kept collection %s", collection)
        else:
            try:
                store._client.delete_collection(collection)  # noqa: SLF001 - temp collection of this script
                report.info("Deleted temporary collection %s", collection)
            except Exception as exc:  # cleanup must not hide the real outcome
                report.error("Could not delete %s: %s", collection, exc)
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keep", action="store_true", help="do not delete the temporary collection")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)  # keep the report readable; warnings (retries) still show
    return run(args.keep, args.top_k)


if __name__ == "__main__":
    sys.exit(main())
