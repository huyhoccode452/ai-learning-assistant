"""End-to-end RAG check on the sample document (real Gemini API + real Qdrant).

    python scripts/try_rag.py                                  # settings from .env
    python scripts/try_rag.py --threshold 0.65 --thinking minimal
    python scripts/try_rag.py --model gemini-3.1-flash-lite    # override LLM_MODEL for this run only

Indexes the sample PDF into a temporary collection (29 embedded texts), asks ~10 Vietnamese
questions in 4 categories and prints for each: answer, sources (cited or not), used_fallback,
best retrieval score and the latency breakdown. It judges nothing: read the answers yourself.
Nothing here changes .env; --model / --thinking / --threshold only apply to this run.
"""

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embeddings import EmbeddingError, GeminiEmbedder  # noqa: E402
from app.ingestion import IngestionError, ingest_file  # noqa: E402
from app.rag import GeminiLLM, RagError, RagPipeline  # noqa: E402
from app.vectorstore import QdrantStore, VectorStoreError  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "postgresql_performance_guide_vi.pdf"
PROVISIONAL_THRESHOLD = 0.65  # used only when SCORE_THRESHOLD is not set in .env; NOT calibrated
report = logging.getLogger("try_rag")

SINGLE, MULTI, ABSENT, UNRELATED = "single-chunk", "multi-chunk", "related-but-absent", "unrelated"


@dataclass(frozen=True)
class Question:
    text: str
    category: str


QUESTIONS = [
    Question("PgBouncer có những chế độ pooling nào?", SINGLE),
    Question("work_mem mặc định là bao nhiêu và dấu hiệu nào cho thấy thiếu work_mem?", SINGLE),
    Question("Vì sao OFFSET lớn làm truy vấn chậm và nên thay bằng cách nào?", SINGLE),
    Question("Với máy chủ 16GB RAM, 8 lõi thì nên cấu hình bộ nhớ và kết nối thế nào, và vì sao không nên đặt max_connections quá cao?", MULTI),
    Question("Autovacuum chạy khi nào? Với bảng 10 triệu dòng thì có vấn đề gì và nên xử lý ra sao?", MULTI),
    Question("Phân vùng bảng có lợi gì so với việc DELETE dữ liệu cũ, và khi nào thì không nên phân vùng?", MULTI),
    Question("pg_upgrade dùng để làm gì và chạy như thế nào?", ABSENT),
    Question("Sharding trong PostgreSQL được cấu hình như thế nào?", ABSENT),
    Question("Cho tôi công thức nấu phở bò truyền thống.", UNRELATED),
    Question("Đội tuyển nào vô địch World Cup 2022?", UNRELATED),
]


def run(args: argparse.Namespace) -> int:
    settings = Settings()
    updates: dict[str, object] = {}
    if args.model:
        updates["gemini_llm_model"] = args.model
    if args.thinking:
        updates["llm_thinking_level"] = args.thinking
    settings = settings.model_copy(update=updates)

    threshold = args.threshold if args.threshold is not None else settings.score_threshold
    if threshold is None:
        threshold = PROVISIONAL_THRESHOLD
        report.info("SCORE_THRESHOLD is not set in .env; using the PROVISIONAL %.2f for this run.", threshold)

    embedder = GeminiEmbedder(settings)
    collection = f"try_rag_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(settings, collection=collection)
    report.info("LLM=%s thinking=%s | embedding=%s | threshold=%.3f top_k=%d | collection=%s\n",
                settings.gemini_llm_model, settings.llm_thinking_level or "default",
                settings.gemini_embedding_model, threshold, args.top_k, collection)
    try:
        store.ensure_collection()
        ingested = ingest_file(SAMPLE.name, SAMPLE.read_bytes(), settings)
        store.index_document(ingested.doc_id, ingested.chunks, embedder.embed_documents([c.text for c in ingested.chunks]))
        report.info("Indexed %s: %d chunks\n", ingested.filename, len(ingested.chunks))

        rag = RagPipeline(settings, embedder=embedder, store=store, llm=GeminiLLM(settings), score_threshold=threshold)
        rows = []
        for number, q in enumerate(QUESTIONS, start=1):
            report.info("=" * 100)
            report.info("[%d] (%s) %s", number, q.category, q.text)
            try:
                result = rag.answer(q.text, top_k=args.top_k)
            except RagError as exc:
                report.info("    RagError [%s]: %s\n", exc.code, exc.message)
                rows.append((number, q.category, "ERROR", None, "-", 0, 0, str(exc.code)))
                continue
            report.info("    used_fallback=%s best_score=%s invalid_citations=%s truncated=%s",
                        result.used_fallback,
                        "-" if result.best_score is None else f"{result.best_score:.4f}",
                        list(result.invalid_citations), result.truncated)
            lat = result.latency
            report.info("    latency: embed=%.0f ms, search=%.0f ms, llm=%.0f ms, total=%.0f ms",
                        lat.embed_ms, lat.search_ms, lat.llm_ms, lat.total_ms)
            report.info("    ANSWER: %s", result.answer.replace("\n", "\n            "))
            for s in result.sources:
                page = "-" if s.page is None else s.page
                report.info("    %s [%d] p%s chunk=%-2d score=%.4f | %s", "CITED  " if s.cited else "unused ",
                            s.number, page, s.chunk_index, s.score, s.excerpt[:70])
            report.info("")
            rows.append((number, q.category, "fallback" if result.used_fallback else "answered", result.best_score,
                         f"{lat.total_ms / 1000:.1f}s", len(result.sources), sum(s.cited for s in result.sources), ""))

        report.info("=" * 100)
        report.info("%-3s %-19s %-9s %-10s %-8s %-8s %-6s", "#", "category", "outcome", "best_score", "total", "sources", "cited")
        for n, cat, outcome, best, total, nsrc, ncited, note in rows:
            report.info("%-3d %-19s %-9s %-10s %-8s %-8d %-6d %s", n, cat, outcome,
                        "-" if best is None else f"{best:.4f}", total, nsrc, ncited, note)
        return 0
    except (IngestionError, EmbeddingError, VectorStoreError, RagError) as exc:
        report.error("FAILED [%s]: %s", getattr(exc, "code", type(exc).__name__), exc)
        return 1
    finally:
        try:
            store._client.delete_collection(collection)  # noqa: SLF001 - temp collection of this script
            report.info("Deleted temporary collection %s", collection)
        except Exception as exc:  # cleanup must not hide the real outcome
            report.error("Could not delete %s: %s", collection, exc)
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="LLM model for this run only (default: GEMINI_LLM_MODEL)")
    parser.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], help="thinking level for this run")
    parser.add_argument("--threshold", type=float, help="score threshold for this run")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
