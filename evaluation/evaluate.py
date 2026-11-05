"""Evaluation of the RAG pipeline on evaluation/questions.json. Real API calls, real Qdrant.

    python evaluation/evaluate.py                  # everything: retrieval, threshold study, end-to-end, judge
    python evaluation/evaluate.py --skip-llm       # retrieval + threshold study only (embedding calls only)
    python evaluation/evaluate.py --report-only    # rebuild report.md from results.json (no API calls)
    python evaluation/evaluate.py --model gemini-3.1-flash-lite --thinking minimal   # this run only

Stages (each one saves evaluation/results.json when it ends):
  1. index the sample PDF into a temporary collection (random name, deleted at the end) with the
     same functions the app uses: ingest_file -> embed_documents -> QdrantStore.index_document;
  2. retrieval: top-5 per question WITHOUT any threshold, so the raw scores can be studied;
  3. end-to-end: RagPipeline.answer for every question, with the threshold from .env;
  4. judge: one batched LLM call per answered question (correctness + faithfulness).

Every API call is cached in evaluation/cache/ under a hash of its inputs, so an interrupted or
quota-limited run resumes where it stopped and a finished run can be repeated without any quota.
A question that could not be finished is listed as MISSING, never filled in. Nothing here changes .env.
"""

import argparse
import json
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embeddings import Embedder, EmbeddingError, GeminiEmbedder  # noqa: E402
from app.embeddings import ErrorCode as EmbeddingErrorCode  # noqa: E402
from app.ingestion import Chunk, IngestionError, IngestResult, ingest_file  # noqa: E402
from app.prompts import RAG_SYSTEM_PROMPT  # noqa: E402
from app.rag import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    LLM,
    GeminiLLM,
    LLMResult,
    RagAnswer,
    RagError,
    RagErrorCode,
    RagPipeline,
)
from app.vectorstore import QdrantStore, VectorStoreError  # noqa: E402
from evaluation import report as report_module  # noqa: E402
from evaluation.judge import (  # noqa: E402
    DiskCache,
    JudgeChunk,
    JudgeError,
    judge_answer,
)
from evaluation.metrics import classify_outcome  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_PATH = EVAL_DIR / "results.json"
REPORT_PATH = EVAL_DIR / "report.md"
ANALYSIS_PATH = EVAL_DIR / "analysis_vi.md"  # hand-written, inserted into the report if present
REVIEW_PATH = EVAL_DIR / "claude_review.json"  # second opinion on the judge, if present

KS = (3, 5)
RETRIEVAL_K = max(KS)
SAMPLE_VERDICTS = 5
DEFAULT_JUDGE_THINKING = "low"
QUESTION_TYPES = {"single", "multi", "insufficient", "unrelated"}

log = logging.getLogger("evaluate")


# ---------------------------------------------------------------------- questions


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    type: str
    question: str
    reference_answer: str
    relevant_pages: tuple[int, ...]
    expect_fallback: bool


def load_questions(path: Path) -> tuple[str, list[Question]]:
    """Read and validate the labelled questions. The file is never modified."""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for raw in data["questions"]:
        q = Question(
            id=raw["id"],
            type=raw["type"],
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            relevant_pages=tuple(raw["relevant_pages"]),
            expect_fallback=raw["expect_fallback"],
        )
        if q.type not in QUESTION_TYPES:
            raise ValueError(f"{q.id}: unknown type {q.type!r}")
        if q.expect_fallback == bool(q.relevant_pages):
            raise ValueError(f"{q.id}: expect_fallback={q.expect_fallback} contradicts relevant_pages={list(q.relevant_pages)}")
        questions.append(q)
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return data["document"], questions


# ---------------------------------------------------------------------- small helpers


class QuotaGuardLLM:
    """After the first RATE_LIMITED error, every later call fails at once with the same error.

    Without it, each remaining question would wait out its own retry/backoff against a quota that
    is already gone (a daily quota does not come back within a run).
    """

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self.exhausted: RagError | None = None

    def generate(self, system_instruction: str, prompt: str) -> LLMResult:
        if self.exhausted is not None:
            raise self.exhausted
        try:
            return self._inner.generate(system_instruction, prompt)
        except RagError as exc:
            if exc.code is RagErrorCode.RATE_LIMITED:
                self.exhausted = exc
            raise


class RetryCounter(logging.Handler):
    """Counts the 'retry n/m' warnings of the Gemini clients, so a slow answer can be told apart from
    a slow call that was retried."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("app.") and "retry" in record.getMessage():
            self.count += 1


def _error(exc: Exception) -> dict[str, str]:
    code = getattr(exc, "code", type(exc).__name__)
    return {"code": str(code), "message": str(getattr(exc, "message", exc))[:300]}


# ---------------------------------------------------------------------- stage 1: index


@dataclass
class Index:
    ingested: IngestResult
    store: QdrantStore
    fingerprint: dict[str, Any]  # everything that decides what the vectors are; part of every cache key
    chunks_by_index: dict[int, Chunk]


def build_index(settings: Settings, embedder: Embedder, cache: DiskCache, collection: str, sample: Path) -> Index:
    ingested = ingest_file(sample.name, sample.read_bytes(), settings)
    texts = [c.text for c in ingested.chunks]
    fingerprint = {
        "doc_id": ingested.doc_id,
        "chunks": len(texts),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "texts_sha": DiskCache.key({"texts": texts}),
        "embedding_model": settings.gemini_embedding_model,
        "embedding_dim": settings.embedding_dim,
    }
    vectors = cache.get("doc_vectors", fingerprint)
    if not (isinstance(vectors, list) and len(vectors) == len(texts)):
        vectors = embedder.embed_documents(texts)
        cache.put("doc_vectors", fingerprint, vectors)
    else:
        log.info("Document vectors: from cache (no embedding call)")
    store = QdrantStore(settings, collection=collection)
    store.ensure_collection()
    store.index_document(ingested.doc_id, ingested.chunks, vectors)
    log.info("Indexed %s into temporary collection %s: %d chunks, %s pages",
             ingested.filename, collection, len(texts), ingested.num_pages)
    return Index(ingested, store, fingerprint, {c.chunk_index: c for c in ingested.chunks})


# ---------------------------------------------------------------------- stage 2: retrieval


def run_retrieval(
    questions: list[Question], index: Index, embedder: Embedder, cache: DiskCache, records: dict[str, dict]
) -> None:
    """Top-RETRIEVAL_K per question with no threshold. Stops early only on embedding quota."""
    for q in questions:
        material = {"question": q.question, "index": index.fingerprint, "k": RETRIEVAL_K}
        hits = cache.get("retrieval", material)
        if hits is None:
            try:
                vector = embedder.embed_query(q.question)
                found = index.store.search(vector, RETRIEVAL_K)
            except (EmbeddingError, VectorStoreError) as exc:
                records[q.id]["retrieval_error"] = _error(exc)
                log.error("%s retrieval failed: %s", q.id, _error(exc))
                if getattr(exc, "code", None) == EmbeddingErrorCode.RATE_LIMITED:
                    log.error("Embedding quota exhausted; stopping the retrieval stage.")
                    return
                continue
            hits = [{"score": h.score, "page": h.page, "chunk_index": h.chunk_index} for h in found]
            cache.put("retrieval", material, hits)
        records[q.id]["hits"] = hits


# ---------------------------------------------------------------------- stage 3: end-to-end


def _answer_record(answer: RagAnswer, retries: int) -> dict[str, Any]:
    return {
        "answer": answer.answer,
        "used_fallback": answer.used_fallback,
        "best_score": answer.best_score,
        "sources": [
            {"number": s.number, "page": s.page, "chunk_index": s.chunk_index, "score": s.score, "cited": s.cited}
            for s in answer.sources
        ],
        "latency": {
            "embed_ms": answer.latency.embed_ms,
            "search_ms": answer.latency.search_ms,
            "llm_ms": answer.latency.llm_ms,
            "total_ms": answer.latency.total_ms,
        },
        "retries": retries,
        "invalid_citations": list(answer.invalid_citations),
        "truncated": answer.truncated,
    }


def run_end_to_end(
    questions: list[Question], index: Index, pipeline: RagPipeline, guard: QuotaGuardLLM, settings: Settings,
    cache: DiskCache, records: dict[str, dict],
) -> None:
    retry_counter = RetryCounter()
    logging.getLogger("app").addHandler(retry_counter)
    try:
        for q in questions:
            material = {
                "question": q.question,
                "index": index.fingerprint,
                "llm_model": settings.gemini_llm_model,
                "thinking": settings.llm_thinking_level,
                "temperature": DEFAULT_TEMPERATURE,
                "system": DiskCache.key({"s": RAG_SYSTEM_PROMPT}),
                "threshold": pipeline.score_threshold,
                "top_k": settings.top_k,
            }
            cached = cache.get("e2e", material)
            if cached is not None:
                records[q.id]["e2e"] = {**cached, "from_cache": True}
                continue
            retry_counter.count = 0
            try:
                answer = pipeline.answer(q.question)
            except (RagError, EmbeddingError, VectorStoreError) as exc:
                records[q.id]["e2e_error"] = _error(exc)
                log.error("%s answer failed: %s", q.id, _error(exc))
                continue
            record = _answer_record(answer, retry_counter.count)
            cache.put("e2e", material, record)
            records[q.id]["e2e"] = {**record, "from_cache": False}
            log.info("%s %s fallback=%s total=%.1fs retries=%d", q.id, q.type, answer.used_fallback,
                     answer.latency.total_ms / 1000, retry_counter.count)
        if guard.exhausted:
            log.error("LLM quota exhausted during the end-to-end stage: %s", guard.exhausted.message)
    finally:
        logging.getLogger("app").removeHandler(retry_counter)


# ---------------------------------------------------------------------- stage 4: judge


def run_judge(
    questions: list[Question], index: Index, judge_llm: LLM, judge_id: str, cache: DiskCache, records: dict[str, dict],
) -> None:
    for q in questions:
        e2e = records[q.id].get("e2e")
        if e2e is None or e2e["used_fallback"]:
            continue  # nothing to grade: no answer yet, or the pipeline's fixed fallback text
        chunks = [
            JudgeChunk(s["number"], s["page"], index.chunks_by_index[s["chunk_index"]].text) for s in e2e["sources"]
        ]
        try:
            verdict, from_cache = judge_answer(
                judge_llm, cache, model_id=judge_id, question=q.question, reference=q.reference_answer,
                answer=e2e["answer"], chunks=chunks,
            )
        except (RagError, JudgeError) as exc:
            records[q.id]["judge_error"] = _error(exc)
            log.error("%s judge failed: %s", q.id, _error(exc))
            continue
        records[q.id]["judge"] = {
            "correctness": verdict.correctness,
            "is_refusal": verdict.is_refusal,
            "faithful": verdict.faithful,
            "unsupported_claims": list(verdict.unsupported_claims),
            "notes": verdict.notes,
            "from_cache": from_cache,
        }
        log.info("%s judged: %s faithful=%s%s", q.id, verdict.correctness, verdict.faithful,
                 " (cache)" if from_cache else "")


# ---------------------------------------------------------------------- assembling the results


def assemble(meta: dict, questions: list[Question], records: dict[str, dict], index: Index) -> dict:
    chunks_per_page: dict[str, int] = {}
    for chunk in index.ingested.chunks:
        key = str(chunk.page)
        chunks_per_page[key] = chunks_per_page.get(key, 0) + 1
    rows = []
    for q in questions:
        rec = records[q.id]
        e2e, judge = rec.get("e2e"), rec.get("judge")
        outcome = None
        if e2e is not None:
            outcome = str(classify_outcome(
                q.expect_fallback, e2e["used_fallback"],
                judge["correctness"] if judge else None, judge["is_refusal"] if judge else None,
            ))
        rows.append({
            "id": q.id, "type": q.type, "question": q.question, "reference_answer": q.reference_answer,
            "relevant_pages": list(q.relevant_pages), "expect_fallback": q.expect_fallback,
            "hits": rec.get("hits"), "retrieval_error": rec.get("retrieval_error"),
            "e2e": e2e, "e2e_error": rec.get("e2e_error"),
            "judge": judge, "judge_error": rec.get("judge_error"),
            "outcome": outcome,
        })
    return {"meta": meta, "chunks_per_page": chunks_per_page, "questions": rows}


def choose_sample(results: dict, seed: int) -> list[str]:
    judged = [q["id"] for q in results["questions"] if q["judge"]]
    return sorted(random.Random(seed).sample(judged, min(SAMPLE_VERDICTS, len(judged))))


def print_sample(results: dict) -> None:
    by_id = {q["id"]: q for q in results["questions"]}
    log.info("\n" + "=" * 100)
    log.info("%d judge verdicts picked at random (seed %s) for YOUR manual check:", len(results["meta"]["sample_ids"]),
             results["meta"]["seed"])
    for qid in results["meta"]["sample_ids"]:
        q, j = by_id[qid], by_id[qid]["judge"]
        log.info("\n[%s] %s\n  REFERENCE: %s\n  ANSWER   : %s\n  JUDGE    : %s | refusal=%s | faithful=%s | %s",
                 qid, q["question"], q["reference_answer"], q["e2e"]["answer"].replace("\n", " "),
                 j["correctness"], j["is_refusal"], j["faithful"], j["notes"])
        for claim in j["unsupported_claims"]:
            log.info("    unsupported claim: %s", claim)


def write_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_report(results: dict) -> None:
    analysis = ANALYSIS_PATH.read_text(encoding="utf-8") if ANALYSIS_PATH.exists() else None
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8")) if REVIEW_PATH.exists() else None
    REPORT_PATH.write_text(report_module.render_report(results, analysis, review), encoding="utf-8", newline="\n")
    log.info("Wrote %s and %s", RESULTS_PATH.name, REPORT_PATH.name)


# ---------------------------------------------------------------------- main


def run(args: argparse.Namespace) -> int:
    if args.report_only:
        write_report(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
        return 0

    base = Settings()
    updates: dict[str, object] = {}
    if args.model:
        updates["gemini_llm_model"] = args.model
    if args.thinking:
        updates["llm_thinking_level"] = args.thinking
    settings = base.model_copy(update=updates)
    threshold = args.threshold if args.threshold is not None else settings.score_threshold
    if threshold is None:
        log.error("SCORE_THRESHOLD is not set in .env and --threshold was not given.")
        return 1

    document, questions = load_questions(QUESTIONS_PATH)
    sample = PROJECT_ROOT / "data" / "samples" / document
    cache = DiskCache(CACHE_DIR)
    embedder = GeminiEmbedder(settings)
    judge_settings = settings.model_copy(update={
        "gemini_llm_model": args.judge_model or settings.gemini_llm_model,
        "llm_thinking_level": args.judge_thinking,
    })
    judge_id = f"{judge_settings.gemini_llm_model}|thinking={args.judge_thinking}|temperature=0"
    seed = args.seed if args.seed is not None else int(time.time())
    collection = f"eval_{uuid.uuid4().hex[:8]}"
    records: dict[str, dict] = {q.id: {} for q in questions}
    index: Index | None = None
    store: QdrantStore | None = None
    log.info("LLM=%s thinking=%s | judge=%s | embedding=%s dim=%s | chunk=%s/%s top_k=%s threshold=%.3f | %s",
             settings.gemini_llm_model, settings.llm_thinking_level, judge_id, settings.gemini_embedding_model,
             settings.embedding_dim, settings.chunk_size, settings.chunk_overlap, settings.top_k, threshold, collection)
    try:
        index = build_index(settings, embedder, cache, collection, sample)
        store = index.store
        meta = {
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "document": document, "doc_id": index.ingested.doc_id, "num_pages": index.ingested.num_pages,
            "num_chunks": len(index.ingested.chunks), "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "top_k": settings.top_k, "ks": list(KS),
            "score_threshold": threshold, "embedding_model": settings.gemini_embedding_model,
            "embedding_dim": settings.embedding_dim, "llm_model": settings.gemini_llm_model,
            "llm_thinking": settings.llm_thinking_level, "env_llm_model": base.gemini_llm_model,
            "env_llm_thinking": base.llm_thinking_level, "judge_model": judge_settings.gemini_llm_model,
            "judge_thinking": args.judge_thinking, "seed": seed, "sample_ids": [], "stages": [],
        }

        run_retrieval(questions, index, embedder, cache, records)
        meta["stages"].append("retrieval")
        write_results(assemble(meta, questions, records, index))

        if not args.skip_llm:
            generation_guard = QuotaGuardLLM(GeminiLLM(settings))
            pipeline = RagPipeline(settings, embedder=embedder, store=index.store, llm=generation_guard,
                                   score_threshold=threshold)
            run_end_to_end(questions, index, pipeline, generation_guard, settings, cache, records)
            meta["stages"].append("end_to_end")
            write_results(assemble(meta, questions, records, index))

            judge_guard = QuotaGuardLLM(GeminiLLM(judge_settings, temperature=0.0))
            run_judge(questions, index, judge_guard, judge_id, cache, records)
            meta["stages"].append("judge")

        results = assemble(meta, questions, records, index)
        meta["sample_ids"] = choose_sample(results, seed)
        results = assemble(meta, questions, records, index)
        write_results(results)
        write_report(results)
        if meta["sample_ids"]:
            print_sample(results)
        for stage, ids in report_module.find_missing(results).items():
            if ids:
                log.warning("MISSING in stage %s: %s (run again to resume; cached work is not repeated)",
                            stage, ", ".join(ids))
        return 0
    except (IngestionError, EmbeddingError, VectorStoreError, RagError) as exc:
        log.error("FAILED [%s]: %s", getattr(exc, "code", type(exc).__name__), exc)
        return 1
    finally:
        if store is not None:
            try:
                store._client.delete_collection(collection)  # noqa: SLF001 - temporary collection of this script
                gone = not store._client.collection_exists(collection)  # noqa: SLF001
                log.info("Temporary collection %s %s", collection, "deleted" if gone else "STILL EXISTS")
            except Exception as exc:  # cleanup must not hide the real outcome
                log.error("Could not delete %s: %s", collection, exc)
            store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="LLM model for the answers, this run only (default: GEMINI_LLM_MODEL)")
    parser.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], help="thinking level of the answering LLM")
    parser.add_argument("--judge-model", help="model for the judge (default: the answering model)")
    parser.add_argument("--judge-thinking", choices=["minimal", "low", "medium", "high"], default=DEFAULT_JUDGE_THINKING)
    parser.add_argument("--threshold", type=float, help="score threshold for the end-to-end stage (default: SCORE_THRESHOLD)")
    parser.add_argument("--seed", type=int, help="seed for the random choice of judge verdicts to print")
    parser.add_argument("--skip-llm", action="store_true", help="run only the retrieval stage and the threshold study")
    parser.add_argument("--report-only", action="store_true", help="rebuild report.md from results.json, no API calls")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)  # keeps the retry warnings, hides routine INFO
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
