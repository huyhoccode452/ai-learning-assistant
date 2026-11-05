"""Measure real LLM latency with the real RAG prompt (real Gemini API, no Qdrant, no embedding).

    python scripts/bench_llm.py                      # default matrix, 3 runs each
    python scripts/bench_llm.py --runs 5 --pause 8

Each configuration is (model, thinking_level). The prompt is built from real chunks of the sample
PDF (5 chunks, like TOP_K=5). Retries are disabled so a 429 is reported as an error and never
hides inside a latency number. Nothing here changes .env.
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.ingestion import ingest_file  # noqa: E402
from app.prompts import RAG_SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from app.rag import GeminiLLM, RagError  # noqa: E402
from app.vectorstore import RetrievedChunk  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "postgresql_performance_guide_vi.pdf"
QUESTION = "PgBouncer có những chế độ pooling nào, và chế độ nào được dùng phổ biến nhất?"
CHUNK_INDEXES = [13, 14, 11, 12, 10]  # chunks around the PgBouncer / connection text

# (model, thinking_level). None = the model's default behaviour.
MATRIX = [
    ("gemini-3.5-flash", None),
    ("gemini-3.5-flash", "minimal"),
    ("gemini-3.5-flash", "low"),
    ("gemini-3.1-flash-lite", None),
    ("gemini-3.1-flash-lite", "minimal"),
]

log = logging.getLogger("bench_llm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between calls (rate limits)")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--only", nargs="+", metavar="MODEL:LEVEL",
                        help="override the matrix, e.g. gemini-3.5-flash:minimal gemini-3.5-flash:default")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings()
    result = ingest_file(SAMPLE.name, SAMPLE.read_bytes(), settings)
    chunks = [
        RetrievedChunk(0.8, c.doc_id, c.filename, c.page, c.chunk_index, c.text)
        for c in (result.chunks[i] for i in CHUNK_INDEXES)
    ]
    prompt = build_user_prompt(QUESTION, chunks)
    log.info("Prompt: %d chars (%d chunks). Question: %s\n", len(prompt), len(chunks), QUESTION)

    matrix = MATRIX
    if args.only:
        matrix = [(m, None if lv == "default" else lv) for m, lv in (item.split(":") for item in args.only)]
    summary = []
    for model, level in matrix:
        llm = GeminiLLM(
            settings.model_copy(update={"gemini_llm_model": model}),
            thinking_level=level, timeout_s=args.timeout, max_retries=0,
        )
        label = f"{model} thinking={level or 'default'}"
        times: list[float] = []
        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            try:
                out = llm.generate(RAG_SYSTEM_PROMPT, prompt)
            except RagError as exc:
                log.info("%-48s run %d: ERROR [%s] %s", label, run, exc.code, exc.message[:160])
                time.sleep(args.pause)
                continue
            seconds = time.perf_counter() - started
            times.append(seconds)
            log.info("%-48s run %d: %5.2fs  tokens in/out/thinking=%s/%s/%s  answer=%d chars%s",
                     label, run, seconds, out.input_tokens, out.output_tokens, out.thinking_tokens,
                     len(out.text), "  TRUNCATED" if out.truncated else "")
            if run == 1:
                log.info("    answer: %s", " ".join(out.text.split())[:300])
            time.sleep(args.pause)
        summary.append((label, times))

    log.info("\n%-48s %5s %8s %8s %8s", "configuration", "ok", "min s", "median s", "max s")
    for label, times in summary:
        if times:
            log.info("%-48s %5d %8.2f %8.2f %8.2f", label, len(times), min(times),
                     statistics.median(times), max(times))
        else:
            log.info("%-48s %5d %8s %8s %8s", label, 0, "-", "-", "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
