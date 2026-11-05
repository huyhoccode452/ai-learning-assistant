"""Smoke test: verify config, Gemini (models/embed/generate) and Qdrant end to end.

Usage (from the project root, venv active):
    python scripts/smoke_test.py                 # run all steps
    python scripts/smoke_test.py config models   # run only some steps

Steps: config, models, embed, generate, qdrant. Exit code 0 only if every
requested step passes.
"""

import logging
import math
import sys
import time
import uuid
from pathlib import Path

# Allow `python scripts/smoke_test.py` (script dir is on sys.path, not the root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http.exceptions import (  # noqa: E402
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from app.config import Settings  # noqa: E402

logger = logging.getLogger("smoke_test")

STEPS = ("config", "models", "embed", "generate", "qdrant")
SAMPLE_VI = "Trí tuệ nhân tạo giúp sinh viên học tập hiệu quả hơn."
QUERY_VI = "AI có giúp sinh viên học tốt hơn không?"
GENERATE_PROMPT = "Trả lời bằng đúng một câu tiếng Việt: RAG là gì?"

_secret: str | None = None  # used only to redact error messages


def redact(text: str) -> str:
    """Remove the API key from any text before it is logged."""
    return text.replace(_secret, "***REDACTED***") if _secret else text


def report(step: str, ok: bool, detail: str) -> bool:
    logger.info("[%s] %-8s %s", "PASS" if ok else "FAIL", step, redact(detail))
    return ok


def gemini_error(exc: Exception) -> str:
    if isinstance(exc, genai_errors.APIError):
        return f"Gemini API error {exc.code} {exc.status}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embed_one(
    client: genai.Client, s: Settings, text: str, task_type: str, dim: int | None
) -> list[float]:
    """Embed one text. dim=None asks for the model's native dimensionality."""
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
    resp = client.models.embed_content(model=s.gemini_embedding_model, contents=text, config=config)
    return resp.embeddings[0].values


# ---------------------------------------------------------------- steps


def step_config() -> Settings | None:
    global _secret
    try:
        settings = Settings()
    except ValidationError as exc:
        report("config", False, f"invalid settings:\n{exc}")
        return None
    _secret = settings.gemini_api_key.get_secret_value()
    report("config", True, "settings loaded, GEMINI_API_KEY is set (value hidden)")
    return settings


def step_models(client: genai.Client) -> bool:
    try:
        models = list(client.models.list())
    except (genai_errors.APIError, httpx.HTTPError) as exc:
        return report("models", False, gemini_error(exc))

    generate = [m for m in models if "generateContent" in (m.supported_actions or [])]
    embed = [m for m in models if "embedContent" in (m.supported_actions or [])]

    def show(title: str, items: list[types.Model]) -> None:
        logger.info("\n  %s (%d):", title, len(items))
        for m in sorted(items, key=lambda m: m.name or ""):
            logger.info(
                "    %-45s in_tokens=%-8s %s",
                (m.name or "").removeprefix("models/"),
                m.input_token_limit,
                m.display_name or "",
            )

    show("generateContent -> candidates for GEMINI_LLM_MODEL", generate)
    show("embedContent    -> candidates for GEMINI_EMBEDDING_MODEL", embed)
    logger.info("")
    return report(
        "models",
        bool(generate and embed),
        f"{len(models)} models visible; {len(generate)} generateContent, "
        f"{len(embed)} embedContent",
    )


def step_embed(
    client: genai.Client, s: Settings
) -> tuple[bool, tuple[list[float], list[float]] | None]:
    """Return (passed, (document_vector, query_vector)) for reuse by the Qdrant step."""
    if not s.gemini_embedding_model or not s.embedding_dim:
        msg = "set GEMINI_EMBEDDING_MODEL and EMBEDDING_DIM in .env"
        return report("embed", False, msg), None
    try:
        # Native output size (informational: shows what EMBEDDING_DIM truncates from).
        native = embed_one(client, s, SAMPLE_VI, "RETRIEVAL_DOCUMENT", None)
        logger.info("  native dimension of %s: %d", s.gemini_embedding_model, len(native))
        # What the pipeline will use: asymmetric task types, dimension from .env.
        doc_vec = embed_one(client, s, SAMPLE_VI, "RETRIEVAL_DOCUMENT", s.embedding_dim)
        query_vec = embed_one(client, s, QUERY_VI, "RETRIEVAL_QUERY", s.embedding_dim)
    except (genai_errors.APIError, httpx.HTTPError) as exc:
        return report("embed", False, gemini_error(exc)), None

    norm = math.sqrt(sum(x * x for x in doc_vec))
    ok = len(doc_vec) == len(query_vec) == s.embedding_dim
    detail = (
        f"doc dim={len(doc_vec)}, query dim={len(query_vec)}, EMBEDDING_DIM={s.embedding_dim}, "
        f"doc L2 norm={norm:.4f}, cosine(doc, query)={cosine(doc_vec, query_vec):.4f}"
    )
    return report("embed", ok, detail), (doc_vec, query_vec)


def step_generate(client: genai.Client, s: Settings) -> bool:
    if not s.gemini_llm_model:
        return report("generate", False, "set GEMINI_LLM_MODEL in .env")
    start = time.perf_counter()
    try:
        resp = client.models.generate_content(model=s.gemini_llm_model, contents=GENERATE_PROMPT)
    except (genai_errors.APIError, httpx.HTTPError) as exc:
        return report("generate", False, gemini_error(exc))
    text = (resp.text or "").strip()
    elapsed = time.perf_counter() - start
    return report(
        "generate",
        bool(text),
        f"{elapsed:.2f}s -> {text!r}" if text else f"empty response: {resp!r}",
    )


def step_qdrant(s: Settings, vectors: tuple[list[float], list[float]] | None) -> bool:
    """Upsert a document vector, search with a query vector, delete the collection.

    Uses the real Gemini vectors from the embed step when available, otherwise
    two synthetic vectors. Expected score = cosine computed locally, which also
    proves the collection really uses Cosine distance.
    """
    if not s.embedding_dim:
        return report("qdrant", False, "set EMBEDDING_DIM in .env")
    if vectors is None:
        logger.info("  (no embed step results: using synthetic vectors)")
        doc_vec = [math.sin(i + 1) for i in range(s.embedding_dim)]
        query_vec = [math.sin(i + 1.5) for i in range(s.embedding_dim)]
    else:
        doc_vec, query_vec = vectors
    expected = cosine(doc_vec, query_vec)

    name = f"smoke_test_{uuid.uuid4().hex[:8]}"
    key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
    client = QdrantClient(url=s.qdrant_url, api_key=key, timeout=10)
    created = False
    try:
        logger.info("  Qdrant server version: %s", client.info().version)
        client.create_collection(
            name, vectors_config=VectorParams(size=s.embedding_dim, distance=Distance.COSINE)
        )
        created = True
        distance = client.get_collection(name).config.params.vectors.distance
        point_id = str(uuid.uuid4())
        client.upsert(
            name,
            points=[PointStruct(id=point_id, vector=doc_vec, payload={"text": SAMPLE_VI})],
            wait=True,
        )
        hits = client.query_points(name, query=query_vec, limit=1, with_payload=True).points
        ok = (
            bool(hits)
            and str(hits[0].id) == point_id
            and distance == Distance.COSINE
            and abs(hits[0].score - expected) < 1e-3
        )
        detail = (
            f"collection {name} dim={s.embedding_dim} distance={distance.value}: "
            f"upsert+search OK, score={hits[0].score:.4f} (local cosine {expected:.4f})"
            if ok
            else f"unexpected result: distance={distance}, expected={expected:.4f}, hits={hits!r}"
        )
    except (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError) as exc:
        return report(
            "qdrant",
            False,
            f"{type(exc).__name__}: {exc} (is Qdrant running? `docker compose up -d`)",
        )
    finally:
        if created:
            try:
                client.delete_collection(name)
                logger.info("  temporary collection %s deleted", name)
            except (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError) as exc:
                logger.error("  could not delete %s: %s", name, exc)
        client.close()
    return report("qdrant", ok, detail)


# ---------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request INFO lines
    requested = argv or list(STEPS)
    unknown = set(requested) - set(STEPS)
    if unknown:
        logger.error("unknown step(s): %s. Valid: %s", ", ".join(sorted(unknown)), ", ".join(STEPS))
        return 2

    settings = step_config()  # always needed
    if settings is None:
        return 1

    results: list[bool] = []
    vectors: tuple[list[float], list[float]] | None = None
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    for step in STEPS[1:]:
        if step not in requested:
            continue
        if step == "models":
            results.append(step_models(client))
        elif step == "embed":
            ok, vectors = step_embed(client, settings)
            results.append(ok)
        elif step == "generate":
            results.append(step_generate(client, settings))
        elif step == "qdrant":
            results.append(step_qdrant(settings, vectors))

    passed = sum(results) + 1  # + config
    logger.info("\n%d/%d steps passed", passed, len(results) + 1)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
