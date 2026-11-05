"""What summary and quiz generation share: domain errors, an LLM call counter, text normalisation and the
SQLite result cache.

Why a cache: every LLM call spends free-tier quota (KI-18). A result is stored under a key made of
(kind, doc_id, hash of the indexed text, request parameters, model + thinking level, prompt version), so the
same request is answered from SQLite with zero LLM calls, and any change that could change the answer
(another model, an edited prompt, a re-indexed document) produces a different key.
"""

import hashlib
import json
import logging
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.config import Settings
from app.db import Database
from app.rag import LLM, LLMResult
from app.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)


class GenerationErrorCode(StrEnum):
    NO_CONTENT = "no_content"  # the document is 'ready' but the index has no text for it (KI-25)
    EMPTY_SELECTION = "empty_selection"  # the page range / topic matched nothing
    MALFORMED_OUTPUT = "malformed_output"  # the model's reply could not be turned into the requested structure


class GenerationError(Exception):
    """Expected generation failure with a stable `code` and a readable `message` (mapped to HTTP in main.py)."""

    def __init__(self, code: GenerationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CountingLLM:
    """Wraps an LLM and counts the calls made through it (retries inside GeminiLLM are not visible here)."""

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self.calls = 0
        self.llm_seconds = 0.0

    def generate(
        self, system_instruction: str, prompt: str, *, response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        self.calls += 1
        started = time.perf_counter()
        try:
            return self._inner.generate(
                system_instruction, prompt, response_schema=response_schema, max_output_tokens=max_output_tokens
            )
        finally:
            self.llm_seconds += time.perf_counter() - started


def normalize_text(text: str) -> str:
    """Unicode NFC and single spaces: PDF text has arbitrary line breaks and may use decomposed accents."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_hash(chunks: Sequence[RetrievedChunk]) -> str:
    """Identifies the indexed text. If the document is re-indexed with other chunking, cached results stop matching."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_index}:{chunk.page}:".encode())
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:20]


def llm_identity(settings: Settings) -> str:
    return f"{settings.gemini_llm_model}|thinking={settings.llm_thinking_level or 'default'}"


def cache_key(kind: str, doc_id: str, text_hash: str, params: dict[str, Any], model: str, prompt_version: str) -> str:
    material = {"kind": kind, "doc": doc_id, "text": text_hash, "params": params, "model": model, "prompt": prompt_version}
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GenResult:
    payload: dict[str, Any]
    cached: bool
    llm_calls: int  # made by THIS request: 0 when answered from the cache
    original_llm_calls: int  # what producing the result cost when it was generated
    total_ms: float
    llm_ms: float


def run_cached(
    db: Database,
    llm: LLM,
    *,
    kind: str,
    doc_id: str,
    text_hash: str,
    params: dict[str, Any],
    model: str,
    prompt_version: str,
    refresh: bool,
    compute: Callable[[CountingLLM], tuple[dict[str, Any], bool]],
) -> GenResult:
    """Answer from the cache, or call `compute(counting_llm)` which returns (payload, cacheable).

    `refresh=True` skips the lookup (a new result is still stored). A result that is not cacheable (for example a
    quiz whose verification could not run, or one with no valid question) is returned but never stored, so asking
    again gets another chance.
    """
    started = time.perf_counter()
    key = cache_key(kind, doc_id, text_hash, params, model, prompt_version)
    if not refresh:
        hit = db.get_generation(key)
        if hit is not None:
            try:
                payload = json.loads(hit.result_json)
            except ValueError:
                logger.warning("Cached %s for doc_id=%s is unreadable; regenerating", kind, doc_id)
            else:
                logger.info("%s cache hit doc_id=%s (0 LLM calls; it cost %d when generated)", kind, doc_id, hit.llm_calls)
                return GenResult(payload, True, 0, hit.llm_calls, (time.perf_counter() - started) * 1000, 0.0)
    counting = CountingLLM(llm)
    payload, cacheable = compute(counting)
    if cacheable:
        try:
            db.put_generation(key, doc_id, kind, json.dumps(params, sort_keys=True, ensure_ascii=False), model,
                              prompt_version, json.dumps(payload, ensure_ascii=False), counting.calls)
        except sqlite3.IntegrityError:  # the document was deleted while we were generating
            logger.warning("Not caching %s: doc_id=%s no longer exists", kind, doc_id)
    logger.info("%s generated doc_id=%s with %d LLM call(s)", kind, doc_id, counting.calls)
    return GenResult(payload, False, counting.calls, counting.calls, (time.perf_counter() - started) * 1000,
                     counting.llm_seconds * 1000)
