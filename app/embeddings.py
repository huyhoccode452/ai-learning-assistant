"""Embedding interface and Gemini embedding implementation.

Facts this module is built on (each one measured against the real API, not assumed):
  * One batch request accepts at most 100 texts (the API answers INVALID_ARGUMENT above that).
  * Free-tier quota counts every TEXT as one request per minute (~100/min), not every call.
    A 429 carries a RetryInfo with the delay the server wants (e.g. "54s").
  * gemini-embedding-001 accepts over-long input WITHOUT an error and without any "truncated"
    flag, so the only way to notice is to count tokens first (count_tokens works on it).
  * An empty string fails the whole batch (400); a whitespace-only string is accepted and
    yields a meaningless vector, so both must be rejected client-side.
  * output_dimensionality truncates the 3072-d vector to its first N values (Matryoshka). The
    result is no longer unit length, and the docs say to normalize it ourselves.
"""

import logging
import math
import random
import re
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_BATCH_SIZE = 100  # hard API limit, see module docstring
MAX_INPUT_TOKENS = 2048  # gemini-embedding-001 input_token_limit

# Texts up to this many characters are not token-counted. Assumption: a token covers at least
# half a character, so 1000 chars <= 2000 tokens < 2048. Chunks are ~800 chars, so in normal
# operation count_tokens is never called; it only guards unusually long texts.
EXACT_COUNT_MIN_CHARS = 1000

RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
AUTH_HTTP_CODES = frozenset({401, 403})

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"  # texts that will be stored and searched
TASK_QUERY = "RETRIEVAL_QUERY"  # the user's question

_DURATION = re.compile(r"(\d+(?:\.\d+)?)s")  # protobuf Duration, e.g. "54s" or "0.5s"


class ErrorCode(StrEnum):
    NOT_CONFIGURED = "not_configured"
    EMPTY_TEXT = "empty_text"
    TEXT_TOO_LONG = "text_too_long"
    DIMENSION_MISMATCH = "dimension_mismatch"
    BAD_RESPONSE = "bad_response"
    REQUEST_REJECTED = "request_rejected"  # 400/404/...: retrying cannot help
    AUTH_ERROR = "auth_error"  # 401/403
    RATE_LIMITED = "rate_limited"  # 429 and retries exhausted (or wait too long)
    SERVICE_UNAVAILABLE = "service_unavailable"  # 5xx / network, retries exhausted


class EmbeddingError(Exception):
    """Expected embedding failure with a stable `code` and a readable `message`."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Embedder(Protocol):
    """What the rest of the app needs. Swap implementations (or fake it in tests) freely."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in the same order as `texts`."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Vector for a search question (different task type than documents)."""
        ...


def _retry_delay_from(exc: genai_errors.APIError) -> float | None:
    """Seconds the server asked us to wait (google.rpc.RetryInfo), if it said so."""
    details = exc.details if isinstance(exc.details, dict) else {}
    error = details.get("error", details)
    items = error.get("details", []) if isinstance(error, dict) else []
    for item in items:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and (match := _DURATION.fullmatch(delay.strip())):
            return float(match.group(1))
    return None


class GeminiEmbedder:
    """Gemini embeddings with batching, retry/backoff, validation and L2 normalization."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        max_retries: int | None = None,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        batch_size: int = MAX_BATCH_SIZE,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not settings.gemini_embedding_model or not settings.embedding_dim:
            raise EmbeddingError(
                ErrorCode.NOT_CONFIGURED,
                "GEMINI_EMBEDDING_MODEL and EMBEDDING_DIM must be set in .env.",
            )
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
        self._model: str = settings.gemini_embedding_model
        self._dim: int = settings.embedding_dim
        self._max_retries = settings.embed_max_retries if max_retries is None else max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._batch_size = batch_size
        self._sleep = sleep
        self._jitter = jitter
        self._secret = settings.gemini_api_key.get_secret_value()
        self._client = client or genai.Client(
            api_key=self._secret,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # milliseconds
        )

    # ------------------------------------------------------------------ public API

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        self._reject_empty(texts)
        self._reject_too_long(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):  # sequential => order preserved
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size], TASK_DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self._reject_empty([text])
        self._reject_too_long([text])
        return self._embed_batch([text], TASK_QUERY)[0]

    # ------------------------------------------------------------------ validation

    def _reject_empty(self, texts: list[str]) -> None:
        bad = [i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()]
        if bad:
            raise EmbeddingError(
                ErrorCode.EMPTY_TEXT,
                f"{len(bad)} text(s) are empty or whitespace-only (index {_preview(bad)}); "
                "the API would fail the whole batch or return a meaningless vector. "
                "Filter them out before embedding.",
            )

    def _reject_too_long(self, texts: list[str]) -> None:
        too_long: list[tuple[int, int]] = []
        for index, text in enumerate(texts):
            if len(text) <= EXACT_COUNT_MIN_CHARS:
                continue
            response = self._call_with_retry(
                lambda text=text: self._client.models.count_tokens(model=self._model, contents=text),
                "count_tokens",
            )
            tokens = response.total_tokens
            if tokens is None:
                raise EmbeddingError(ErrorCode.BAD_RESPONSE, "count_tokens returned no token count.")
            if tokens > MAX_INPUT_TOKENS:
                too_long.append((index, tokens))
        if too_long:
            listing = ", ".join(f"index {i} ({n} tokens)" for i, n in too_long[:5])
            raise EmbeddingError(
                ErrorCode.TEXT_TOO_LONG,
                f"{len(too_long)} text(s) exceed the {MAX_INPUT_TOKENS}-token limit: {listing}. "
                "The API would truncate them silently, so they are rejected; split the text.",
            )

    # ------------------------------------------------------------------ one batch

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim)
        started = time.perf_counter()
        response = self._call_with_retry(
            lambda: self._client.models.embed_content(model=self._model, contents=batch, config=config),
            "embed_content",
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                ErrorCode.BAD_RESPONSE,
                f"Sent {len(batch)} texts but received {len(embeddings)} embeddings.",
            )
        vectors = [self._checked(list(e.values or [])) for e in embeddings]
        logger.info(
            "Embedded batch: size=%d task=%s dim=%d latency=%.2fs",
            len(batch), task_type, self._dim, time.perf_counter() - started,
        )
        return vectors

    def _checked(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dim:
            raise EmbeddingError(
                ErrorCode.DIMENSION_MISMATCH,
                f"The model returned {len(vector)} dimensions but EMBEDDING_DIM={self._dim}. "
                "Check GEMINI_EMBEDDING_MODEL and EMBEDDING_DIM; changing either requires "
                "re-indexing every document.",
            )
        norm = math.sqrt(sum(x * x for x in vector))
        if not math.isfinite(norm) or norm == 0:
            raise EmbeddingError(ErrorCode.BAD_RESPONSE, "The model returned a zero or invalid vector.")
        return [x / norm for x in vector]  # truncated Matryoshka vectors are not unit length

    # ------------------------------------------------------------------ retry

    def _call_with_retry(self, call: Callable[[], T], operation: str) -> T:
        """Retry 429 / 5xx / network errors with exponential backoff + jitter.

        Everything else (400 invalid argument, 401/403 auth, 404 ...) fails immediately:
        sending the same bad request again cannot succeed and only burns quota.
        """
        attempt = 0
        while True:
            server_delay: float | None = None
            try:
                return call()
            except genai_errors.APIError as exc:
                if exc.code not in RETRYABLE_HTTP_CODES:
                    raise self._rejected(exc, operation) from exc
                failure = ErrorCode.RATE_LIMITED if exc.code == 429 else ErrorCode.SERVICE_UNAVAILABLE
                reason = f"HTTP {exc.code} {exc.status}"
                server_delay = _retry_delay_from(exc)
                error: Exception = exc
            except httpx.TransportError as exc:  # timeouts, connection resets, DNS...
                failure, reason, error = ErrorCode.SERVICE_UNAVAILABLE, type(exc).__name__, exc

            if attempt >= self._max_retries:
                hint = (
                    "Free-tier quota is about 100 texts per minute; wait a minute and retry."
                    if failure is ErrorCode.RATE_LIMITED
                    else "The service looks unavailable; try again later."
                )
                raise EmbeddingError(
                    failure, f"{operation} failed after {attempt} retries ({reason}). {hint}"
                ) from error

            backoff = min(self._max_delay, self._base_delay * 2**attempt)
            delay = backoff / 2 + self._jitter(0.0, backoff / 2)  # "equal jitter"
            if server_delay is not None:
                if server_delay > self._max_delay:
                    raise EmbeddingError(
                        failure,
                        f"{operation}: the server asks to wait {server_delay:.0f}s, longer than "
                        f"the {self._max_delay:.0f}s limit (a daily quota?). Not retrying.",
                    ) from error
                delay = max(delay, server_delay + self._jitter(0.0, 1.0))  # obey the server
            attempt += 1
            logger.warning(
                "%s: %s, retry %d/%d in %.1fs", operation, reason, attempt, self._max_retries, delay
            )
            self._sleep(delay)

    def _rejected(self, exc: genai_errors.APIError, operation: str) -> EmbeddingError:
        message = self._redact(str(exc.message or exc.status or ""))[:300]
        if exc.code in AUTH_HTTP_CODES:
            return EmbeddingError(
                ErrorCode.AUTH_ERROR,
                f"{operation}: Gemini rejected the credentials (HTTP {exc.code} {exc.status}): "
                f"{message} Check GEMINI_API_KEY.",
            )
        return EmbeddingError(
            ErrorCode.REQUEST_REJECTED,
            f"{operation}: Gemini rejected the request (HTTP {exc.code} {exc.status}): {message}",
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text


def _preview(indices: list[int], limit: int = 10) -> str:
    shown = ", ".join(map(str, indices[:limit]))
    return shown + (f", ... (+{len(indices) - limit} more)" if len(indices) > limit else "")
