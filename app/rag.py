"""RAG pipeline: embed question -> search -> score threshold -> prompt -> Gemini -> verified citations.

Design points (each one is something to be able to defend in an interview):
  * The score threshold is a cheap, deterministic gate BEFORE the LLM. If no chunk passes, we
    answer with a fixed fallback and never call the LLM (no cost, no chance to hallucinate).
    It is only a similarity gate: a chunk can pass and still not contain the answer, so the
    prompt also tells the model to say when the context is insufficient.
  * Filtering by threshold happens here, not inside Qdrant, so that on a fallback we still know
    the best score. That number is what you need to tune SCORE_THRESHOLD.
  * The model's [n] citations are parsed and checked against the chunks we really sent. A number
    that does not exist is removed from the text and reported; a source is never invented.
  * LLM failures become RagError with a stable `code`. Embedding and vector-store failures keep
    their own exception types (EmbeddingError, VectorStoreError) and are not re-wrapped.
"""

import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.embeddings import AUTH_HTTP_CODES, RETRYABLE_HTTP_CODES, Embedder, _retry_delay_from
from app.prompts import FALLBACK_MESSAGE, RAG_SYSTEM_PROMPT, build_user_prompt, new_nonce
from app.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_QUESTION_CHARS = 2000
EXCERPT_CHARS = 200
DEFAULT_TEMPERATURE = 0.2
# Thinking models count their hidden reasoning inside max_output_tokens, so this must be generous.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_S = 60.0

# Citations look like [3], [1][2] or [1, 2]. At most 3 digits so a year like [2024] is left alone.
_CITATION = re.compile(r"(\s*)\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")

_BLOCKING_FINISH_REASONS = frozenset({
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
})


class RagErrorCode(StrEnum):
    NOT_CONFIGURED = "not_configured"
    INVALID_INPUT = "invalid_input"
    AUTH_ERROR = "auth_error"  # 401/403
    REQUEST_REJECTED = "request_rejected"  # 400/404/...: retrying cannot help
    RATE_LIMITED = "rate_limited"  # 429 and retries exhausted
    SERVICE_UNAVAILABLE = "service_unavailable"  # 5xx / network
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # safety filters
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED = "truncated"  # hit max_output_tokens before producing any text


class RagError(Exception):
    """Expected pipeline failure with a stable `code` and a readable `message`."""

    def __init__(self, code: RagErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------- result types


@dataclass(frozen=True, slots=True)
class Source:
    """A chunk that was sent to the LLM. `cited` = the answer really referred to it as [number]."""

    number: int  # the [n] used in the prompt and in the answer, starting at 1
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float
    excerpt: str
    cited: bool


@dataclass(frozen=True, slots=True)
class Latency:
    embed_ms: float
    search_ms: float
    llm_ms: float  # 0.0 on fallback
    total_ms: float


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str
    sources: list[Source]  # chunks sent to the LLM (empty on fallback)
    used_fallback: bool  # True = nothing passed the threshold and the LLM was NOT called
    latency: Latency
    best_score: float | None  # top retrieved score even if below the threshold (None = no hits)
    invalid_citations: tuple[int, ...] = ()  # [n] the model wrote that match no source (removed)
    truncated: bool = False  # the LLM stopped at max_output_tokens, so the answer may be cut off


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    truncated: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None


class LLM(Protocol):
    def generate(
        self,
        system_instruction: str,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """One single-turn completion. Raises RagError on any expected failure.

        `response_schema` (a JSON schema) asks the model for JSON that follows it; callers must still validate
        the text, a schema is a request and not a guarantee. `max_output_tokens` overrides the default for one call.
        """
        ...


class SearchStore(Protocol):
    """The part of QdrantStore the pipeline needs."""

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------- Gemini LLM


class GeminiLLM:
    """Gemini text generation with timeout, retry/backoff (429/5xx only) and error mapping."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        thinking_level: str | None = None,
        max_retries: int | None = None,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not settings.gemini_llm_model:
            raise RagError(RagErrorCode.NOT_CONFIGURED, "GEMINI_LLM_MODEL must be set in .env.")
        self._model: str = settings.gemini_llm_model
        level = thinking_level or settings.llm_thinking_level
        self._thinking_level = types.ThinkingLevel[level.upper()] if level else None
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._max_retries = settings.embed_max_retries if max_retries is None else max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._secret = settings.gemini_api_key.get_secret_value()
        self._client = client or genai.Client(api_key=self._secret)

    def generate(
        self,
        system_instruction: str,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._temperature,
            max_output_tokens=max_output_tokens or self._max_output_tokens,
            response_mime_type="application/json" if response_schema else None,
            response_json_schema=response_schema,
            # We pass no tools; make sure the SDK never starts a function-calling loop.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=self._thinking_level)
                if self._thinking_level
                else None
            ),
            http_options=types.HttpOptions(timeout=int(self._timeout_s * 1000)),  # milliseconds
        )
        response = self._call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        )
        return self._interpret(response)

    # ------------------------------------------------------------------ response

    def _interpret(self, response: Any) -> LLMResult:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            raise RagError(
                RagErrorCode.BLOCKED,
                f"Gemini blocked the request ({getattr(block_reason, 'name', block_reason)}). "
                "Try rephrasing the question.",
            )
        candidates = response.candidates or []
        finish = candidates[0].finish_reason if candidates else None
        if finish in _BLOCKING_FINISH_REASONS:
            raise RagError(
                RagErrorCode.BLOCKED,
                f"Gemini stopped the answer ({finish.name}) because of its safety filters.",
            )
        text = (response.text or "").strip()
        truncated = finish == types.FinishReason.MAX_TOKENS
        if not text:
            if truncated:
                raise RagError(
                    RagErrorCode.TRUNCATED,
                    f"The model used all {self._max_output_tokens} output tokens (thinking "
                    "included) before writing an answer. Lower LLM_THINKING_LEVEL or raise "
                    "max_output_tokens.",
                )
            raise RagError(
                RagErrorCode.EMPTY_RESPONSE,
                f"Gemini returned an empty answer (finish reason: {getattr(finish, 'name', finish)}).",
            )
        usage = getattr(response, "usage_metadata", None)
        if truncated:
            logger.warning("LLM answer was cut at max_output_tokens=%d", self._max_output_tokens)
        return LLMResult(
            text=text,
            truncated=truncated,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
        )

    # ------------------------------------------------------------------ retry

    def _call_with_retry(self, call: Callable[[], T]) -> T:
        """Retry 429 and 5xx with exponential backoff + jitter, obeying the server's retryDelay.

        Same policy as GeminiEmbedder. Timeouts are NOT retried: one attempt already waited the
        whole timeout, and repeating it would block the caller for minutes.
        """
        attempt = 0
        while True:
            try:
                return call()
            except genai_errors.APIError as exc:
                if exc.code not in RETRYABLE_HTTP_CODES:
                    raise self._rejected(exc) from exc
                failure = (
                    RagErrorCode.RATE_LIMITED if exc.code == 429 else RagErrorCode.SERVICE_UNAVAILABLE
                )
                reason = f"HTTP {exc.code} {exc.status}"
                server_delay = _retry_delay_from(exc)
                error: Exception = exc
            except httpx.TimeoutException as exc:
                raise RagError(
                    RagErrorCode.TIMEOUT,
                    f"Gemini did not answer within {self._timeout_s:g}s. Try again, or use a "
                    "faster model / lower LLM_THINKING_LEVEL.",
                ) from exc
            except httpx.TransportError as exc:
                raise RagError(
                    RagErrorCode.SERVICE_UNAVAILABLE,
                    f"Cannot reach Gemini ({type(exc).__name__}). Check the network connection.",
                ) from exc

            if attempt >= self._max_retries:
                hint = (
                    "Rate limit reached (free tier); wait a minute and try again."
                    if failure is RagErrorCode.RATE_LIMITED
                    else "The Gemini service looks unavailable; try again later."
                )
                raise RagError(
                    failure, f"LLM call failed after {attempt} retries ({reason}). {hint}"
                ) from error

            backoff = min(self._max_delay, self._base_delay * 2**attempt)
            delay = backoff / 2 + self._jitter(0.0, backoff / 2)  # "equal jitter"
            if server_delay is not None:
                if server_delay > self._max_delay:
                    raise RagError(
                        failure,
                        f"The server asks to wait {server_delay:.0f}s, longer than the "
                        f"{self._max_delay:.0f}s limit (a daily quota?). Not retrying.",
                    ) from error
                delay = max(delay, server_delay + self._jitter(0.0, 1.0))
            attempt += 1
            logger.warning("LLM call: %s, retry %d/%d in %.1fs", reason, attempt, self._max_retries, delay)
            self._sleep(delay)

    def _rejected(self, exc: genai_errors.APIError) -> RagError:
        message = self._redact(str(exc.message or exc.status or ""))[:300]
        # Measured: a wrong key comes back as 400 INVALID_ARGUMENT "API key not valid", not 401/403.
        bad_key = exc.code == 400 and "api key" in message.lower()
        if exc.code in AUTH_HTTP_CODES or bad_key:
            return RagError(
                RagErrorCode.AUTH_ERROR,
                f"Gemini rejected the credentials (HTTP {exc.code} {exc.status}): {message} "
                "Check GEMINI_API_KEY.",
            )
        return RagError(
            RagErrorCode.REQUEST_REJECTED,
            f"Gemini rejected the request (HTTP {exc.code} {exc.status}): {message} "
            f"(model: {self._model})",
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text


# ---------------------------------------------------------------------- citations


def resolve_citations(answer: str, source_count: int) -> tuple[str, set[int], list[int]]:
    """Check the [n] markers of `answer` against `source_count` real sources (numbered 1..N).

    Returns (answer without invalid markers, valid numbers cited, invalid numbers found).
    "[1, 3]" is normalized to "[1][3]". A marker with only invalid numbers is removed together
    with the space before it. Never adds a citation.
    """
    cited: set[int] = set()
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.split(r"\s*[,;]\s*", match.group(2))]
        valid = [n for n in dict.fromkeys(numbers) if 1 <= n <= source_count]
        invalid.extend(n for n in numbers if not 1 <= n <= source_count)
        cited.update(valid)
        return match.group(1) + "".join(f"[{n}]" for n in valid) if valid else ""

    return _CITATION.sub(replace, answer), cited, invalid


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:EXCERPT_CHARS] + ("…" if len(flat) > EXCERPT_CHARS else "")


# ---------------------------------------------------------------------- pipeline


class RagPipeline:
    """Single-turn question answering over the indexed documents (no conversation memory yet)."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder,
        store: SearchStore,
        llm: LLM,
        score_threshold: float | None = None,
    ) -> None:
        threshold = score_threshold if score_threshold is not None else settings.score_threshold
        if threshold is None:
            raise RagError(
                RagErrorCode.NOT_CONFIGURED,
                "SCORE_THRESHOLD must be set in .env (cosine similarity, e.g. 0.65). "
                "Without it the 'insufficient evidence' fallback could never trigger.",
            )
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self.score_threshold: float = threshold

    def answer(
        self,
        question: str,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> RagAnswer:
        """doc_ids=None searches all documents; an empty list searches none (-> fallback)."""
        question = question.strip()
        if not question:
            raise RagError(RagErrorCode.INVALID_INPUT, "The question is empty.")
        if len(question) > MAX_QUESTION_CHARS:
            raise RagError(
                RagErrorCode.INVALID_INPUT,
                f"The question has {len(question)} characters; the limit is {MAX_QUESTION_CHARS}.",
            )
        k = self._settings.top_k if top_k is None else top_k
        if k < 1:
            raise RagError(RagErrorCode.INVALID_INPUT, "top_k must be at least 1.")

        started = time.perf_counter()
        query_vector = self._embedder.embed_query(question)
        embedded = time.perf_counter()
        # No score_threshold here on purpose: we filter below so the best score stays visible.
        hits = self._store.search(query_vector, k, doc_ids=doc_ids)
        searched = time.perf_counter()

        best_score = max((hit.score for hit in hits), default=None)
        passing = [hit for hit in hits if hit.score >= self.score_threshold]
        embed_ms = (embedded - started) * 1000
        search_ms = (searched - embedded) * 1000

        if not passing:
            logger.info(
                "RAG fallback: %d hits, best_score=%s < threshold=%.3f, LLM not called",
                len(hits), "none" if best_score is None else f"{best_score:.4f}", self.score_threshold,
            )
            total_ms = (time.perf_counter() - started) * 1000
            return RagAnswer(
                answer=FALLBACK_MESSAGE,
                sources=[],
                used_fallback=True,
                latency=Latency(embed_ms, search_ms, 0.0, total_ms),
                best_score=best_score,
            )

        prompt = build_user_prompt(question, passing, nonce=new_nonce())
        llm_started = time.perf_counter()
        result = self._llm.generate(RAG_SYSTEM_PROMPT, prompt)
        llm_ms = (time.perf_counter() - llm_started) * 1000

        text, cited, invalid = resolve_citations(result.text, len(passing))
        if invalid:
            logger.warning("LLM cited sources that do not exist: %s (removed from the answer)", invalid)
        if not cited:
            logger.warning("LLM answer has no valid citation")
        sources = [
            Source(
                number=number,
                doc_id=hit.doc_id,
                filename=hit.filename,
                page=hit.page,
                chunk_index=hit.chunk_index,
                score=hit.score,
                excerpt=_excerpt(hit.text),
                cited=number in cited,
            )
            for number, hit in enumerate(passing, start=1)
        ]
        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "RAG answered: sources=%d cited=%d best_score=%.4f embed=%.0fms search=%.0fms llm=%.0fms "
            "tokens(in/out/thinking)=%s/%s/%s",
            len(sources), len(cited), best_score or 0.0, embed_ms, search_ms, llm_ms,
            result.input_tokens, result.output_tokens, result.thinking_tokens,
        )
        return RagAnswer(
            answer=text,
            sources=sources,
            used_fallback=False,
            latency=Latency(embed_ms, search_ms, llm_ms, total_ms),
            best_score=best_score,
            invalid_citations=tuple(dict.fromkeys(invalid)),
            truncated=result.truncated,
        )
