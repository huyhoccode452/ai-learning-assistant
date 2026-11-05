"""Multiple-choice quiz from one document. The model proposes; the CODE decides what is kept.

Pipeline (one LLM call to generate, one optional call to verify, both batched):
  1. pick up to MAX_QUIZ_CHUNKS chunks spread over the document (page range / topic respected);
  2. ask for a few more questions than requested (OVERSAMPLE) as JSON that follows QUIZ_SCHEMA;
  3. validate every question with Pydantic (4 distinct options, valid key, explanation, quote present);
  4. grounding in code: the quote must really occur in the chunk the question cites (after NFC + whitespace
     normalisation); the page always comes from that chunk, never from the model; questions whose options refer to
     other options ("cả A và B") are dropped because the options are shuffled later;
  5. optional verifier call: is the key supported by its quote, and is no other option also correct?
  6. keep at most `num_questions`; shuffle the options in code with balanced positions and recompute the key.
Nothing is invented to reach the requested number: fewer valid questions are returned with an explanation.
LLM calls: 1 generation (+ 1 verification when verify=True) = 1 or 2 per request, 0 when cached.
"""

import json
import logging
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import Settings
from app.db import Database
from app.embeddings import Embedder
from app.generation import (
    CountingLLM,
    GenerationError,
    GenerationErrorCode,
    GenResult,
    content_hash,
    llm_identity,
    normalize_text,
    run_cached,
)
from app.prompts import (
    QUIZ_PROMPT_VERSION,
    QUIZ_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_quiz_prompt,
    build_source_block,
    build_text_block,
    build_verify_prompt,
    new_nonce,
)
from app.rag import LLM, RagError
from app.vectorstore import QdrantStore, RetrievedChunk

logger = logging.getLogger(__name__)

MIN_QUOTE_CHARS = 15
MAX_QUIZ_CHUNKS = 12  # bounded prompt: at most this many chunks per generation call
MIN_CHUNK_CHARS = 150  # shorter chunks (titles, headings) are not used as sources unless nothing else is left
OVERSAMPLE = 2  # ask for this many extra questions: some will be dropped by the checks
MAX_ASKED = 12
TOPIC_TOP_K = 12
QUIZ_MAX_OUTPUT_TOKENS = 8192
OPTION_COUNT = 4

QUIZ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": OPTION_COUNT, "maxItems": OPTION_COUNT},
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": OPTION_COUNT - 1},
                    "explanation": {"type": "string"},
                    "source_chunk": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["question", "options", "answer_index", "explanation", "source_chunk", "quote"],
            },
        }
    },
    "required": ["questions"],
}
VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "key_supported": {"type": "boolean"},
                    "another_option_correct": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "key_supported", "another_option_correct", "reason"],
            },
        }
    },
    "required": ["results"],
}

DROP_REASONS_VI = {
    "schema_invalid": "sai định dạng",
    "duplicate_options": "đáp án trùng nhau",
    "positional_option": "đáp án tham chiếu đáp án khác",
    "unknown_source": "dẫn nguồn không có trong các đoạn đã gửi",
    "quote_not_in_source": "trích dẫn không có nguyên văn trong đoạn nguồn",
    "duplicate_question": "câu hỏi trùng",
    "verifier_key_not_supported": "bộ xác minh cho rằng đáp án chưa được nguồn chứng minh",
    "verifier_ambiguous": "bộ xác minh cho rằng có đáp án khác cũng đúng",
}

# "A.", "B)", "(c)" at the start of an option: the model was told not to, we strip it anyway.
_LABEL = re.compile(r"^\s*(?:\(?[A-Da-d][\.\):]|\([A-Da-d]\))\s+")
# Options must stay meaningful after shuffling, so options that point at other options are rejected.
_POSITIONAL = re.compile(
    r"(?i:(tất cả|toàn bộ|không có)\s+(các\s+)?(đáp án|phương án|lựa chọn)|(đáp án|phương án|lựa chọn)\s+[a-d]\b"
    r"|(all|none)\s+of\s+the\s+above|both\s+[a-d]\b)|\bcả\s+[A-D]\b|\b[A-D]\s+(và|hoặc)\s+[A-D]\b"
)
_QUOTE_EDGES = "\"'“”‘’«»`"


# ====================================================================== validation


class RawQuestion(BaseModel):
    """One question as the model wrote it. Anything that does not fit raises ValidationError."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=8)
    options: list[str] = Field(min_length=OPTION_COUNT, max_length=OPTION_COUNT)
    answer_index: int = Field(ge=0, le=OPTION_COUNT - 1)
    explanation: str = Field(min_length=1)
    source_chunk: int
    quote: str = Field(min_length=MIN_QUOTE_CHARS)

    @field_validator("question", "explanation", "quote", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("source_chunk", mode="before")
    @classmethod
    def _chunk_id(cls, value: Any) -> Any:
        return value.strip().lstrip("#") if isinstance(value, str) else value

    @field_validator("options", mode="before")
    @classmethod
    def _clean_options(cls, value: Any) -> Any:
        if isinstance(value, list) and all(isinstance(o, str) for o in value):
            return [_LABEL.sub("", o).strip() for o in value]
        return value

    @field_validator("options")
    @classmethod
    def _check_options(cls, value: list[str]) -> list[str]:
        if any(not option for option in value):
            raise ValueError("schema_invalid: empty option")
        if len({normalize_text(option).casefold() for option in value}) != len(value):
            raise ValueError("duplicate_options")
        if any(_POSITIONAL.search(option) for option in value):
            raise ValueError("positional_option")
        return value


def _reason(error: ValidationError) -> str:
    text = " ".join(item["msg"] for item in error.errors())
    for tag in ("duplicate_options", "positional_option"):
        if tag in text:
            return tag
    return "schema_invalid"


def parse_questions(text: str) -> tuple[list[RawQuestion], Counter[str]]:
    """Strict parse of the model's JSON. Malformed JSON raises GenerationError; an invalid single question is
    dropped (and counted) without losing the others."""
    cleaned = text.strip()
    if cleaned.startswith("```"):  # a fenced block, in case the model ignored the JSON mode
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise GenerationError(GenerationErrorCode.MALFORMED_OUTPUT, "The quiz reply is not valid JSON.") from exc
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise GenerationError(GenerationErrorCode.MALFORMED_OUTPUT, "The quiz reply has no 'questions' list.")
    kept: list[RawQuestion] = []
    dropped: Counter[str] = Counter()
    for item in items:
        try:
            kept.append(RawQuestion.model_validate(item))
        except ValidationError as exc:
            dropped[_reason(exc)] += 1
    return kept, dropped


# ====================================================================== grounding


def clean_quote(quote: str) -> str:
    """Remove the quotation marks and ellipses a model likes to put around a quote (they are not in the document)."""
    text = quote.strip()
    for _ in range(3):
        text = text.strip(_QUOTE_EDGES + " ").removeprefix("…").removeprefix("...").removesuffix("…").removesuffix("...")
    return text.strip(_QUOTE_EDGES + " ")


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote occurs verbatim in the text, ignoring only whitespace layout and Unicode composition."""
    needle = normalize_text(clean_quote(quote))
    return len(needle) >= MIN_QUOTE_CHARS and needle in normalize_text(text)


def ground(raw: Sequence[RawQuestion], chunks: dict[int, RetrievedChunk]) -> tuple[list[tuple[RawQuestion, RetrievedChunk]], Counter[str]]:
    """Keep questions whose cited chunk was really sent and really contains the quote; drop repeated stems."""
    kept: list[tuple[RawQuestion, RetrievedChunk]] = []
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    for question in raw:
        chunk = chunks.get(question.source_chunk)
        if chunk is None:
            dropped["unknown_source"] += 1
        elif not quote_in_text(question.quote, chunk.text):
            dropped["quote_not_in_source"] += 1
        elif (stem := normalize_text(question.question).casefold()) in seen:
            dropped["duplicate_question"] += 1
        else:
            seen.add(stem)
            kept.append((question, chunk))
    return kept, dropped


# ====================================================================== option positions (position bias)


def assign_targets(count: int, rng: random.Random) -> list[int]:
    """The position of the correct option for each question: shuffled permutations of 0..3 chained together.
    Every block of four questions uses all four positions once, so with 2 or more questions the key is never
    always in the same place."""
    targets: list[int] = []
    while len(targets) < count:
        block = list(range(OPTION_COUNT))
        rng.shuffle(block)
        targets.extend(block)
    return targets[:count]


def place_correct(options: Sequence[str], answer_index: int, target: int, rng: random.Random) -> tuple[list[str], int]:
    """Options reordered so that the correct one sits at `target`; the others are shuffled. Returns (options, key)."""
    others = [option for i, option in enumerate(options) if i != answer_index]
    rng.shuffle(others)
    return others[:target] + [options[answer_index]] + others[target:], target


# ====================================================================== chunk selection


def select_chunks(
    chunks: Sequence[RetrievedChunk], count: int, rng: random.Random, *, page_from: int | None = None,
    page_to: int | None = None, preferred: set[int] | None = None,
) -> list[RetrievedChunk]:
    """Up to `count` chunks spread evenly over the candidates (one random pick inside each of `count` equal
    slices), so the questions cover the whole document instead of its first pages."""
    pool = list(chunks)
    if page_from is not None or page_to is not None:
        pool = [c for c in pool if c.page is not None and (page_from or 1) <= c.page <= (page_to or c.page)]
    if preferred is not None:
        pool = [c for c in pool if c.chunk_index in preferred]
    if not pool:
        raise GenerationError(GenerationErrorCode.EMPTY_SELECTION, "No chunk matches the page range or topic.")
    substantial = [c for c in pool if len(c.text) >= MIN_CHUNK_CHARS]
    pool = sorted(substantial or pool, key=lambda c: c.chunk_index)
    if len(pool) <= count:
        return pool
    # Slice i is pool[i*L//count : (i+1)*L//count]: disjoint and never empty when L >= count, so no chunk is picked twice
    # (a float formula such as int((i + r) * L / count) can pick the same index in two neighbouring slices).
    bounds = [i * len(pool) // count for i in range(count + 1)]
    return [pool[low + int(rng.random() * (high - low))] for low, high in zip(bounds, bounds[1:])]


# ====================================================================== verifier


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    key_supported: bool
    another_option_correct: bool
    reason: str = ""


def _verify_block(items: list[tuple[RawQuestion, RetrievedChunk]], nonce: str) -> str:
    parts = []
    for number, (q, chunk) in enumerate(items, start=1):
        options = "\n".join(f"{'ABCD'[i]}) {option}" for i, option in enumerate(q.options))
        page = f"trang {chunk.page}" if chunk.page is not None else "không có số trang"
        parts.append(f"Question {number}\nQ: {q.question}\n{options}\nClaimed key: {'ABCD'[q.answer_index]}\n"
                     f"Evidence quote: {clean_quote(q.quote)}\nEvidence chunk (#{chunk.chunk_index}, {page}):\n{chunk.text}")
    return build_text_block("\n\n---\n\n".join(parts), nonce, "QUESTIONS_AND_EVIDENCE")


def verify_questions(
    items: list[tuple[RawQuestion, RetrievedChunk]], llm: LLM
) -> tuple[list[Verdict | None], str | None]:
    """ONE batched call. Returns a verdict per question (None where the verifier said nothing) and a warning text
    when the verifier could not be used at all. Never raises for LLM or format problems: the caller keeps the
    questions but marks them unverified."""
    prompt = build_verify_prompt(_verify_block(items, new_nonce()))
    try:
        reply = llm.generate(VERIFY_SYSTEM_PROMPT, prompt, response_schema=VERIFY_SCHEMA, max_output_tokens=QUIZ_MAX_OUTPUT_TOKENS)
        data = json.loads(reply.text)
        verdicts = [Verdict.model_validate(entry) for entry in data["results"]]
    except RagError as exc:
        logger.warning("Verifier call failed: %s", exc.code)
        return [None] * len(items), f"Không chạy được bước xác minh ({exc.message})"
    except (ValueError, KeyError, TypeError, ValidationError):
        logger.warning("Verifier reply was not usable")
        return [None] * len(items), "Bước xác minh trả về kết quả không đọc được"
    by_id = {v.id: v for v in verdicts}
    return [by_id.get(number) for number in range(1, len(items) + 1)], None


# ====================================================================== the pipeline


def _message(requested: int, kept: int, dropped: Counter[str]) -> str | None:
    if kept >= requested:
        return None
    reasons = ", ".join(f"{count} câu {DROP_REASONS_VI.get(reason, reason)}" for reason, count in dropped.most_common())
    removed = f" (đã loại: {reasons})" if reasons else ""
    if kept == 0:
        return f"Không tạo được câu hỏi hợp lệ nào từ phần tài liệu này{removed}. Hãy thử lại hoặc chọn phạm vi khác."
    return f"Chỉ tạo được {kept}/{requested} câu hỏi hợp lệ{removed}. Hệ thống không bịa thêm câu hỏi cho đủ số lượng."


def build_quiz(
    chunks: Sequence[RetrievedChunk], llm: LLM, *, num_questions: int, verify: bool, rng: random.Random,
    topic: str | None = None, page_from: int | None = None, page_to: int | None = None,
    preferred: set[int] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Returns (payload, cacheable). A quiz is cacheable only if it has questions and, when verification was
    requested, every question went through the verifier."""
    ask = min(num_questions + OVERSAMPLE, MAX_ASKED)
    chosen = select_chunks(chunks, min(MAX_QUIZ_CHUNKS, max(ask, 4)), rng, page_from=page_from, page_to=page_to, preferred=preferred)
    by_id = {c.chunk_index: c for c in chosen}
    reply = llm.generate(QUIZ_SYSTEM_PROMPT, build_quiz_prompt(build_source_block(chosen, new_nonce()), ask, topic),
                         response_schema=QUIZ_SCHEMA, max_output_tokens=QUIZ_MAX_OUTPUT_TOKENS)
    raw, dropped = parse_questions(reply.text)
    asked_returned = len(raw) + sum(dropped.values())
    grounded, grounding_dropped = ground(raw, by_id)
    dropped.update(grounding_dropped)
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    verified_all = False

    if verify and grounded:
        verdicts, problem = verify_questions(grounded, llm)
        if problem:
            warnings.append(problem + "; các câu hỏi chưa được xác minh.")
        survivors = []
        for (question, chunk), verdict in zip(grounded, verdicts):
            if verdict is None:
                survivors.append((question, chunk, None))
                continue
            details.append({"question": question.question, "key_supported": verdict.key_supported,
                            "another_option_correct": verdict.another_option_correct, "reason": verdict.reason,
                            "kept": verdict.key_supported and not verdict.another_option_correct})
            if not verdict.key_supported:
                dropped["verifier_key_not_supported"] += 1
            elif verdict.another_option_correct:
                dropped["verifier_ambiguous"] += 1
            else:
                survivors.append((question, chunk, True))
        verified_all = all(flag is True for _, _, flag in survivors) and not problem
    else:
        survivors = [(q, c, None) for q, c in grounded]

    survivors = survivors[:num_questions]
    returned_stems = {question.question for question, _, _ in survivors}
    for decision in details:  # the verifier judged every candidate; only some of them end up in the quiz
        decision["in_quiz"] = decision["question"] in returned_stems
    targets = assign_targets(len(survivors), rng)
    questions = []
    for number, ((question, chunk, flag), target) in enumerate(zip(survivors, targets), start=1):
        options, key = place_correct(question.options, question.answer_index, target, rng)
        questions.append({
            "id": number, "question": question.question, "options": options, "answer_index": key,
            "explanation": question.explanation,
            "source": {"chunk_index": chunk.chunk_index, "page": chunk.page, "filename": chunk.filename,
                       "quote": clean_quote(question.quote)},
            "verified": flag,
        })
    payload = {
        "questions": questions, "requested": num_questions, "asked_of_model": ask, "returned_by_model": asked_returned,
        "dropped": dict(dropped), "verify_requested": verify, "verified": bool(verify and questions and verified_all),
        "verifier_decisions": details, "warnings": warnings, "message": _message(num_questions, len(questions), dropped),
        "topic": topic, "page_from": page_from, "page_to": page_to,
        "chunks_used": [{"chunk_index": c.chunk_index, "page": c.page} for c in chosen],
    }
    cacheable = bool(questions) and (not verify or payload["verified"])
    return payload, cacheable


def generate_quiz(
    db: Database, store: QdrantStore, embedder: Embedder, llm: LLM, settings: Settings, doc_id: str, *,
    num_questions: int = 5, topic: str | None = None, page_from: int | None = None, page_to: int | None = None,
    verify: bool = True, refresh: bool = False, rng: random.Random | None = None,
) -> GenResult:
    chunks = store.get_chunks(doc_id)
    if not chunks:
        raise GenerationError(GenerationErrorCode.NO_CONTENT, "The index has no text for this document.")
    topic = unicodedata.normalize("NFC", topic).strip() if topic and topic.strip() else None
    rng = rng or random.Random()

    def compute(counting: CountingLLM) -> tuple[dict[str, Any], bool]:
        preferred = None
        if topic:  # a semantic search for the topic picks the candidate chunks (embedding call, not an LLM call)
            hits = store.search(embedder.embed_query(topic), TOPIC_TOP_K, doc_ids=[doc_id])
            preferred = {hit.chunk_index for hit in hits}
        return build_quiz(chunks, counting, num_questions=num_questions, verify=verify, rng=rng, topic=topic,
                          page_from=page_from, page_to=page_to, preferred=preferred)

    return run_cached(
        db, llm, kind="quiz", doc_id=doc_id, text_hash=content_hash(chunks),
        params={"n": num_questions, "topic": topic, "pages": [page_from, page_to], "verify": verify},
        model=llm_identity(settings), prompt_version=QUIZ_PROMPT_VERSION, refresh=refresh, compute=compute,
    )
