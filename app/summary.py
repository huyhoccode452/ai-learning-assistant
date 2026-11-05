"""Summary of one document in Vietnamese: ONE LLM call when the text fits, otherwise map-reduce with a hard cap.

Budgets are in characters, not tokens: counting tokens would cost a request, and the real limits are speed and
free-tier quota rather than the context window (a few hundred thousand characters fit in Gemini's window).
Vietnamese runs at about 4 characters per token (KI-16, measured on one phrase), so the default single-call
budget of 60 000 characters is roughly 15 000 to 25 000 tokens.

LLM calls per request:
  * fits the single-call budget            -> 1
  * longer                                  -> one call per group of chunks + 1 merge call (at most MAX_GROUPS + 1)
  * longer than MAX_GROUPS groups           -> only the first MAX_GROUPS groups are used; the answer says so
The chunks are read from Qdrant (QdrantStore.get_chunks); consecutive chunks of one page overlap by a few
characters (chunk_overlap), so the budget counts a little duplicated text. That is accepted.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.config import Settings
from app.db import Database
from app.generation import GenerationError, GenerationErrorCode, GenResult, content_hash, llm_identity, run_cached
from app.prompts import (
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_TASK_DETAILED,
    SUMMARY_TASK_MAP,
    SUMMARY_TASK_REDUCE,
    SUMMARY_TASK_SHORT,
    build_source_block,
    build_summary_prompt,
    build_text_block,
    new_nonce,
)
from app.rag import LLM
from app.vectorstore import QdrantStore, RetrievedChunk

logger = logging.getLogger(__name__)

HEADER_OVERHEAD_CHARS = 30  # the "[#12] (trang 5)" line printed before each chunk


class SummaryStyle(StrEnum):
    SHORT = "short"
    DETAILED = "detailed"


@dataclass(frozen=True, slots=True)
class Limits:
    single_call_chars: int = 60_000  # whole document in ONE call up to this size
    group_chars: int = 40_000  # size of one map group
    max_groups: int = 6  # hard cap: at most max_groups + 1 calls, anything beyond is left out (and reported)


DEFAULT_LIMITS = Limits()


def _size(chunk: RetrievedChunk) -> int:
    return len(chunk.text) + HEADER_OVERHEAD_CHARS


def plan_groups(chunks: list[RetrievedChunk], group_chars: int, max_groups: int) -> list[list[RetrievedChunk]]:
    """Consecutive chunks packed greedily into groups of at most `group_chars`; stops after `max_groups`.

    A chunk larger than the limit gets a group of its own. Chunks that do not fit in the last allowed group are
    simply not returned (the caller reports the truncation).
    """
    groups: list[list[RetrievedChunk]] = []
    current: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        if current and used + _size(chunk) > group_chars:
            groups.append(current)
            if len(groups) == max_groups:
                return groups
            current, used = [], 0
        current.append(chunk)
        used += _size(chunk)
    if current:
        groups.append(current)
    return groups


def _page_span(chunks: list[RetrievedChunk]) -> str:
    pages = sorted({c.page for c in chunks if c.page is not None})
    return f"trang {pages[0]}-{pages[-1]}" if len(pages) > 1 else (f"trang {pages[0]}" if pages else "")


def build_summary(chunks: list[RetrievedChunk], llm: LLM, style: SummaryStyle, limits: Limits) -> dict[str, Any]:
    """Pure generation (no cache): returns the payload that is stored and sent to the client."""
    if not chunks:
        raise GenerationError(GenerationErrorCode.NO_CONTENT, "The index has no text for this document.")
    task = SUMMARY_TASK_SHORT if style is SummaryStyle.SHORT else SUMMARY_TASK_DETAILED
    total_chars = sum(_size(c) for c in chunks)
    groups = [chunks] if total_chars <= limits.single_call_chars else plan_groups(
        chunks, limits.group_chars, limits.max_groups)
    used = [chunk for group in groups for chunk in group]
    truncated = len(used) < len(chunks)
    warning = None
    if truncated:
        left_out = chunks[len(used):]
        warning = (f"Tài liệu quá dài ({len(chunks)} đoạn). Chỉ {len(used)} đoạn đầu ({_page_span(used) or 'phần đầu'}) "
                   f"được tóm tắt; {len(left_out)} đoạn cuối ({_page_span(left_out) or 'phần cuối'}) không được đưa vào.")
        logger.warning("Summary truncated: %d of %d chunks used", len(used), len(chunks))

    if len(groups) == 1:
        text = llm.generate(SUMMARY_SYSTEM_PROMPT, build_summary_prompt(task, build_source_block(groups[0], new_nonce()))).text
    else:
        partials = []
        for part, group in enumerate(groups, start=1):
            prompt = build_summary_prompt(SUMMARY_TASK_MAP.format(part=part, parts=len(groups)),
                                          build_source_block(group, new_nonce()))
            partials.append(f"### Phần {part}/{len(groups)} ({_page_span(group) or 'không có số trang'})\n"
                            + llm.generate(SUMMARY_SYSTEM_PROMPT, prompt).text)
        merged = build_text_block("\n\n".join(partials), new_nonce(), "PARTIAL_SUMMARIES")
        text = llm.generate(SUMMARY_SYSTEM_PROMPT,
                            build_summary_prompt(SUMMARY_TASK_REDUCE.format(task=task), merged)).text
    return {"summary": text.strip(), "style": style.value, "num_chunks_used": len(used),
            "num_chunks_total": len(chunks), "truncated": truncated, "warning": warning}


def summarize_document(
    db: Database, store: QdrantStore, llm: LLM, settings: Settings, doc_id: str, style: SummaryStyle,
    *, refresh: bool = False, limits: Limits = DEFAULT_LIMITS,
) -> GenResult:
    chunks = store.get_chunks(doc_id)
    if not chunks:
        raise GenerationError(GenerationErrorCode.NO_CONTENT, "The index has no text for this document.")
    return run_cached(
        db, llm, kind="summary", doc_id=doc_id, text_hash=content_hash(chunks),
        params={"style": style.value, "limits": [limits.single_call_chars, limits.group_chars, limits.max_groups]},
        model=llm_identity(settings), prompt_version=SUMMARY_PROMPT_VERSION, refresh=refresh,
        compute=lambda counting: (build_summary(chunks, counting, style, limits), True),
    )
