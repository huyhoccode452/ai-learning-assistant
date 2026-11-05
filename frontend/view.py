"""Pure presentation helpers for the Streamlit app (no Streamlit import, so they are unit-tested directly)."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from frontend.api_client import Document, Latency, Quiz, QuizQuestion, Source, Summary

# Document polling is bounded: the UI never waits forever for a free-tier embedding job.
SLOW_AFTER_S = 30.0  # show the "free tier is slow" hint
STOP_AFTER_S = 180.0  # stop polling automatically; the user can press "check again"
POLL_INTERVAL_S = 2.0

STATUS_BADGES = {
    "processing": ":orange-badge[Đang xử lý]",
    "ready": ":green-badge[Sẵn sàng]",
    "failed": ":red-badge[Lỗi]",
}

# Characters that Streamlit's markdown would interpret. Escaping them makes document text show literally.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]#>~|$<:!])")
_LIST_MARKER = re.compile(r"^(\s*)(?:([-+])|(\d+)([.)]))(?=\s)")


def escape_markdown(text: str) -> str:
    """Text from a document or a file name must be displayed, never interpreted (no links, no LaTeX, no lists)."""
    return _LIST_MARKER.sub(_escape_list_marker, _MD_SPECIAL.sub(r"\\\1", text))


def _escape_list_marker(match: re.Match[str]) -> str:
    indent, bullet, number, punct = match.groups()
    return f"{indent}\\{bullet}" if bullet else f"{indent}{number}\\{punct}"  # "- x" -> "\- x", "1. x" -> "1\. x"


def answer_markdown(answer: str) -> str:
    """The model's answer may use markdown (lists, `code`), but a stray '$' must not start LaTeX."""
    return answer.replace("$", "\\$")


def poll_phase(first_seen_s: float, now_s: float) -> str:
    """'fresh' (just started), 'slow' (show the free-tier hint), or 'stopped' (stop polling automatically)."""
    waited = now_s - first_seen_s
    if waited >= STOP_AFTER_S:
        return "stopped"
    return "slow" if waited >= SLOW_AFTER_S else "fresh"


def still_polling(docs: Iterable[Document], first_seen: Mapping[str, float], now_s: float) -> bool:
    """True if some document is 'processing' and has not yet used up its polling time."""
    return any(d.status == "processing" and poll_phase(first_seen.get(d.doc_id, now_s), now_s) != "stopped" for d in docs)


def page_text(page: int | None) -> str:
    return f"trang {page}" if page is not None else "không có số trang"


def source_header(source: Source) -> str:
    badge = ":green-badge[Được trích dẫn]" if source.cited else ":gray-badge[Chỉ truy xuất]"
    return (f"**[{source.number}]** {escape_markdown(source.filename)} · {page_text(source.page)} · "
            f"điểm {source.score:.3f} · {badge}")


def excerpt_quote(source: Source) -> str:
    return "> " + escape_markdown(source.excerpt) if source.excerpt else ""


def latency_caption(latency: Latency, used_fallback: bool) -> str:
    parts = [f"nhúng câu hỏi {latency.embed_ms:.0f} ms", f"tìm kiếm {latency.search_ms:.0f} ms"]
    parts.append("không gọi mô hình" if used_fallback else f"mô hình {latency.llm_ms / 1000:.1f} s")
    return "Thời gian: " + " · ".join(parts) + f" · tổng {latency.total_ms / 1000:.1f} s"


def fallback_score_text(best_score: float | None) -> str:
    if best_score is None:
        return "Không tìm thấy đoạn nào trong các tài liệu đã chọn."
    return (f"Điểm tương đồng cao nhất: {best_score:.3f}, thấp hơn ngưỡng của hệ thống, nên mô hình không được gọi "
            "(tránh trả lời bịa).")


def document_summary(doc: Document) -> str:
    bits = []
    if doc.num_pages:
        bits.append(f"{doc.num_pages} trang")
    if doc.num_chunks:
        bits.append(f"{doc.num_chunks} đoạn")
    return " · ".join(bits)


def selection_after_refresh(current: list[str], ready_ids: list[str], known_ready: set[str]) -> list[str]:
    """Chat scope after the document list changed: keep the user's choices that still exist and add
    documents that just became ready (default = all ready)."""
    kept = [d for d in current if d in ready_ids]
    return kept + [d for d in ready_ids if d not in known_ready and d not in kept]


# ---------------------------------------------------------------------- summary and quiz

OPTION_LETTERS = "ABCD"
# Mirrors the reason codes of app/quiz.py (the frontend never imports the backend).
DROP_REASONS_VI = {
    "schema_invalid": "sai định dạng",
    "duplicate_options": "đáp án trùng nhau",
    "positional_option": "đáp án tham chiếu đáp án khác",
    "unknown_source": "dẫn nguồn không có trong các đoạn đã gửi",
    "quote_not_in_source": "trích dẫn không có nguyên văn trong tài liệu",
    "duplicate_question": "câu hỏi trùng",
    "verifier_key_not_supported": "bộ xác minh: đáp án chưa được nguồn chứng minh",
    "verifier_ambiguous": "bộ xác minh: có đáp án khác cũng đúng",
}


def option_label(index: int, text: str) -> str:
    return f"{OPTION_LETTERS[index]}. {escape_markdown(text)}"


@dataclass(frozen=True, slots=True)
class Graded:
    question_id: int
    chosen: int | None  # None = not answered
    correct_index: int
    is_correct: bool


def grade_quiz(questions: Sequence[QuizQuestion], answers: Mapping[int, int | None]) -> tuple[list[Graded], int]:
    """Compare the chosen option of each question with its key. An unanswered question counts as wrong."""
    graded = [Graded(q.id, answers.get(q.id), q.answer_index, answers.get(q.id) == q.answer_index) for q in questions]
    return graded, sum(g.is_correct for g in graded)


def verdict_text(graded: Graded) -> str:
    if graded.chosen is None:
        return f"Chưa trả lời. Đáp án đúng là {OPTION_LETTERS[graded.correct_index]}."
    if graded.is_correct:
        return f"Đúng ({OPTION_LETTERS[graded.chosen]})."
    return f"Sai: bạn chọn {OPTION_LETTERS[graded.chosen]}, đáp án đúng là {OPTION_LETTERS[graded.correct_index]}."


def drop_summary(dropped: Mapping[str, int]) -> str:
    return "; ".join(f"{count} câu {DROP_REASONS_VI.get(reason, reason)}" for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]))


def _cost_text(calls: int, cached: bool, latency_ms: float) -> str:
    cost = "lấy từ kết quả đã lưu (0 lần gọi mô hình)" if cached else f"{calls} lần gọi mô hình"
    return f"{cost} · {latency_ms / 1000:.1f} s"


def summary_caption(summary: Summary) -> str:
    return f"Dùng {summary.num_chunks_used}/{summary.num_chunks_total} đoạn · " + _cost_text(summary.llm_calls, summary.cached, summary.latency_ms)


def quiz_caption(quiz: Quiz) -> str:
    if quiz.verified:
        checked = "đã xác minh"
    elif quiz.verify_requested:
        checked = "chưa xác minh được"
    else:
        checked = "không chạy bước xác minh"
    return f"{len(quiz.questions)}/{quiz.requested} câu · {checked} · " + _cost_text(quiz.llm_calls, quiz.cached, quiz.latency_ms)


def verified_badge(verified: bool | None) -> str:
    return ":green-badge[Đã xác minh]" if verified else ":gray-badge[Chưa xác minh]"
