"""Tests for frontend/view.py: pure presentation helpers."""

import pytest

from frontend.api_client import Document, Latency, Source
from frontend.view import (
    SLOW_AFTER_S,
    STOP_AFTER_S,
    answer_markdown,
    document_summary,
    escape_markdown,
    excerpt_quote,
    fallback_score_text,
    latency_caption,
    page_text,
    poll_phase,
    selection_after_refresh,
    source_header,
    still_polling,
)


def doc(doc_id: str, status: str, pages: int | None = None, chunks: int | None = None) -> Document:
    return Document(doc_id, f"{doc_id}.pdf", status, None, None, pages, chunks)


def test_escape_markdown_makes_text_literal() -> None:
    assert escape_markdown("**đậm** _nghiêng_ `code`") == "\\*\\*đậm\\*\\* \\_nghiêng\\_ \\`code\\`"
    assert escape_markdown("[a](http://x.test)") == "\\[a\\](http\\://x.test)"
    assert escape_markdown("giá $5 và $6") == "giá \\$5 và \\$6"
    assert escape_markdown("# tiêu đề > trích <b> a|b ~x~ hello!") == "\\# tiêu đề \\> trích \\<b\\> a\\|b \\~x\\~ hello\\!"
    assert escape_markdown("back\\slash") == "back\\\\slash"
    assert escape_markdown("") == "" and escape_markdown("plain text 123") == "plain text 123"


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [("- mục", "\\- mục"), ("+ mục", "\\+ mục"), ("1. Giới thiệu", "1\\. Giới thiệu"), ("12) Phần", "12\\) Phần"),
     ("  - lùi", "  \\- lùi"), ("a - b", "a - b"), ("2026-01-01", "2026-01-01"), ("1.5 MB", "1.5 MB")],
)
def test_a_leading_list_marker_is_not_turned_into_a_list(raw: str, escaped: str) -> None:
    assert escape_markdown(raw) == escaped


def test_answer_markdown_only_protects_the_dollar_sign() -> None:
    assert answer_markdown("**đậm** và $5 [1]") == "**đậm** và \\$5 [1]"


def test_poll_phase_boundaries() -> None:
    assert (SLOW_AFTER_S, STOP_AFTER_S) == (30.0, 180.0)
    assert poll_phase(100.0, 100.0) == "fresh"
    assert poll_phase(100.0, 129.9) == "fresh" and poll_phase(100.0, 130.0) == "slow"
    assert poll_phase(100.0, 279.9) == "slow" and poll_phase(100.0, 280.0) == "stopped"
    assert poll_phase(100.0, 10_000.0) == "stopped"


def test_still_polling_needs_a_processing_document_inside_its_time_limit() -> None:
    docs = [doc("a", "ready"), doc("b", "failed"), doc("c", "processing")]
    assert still_polling(docs, {"c": 100.0}, 150.0)
    assert not still_polling(docs, {"c": 100.0}, 300.0)  # gave up
    assert still_polling(docs, {}, 500.0)  # never seen before: starts counting now
    assert not still_polling([doc("a", "ready"), doc("b", "failed")], {}, 0.0)
    assert still_polling([doc("c", "processing"), doc("d", "processing")], {"c": 0.0, "d": 250.0}, 300.0)  # one is still fresh


def test_page_text_for_pdf_and_for_formats_without_pages() -> None:
    assert page_text(5) == "trang 5" and page_text(None) == "không có số trang"
    assert page_text(0) == "trang 0"  # a falsy page number is still a page


def test_source_header_marks_cited_against_retrieved_only() -> None:
    cited = source_header(Source(1, "bao_cao.pdf", 5, 0.83123, "x", True))
    retrieved = source_header(Source(2, "ghi_chu.txt", None, 0.7, "x", False))
    assert cited == "**[1]** bao\\_cao.pdf · trang 5 · điểm 0.831 · :green-badge[Được trích dẫn]"
    assert retrieved == "**[2]** ghi\\_chu.txt · không có số trang · điểm 0.700 · :gray-badge[Chỉ truy xuất]"


def test_excerpt_quote() -> None:
    assert excerpt_quote(Source(1, "a", 1, 0.5, "work_mem 4MB", True)) == "> work\\_mem 4MB"
    assert excerpt_quote(Source(1, "a", 1, 0.5, "", True)) == ""


def test_latency_caption_for_an_answer_and_for_a_fallback() -> None:
    latency = Latency(embed_ms=123.4, search_ms=15.6, llm_ms=1830.0, total_ms=1990.0)
    assert latency_caption(latency, False) == "Thời gian: nhúng câu hỏi 123 ms · tìm kiếm 16 ms · mô hình 1.8 s · tổng 2.0 s"
    assert latency_caption(Latency(100, 10, 0, 300), True) == "Thời gian: nhúng câu hỏi 100 ms · tìm kiếm 10 ms · không gọi mô hình · tổng 0.3 s"


def test_fallback_score_text() -> None:
    assert "0.543" in fallback_score_text(0.5427) and "thấp hơn ngưỡng" in fallback_score_text(0.5427)
    assert fallback_score_text(None) == "Không tìm thấy đoạn nào trong các tài liệu đã chọn."


def test_document_summary() -> None:
    assert document_summary(doc("a", "ready", pages=9, chunks=29)) == "9 trang · 29 đoạn"
    assert document_summary(doc("a", "ready", chunks=4)) == "4 đoạn"
    assert document_summary(doc("a", "processing")) == ""


def test_selection_keeps_choices_adds_new_ready_documents_and_drops_gone_ones() -> None:
    # the user deselected "b"; "c" just became ready; "d" was deleted
    assert selection_after_refresh(["a", "d"], ["a", "b", "c"], {"a", "b", "d"}) == ["a", "c"]
    assert selection_after_refresh([], ["a", "b"], set()) == ["a", "b"]  # first load: everything
    assert selection_after_refresh([], ["a"], {"a"}) == []  # deselected everything: stays empty
    assert selection_after_refresh(["a"], ["a"], {"a"}) == ["a"]
