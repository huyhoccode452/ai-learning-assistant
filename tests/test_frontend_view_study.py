"""frontend/view.py helpers for the quiz and summary sections (pure functions)."""

from frontend.api_client import Quiz, QuizQuestion, QuizSource, Summary
from frontend.view import (
    Graded,
    drop_summary,
    grade_quiz,
    option_label,
    quiz_caption,
    summary_caption,
    verdict_text,
    verified_badge,
)


def question(qid: int, key: int) -> QuizQuestion:
    return QuizQuestion(qid, f"Câu {qid}?", ("a", "b", "c", "d"), key, "vì sao", QuizSource(qid, 1, "f.pdf", "trích dẫn"), True)


def test_grading_counts_correct_wrong_and_unanswered() -> None:
    questions = [question(1, 2), question(2, 0), question(3, 3), question(4, 1)]
    graded, score = grade_quiz(questions, {1: 2, 2: 1, 3: None})  # right, wrong, unanswered, missing entirely
    assert score == 1
    assert graded == [Graded(1, 2, 2, True), Graded(2, 1, 0, False), Graded(3, None, 3, False), Graded(4, None, 1, False)]


def test_grading_all_right_all_wrong_and_nothing() -> None:
    questions = [question(1, 0), question(2, 1)]
    assert grade_quiz(questions, {1: 0, 2: 1})[1] == 2
    assert grade_quiz(questions, {1: 3, 2: 3})[1] == 0
    assert grade_quiz([], {}) == ([], 0)
    assert grade_quiz(questions, {})[1] == 0


def test_an_answer_of_zero_is_a_real_answer_not_unanswered() -> None:
    graded, score = grade_quiz([question(1, 0)], {1: 0})
    assert score == 1 and graded[0].chosen == 0  # option A has index 0, which is falsy


def test_verdict_texts() -> None:
    assert verdict_text(Graded(1, 2, 2, True)) == "Đúng (C)."
    assert verdict_text(Graded(1, 1, 3, False)) == "Sai: bạn chọn B, đáp án đúng là D."
    assert verdict_text(Graded(1, None, 0, False)) == "Chưa trả lời. Đáp án đúng là A."


def test_option_label_is_lettered_and_escaped() -> None:
    assert option_label(0, "4MB") == "A. 4MB" and option_label(3, "[x](http://e.t)") == "D. \\[x\\](http\\://e.t)"


def test_drop_summary_lists_the_most_common_reason_first_and_keeps_unknown_codes() -> None:
    text = drop_summary({"verifier_ambiguous": 1, "quote_not_in_source": 3, "something_new": 2})
    assert text == "3 câu trích dẫn không có nguyên văn trong tài liệu; 2 câu something_new; 1 câu bộ xác minh: có đáp án khác cũng đúng"
    assert drop_summary({}) == ""


def test_summary_caption_for_a_new_and_a_stored_result() -> None:
    base = dict(summary="x", style="short", num_chunks_used=14, num_chunks_total=29, truncated=True, warning="w")
    assert summary_caption(Summary(**base, llm_calls=3, cached=False, latency_ms=6300.0)) == "Dùng 14/29 đoạn · 3 lần gọi mô hình · 6.3 s"
    assert summary_caption(Summary(**base, llm_calls=0, cached=True, latency_ms=4.0)) == "Dùng 14/29 đoạn · lấy từ kết quả đã lưu (0 lần gọi mô hình) · 0.0 s"


def test_quiz_caption_says_whether_it_was_verified() -> None:
    def quiz(**changes) -> Quiz:
        values = dict(questions=(question(1, 0), question(2, 1)), requested=5, dropped={}, message=None, warnings=(), verify_requested=True,
                      verified=True, decisions=(), llm_calls=2, cached=False, latency_ms=7000.0)
        values.update(changes)
        return Quiz(**values)

    assert quiz_caption(quiz()) == "2/5 câu · đã xác minh · 2 lần gọi mô hình · 7.0 s"
    assert quiz_caption(quiz(verified=False)) == "2/5 câu · chưa xác minh được · 2 lần gọi mô hình · 7.0 s"
    assert quiz_caption(quiz(verified=False, verify_requested=False, llm_calls=1)) == "2/5 câu · không chạy bước xác minh · 1 lần gọi mô hình · 7.0 s"
    assert quiz_caption(quiz(cached=True, llm_calls=0)).endswith("lấy từ kết quả đã lưu (0 lần gọi mô hình) · 7.0 s")


def test_verified_badge() -> None:
    assert verified_badge(True) == ":green-badge[Đã xác minh]" and verified_badge(None) == ":gray-badge[Chưa xác minh]"
    assert verified_badge(False) == ":gray-badge[Chưa xác minh]"
