"""UI tests (AppTest) for the "Tóm tắt" and "Trắc nghiệm" sections, with a fake API client: no browser, no network."""

import pytest

from frontend.api_client import ApiError, Quiz, QuizQuestion, QuizSource, Summary, VerifierDecision
from tests.test_frontend_app import A, B, FakeClient, doc, no_exception, start, texts

SECRET_EXPLANATION = "GIAI-THICH-BI-MAT"
SECRET_QUOTE = "TRICH-DAN-BI-MAT"


def question(qid: int, key: int, page: int | None, **changes) -> QuizQuestion:
    values = dict(id=qid, question=f"Nội dung câu hỏi số {qid}?", options=tuple(f"phương án {qid}{c}" for c in "abcd"), answer_index=key,
                  explanation=f"{SECRET_EXPLANATION}-{qid}", source=QuizSource(qid, page, "bao_cao.pdf", f"{SECRET_QUOTE}-{qid} nguyên văn"),
                  verified=True)
    values.update(changes)
    return QuizQuestion(**values)


def quiz_of(*questions: QuizQuestion, **changes) -> Quiz:
    values = dict(questions=questions or (question(1, 2, 5), question(2, 0, None), question(3, 3, 7)), requested=3, dropped={}, message=None,
                  warnings=(), verify_requested=True, verified=True, decisions=(), llm_calls=2, cached=False, latency_ms=6800.0)
    values.update(changes)
    return Quiz(**values)


def summary_of(**changes) -> Summary:
    values = dict(summary="Tài liệu nói về tối ưu PostgreSQL (trang 2).", style="short", num_chunks_used=29, num_chunks_total=29,
                  truncated=False, warning=None, llm_calls=1, cached=False, latency_ms=1620.0)
    values.update(changes)
    return Summary(**values)


def go(at, section: str):
    return at.radio(key="section").set_value(section).run()


def click(at, key: str):
    return at.button(key=key).click().run()


def page_text(at) -> str:
    """Everything visible: markdown, captions, alerts and the options of every radio."""
    radios = "\n".join(o for r in at.radio for o in r.options)
    return texts(at) + "\n" + radios


def submit_button(at):
    return next(b for b in at.button if b.label == "Kiểm tra")


def quiz_page(fake: FakeClient | None = None):
    fake = fake or FakeClient([doc()])
    fake.quiz_reply = fake.quiz_reply or quiz_of()
    at = go(start(fake), "Trắc nghiệm")
    return click(at, "quiz_make"), fake


# ---------------------------------------------------------------------- navigation


def test_the_chat_is_the_default_section_and_the_other_sections_replace_it() -> None:
    at = start(FakeClient([doc()]))
    assert at.radio(key="section").value == "Hỏi đáp" and len(at.chat_input) == 1
    at = go(at, "Tóm tắt")
    assert len(at.chat_input) == 0 and at.button(key="summary_make")
    at = go(at, "Trắc nghiệm")
    assert len(at.chat_input) == 0 and at.button(key="quiz_make")
    assert len(go(at, "Hỏi đáp").chat_input) == 1


def test_no_sections_when_the_backend_is_down() -> None:
    at = start(FakeClient([doc()], health=ApiError("unreachable", "Không kết nối được tới máy chủ API.")))
    no_exception(at)
    assert not at.radio and at.chat_input[0].disabled


@pytest.mark.parametrize("section", ["Tóm tắt", "Trắc nghiệm"])
def test_without_a_ready_document_the_tools_explain_what_to_do(section: str) -> None:
    at = go(start(FakeClient([doc(A, "a.pdf", "processing")])), section)
    no_exception(at)
    assert any("Hãy tải lên một tài liệu" in i.value for i in at.info)
    assert not [b for b in at.button if b.key in ("summary_make", "quiz_make")]  # nothing to press without a ready document


# ---------------------------------------------------------------------- summary


def test_the_summary_is_requested_on_a_click_and_shown_with_its_cost() -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = summary_of()
    at = go(start(fake), "Tóm tắt")
    assert fake.summary_calls == [] and "Tài liệu nói về" not in texts(at)  # nothing is asked just by opening the tab
    at = click(at, "summary_make")
    no_exception(at)
    assert fake.summary_calls == [(A, "short", False)]
    page = texts(at)
    assert "Tài liệu nói về tối ưu PostgreSQL (trang 2)." in page and "Dùng 29/29 đoạn · 1 lần gọi mô hình · 1.6 s" in page


def test_a_stored_summary_says_it_cost_nothing() -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = summary_of(cached=True, llm_calls=0, latency_ms=3.0)
    at = click(go(start(fake), "Tóm tắt"), "summary_make")
    assert "lấy từ kết quả đã lưu (0 lần gọi mô hình)" in texts(at)


def test_a_rerun_does_not_ask_again_and_each_style_is_kept_separately() -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = summary_of()
    at = click(go(start(fake), "Tóm tắt"), "summary_make")
    at.run()
    at.run()
    assert len(fake.summary_calls) == 1
    fake.summary_reply = summary_of(summary="Bản chi tiết.", style="detailed")
    at = at.radio(key="summary_style").set_value("detailed").run()
    assert "Bản chi tiết." not in texts(at) and "Tài liệu nói về" not in texts(at)  # not asked yet
    at = click(at, "summary_make")
    assert fake.summary_calls[-1] == (A, "detailed", False) and "Bản chi tiết." in texts(at)
    at = at.radio(key="summary_style").set_value("short").run()
    assert "Tài liệu nói về" in texts(at) and len(fake.summary_calls) == 2  # the short one came from the session, no call


def test_regenerate_asks_with_refresh() -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = summary_of()
    click(click(go(start(fake), "Tóm tắt"), "summary_make"), "summary_again")
    assert fake.summary_calls == [(A, "short", False), (A, "short", True)]


def test_a_truncated_summary_shows_the_servers_warning() -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = summary_of(truncated=True, num_chunks_used=14, warning="Tài liệu quá dài (29 đoạn). Chỉ 14 đoạn đầu được tóm tắt.")
    at = click(go(start(fake), "Tóm tắt"), "summary_make")
    assert any("Chỉ 14 đoạn đầu" in w.value for w in at.warning) and "Dùng 14/29 đoạn" in texts(at)


@pytest.mark.parametrize(
    ("error", "level"),
    [(ApiError("http", "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau.", status=429), "error"),
     (ApiError("http", "Không đọc được nội dung của tài liệu trong chỉ mục. Hãy xóa tài liệu và tải lên lại.", status=409), "error"),
     (ApiError("timeout", "Máy chủ chưa trả lời sau 120 giây. Yêu cầu có thể vẫn đang được xử lý ở phía máy chủ."), "warning")],
)
def test_summary_errors_are_shown_without_a_stack_trace(error: ApiError, level: str) -> None:
    fake = FakeClient([doc()])
    fake.summary_reply = error
    at = click(go(start(fake), "Tóm tắt"), "summary_make")
    no_exception(at)
    assert error.message in [e.value for e in getattr(at, level)] and "Traceback" not in texts(at)


def test_the_summary_follows_the_chosen_document() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    fake.summary_reply = summary_of()
    at = go(start(fake), "Tóm tắt")
    at = at.selectbox(key="study_doc").set_value(B).run()
    click(at, "summary_make")
    assert fake.summary_calls == [(B, "short", False)]


# ---------------------------------------------------------------------- quiz: creating it


def test_the_quiz_controls_are_passed_to_the_backend() -> None:
    fake = FakeClient([doc()])
    fake.quiz_reply = quiz_of()
    at = go(start(fake), "Trắc nghiệm")
    at.number_input(key="quiz_count").set_value(7)
    at.number_input(key="quiz_page_from").set_value(2)
    at.number_input(key="quiz_page_to").set_value(4)
    at.text_input(key="quiz_topic").set_value("  replication  ")
    at.checkbox(key="quiz_verify").uncheck()
    click(at.run(), "quiz_make")
    assert fake.quiz_calls == [{"doc_id": A, "num_questions": 7, "topic": "replication", "page_from": 2, "page_to": 4,
                                "verify": False, "refresh": False}]


def test_the_defaults_mean_five_questions_no_limits_and_verification() -> None:
    quiz_page_, fake = quiz_page()
    assert fake.quiz_calls == [{"doc_id": A, "num_questions": 5, "topic": None, "page_from": None, "page_to": None,
                                "verify": True, "refresh": False}]


def test_a_new_set_asks_with_refresh() -> None:
    at, fake = quiz_page()
    click(at, "quiz_new")
    assert [c["refresh"] for c in fake.quiz_calls] == [False, True]


def test_the_quiz_is_shown_with_its_cost_and_all_options_but_nothing_that_gives_the_answer_away() -> None:
    at, fake = quiz_page()
    no_exception(at)
    page = page_text(at)
    assert "Câu 1." in page and "Nội dung câu hỏi số 3?" in page
    assert "A. phương án 1a" in page and "D. phương án 3d" in page  # every option is offered
    assert "3/3 câu · đã xác minh · 2 lần gọi mô hình · 6.8 s" in page
    for secret in (SECRET_EXPLANATION, SECRET_QUOTE, "Đáp án đúng", "Giải thích", "Điểm:", "Nguồn:"):
        assert secret not in page, secret  # the key, the explanation and the source stay hidden until "Kiểm tra"
    assert all(r.value is None for r in at.radio if r.key.startswith("quiz_"))  # nothing is pre-selected
    import re

    labels = [o for r in at.radio if r.key.startswith("quiz_") for o in r.options]
    assert len(labels) == 12 and all(re.fullmatch(r"[A-D]\. phương án \d[a-d]", o) for o in labels)  # no option is marked in any way


def test_the_key_is_not_in_any_widget_the_browser_receives() -> None:
    at, _ = quiz_page()
    visible = [str(r.proto) for r in at.radio] + [str(m.proto) for m in at.markdown] + [str(c.proto) for c in at.caption]
    assert not any(SECRET_QUOTE in text or SECRET_EXPLANATION in text for text in visible)


def test_a_smaller_quiz_comes_with_the_explanation_and_the_dropped_reasons() -> None:
    reply = quiz_of(question(1, 2, 5), requested=5, message="Chỉ tạo được 1/5 câu hỏi hợp lệ. Hệ thống không bịa thêm câu hỏi cho đủ số lượng.",
                    dropped={"quote_not_in_source": 3, "verifier_ambiguous": 1},
                    decisions=(VerifierDecision("Câu mơ hồ nọ?", True, True, "Hai đáp án đều đúng", False, False),))
    fake = FakeClient([doc()])
    fake.quiz_reply = reply
    at, _ = quiz_page(fake)
    assert any("Chỉ tạo được 1/5" in w.value for w in at.warning)
    text = texts(at)
    assert "3 câu trích dẫn không có nguyên văn trong tài liệu" in text and "1 câu bộ xác minh: có đáp án khác cũng đúng" in text
    assert "Câu mơ hồ nọ? — Hai đáp án đều đúng" in text


def test_an_unverified_quiz_is_labelled_so() -> None:
    fake = FakeClient([doc()])
    fake.quiz_reply = quiz_of(verified=False, warnings=("Không chạy được bước xác minh (hết hạn mức); các câu hỏi chưa được xác minh.",))
    at, _ = quiz_page(fake)
    assert any("chưa được xác minh" in w.value for w in at.warning) and "chưa xác minh được" in texts(at)


def test_an_empty_quiz_shows_the_message_and_no_form() -> None:
    fake = FakeClient([doc()])
    fake.quiz_reply = quiz_of(questions=(), message="Không tạo được câu hỏi hợp lệ nào từ phần tài liệu này.")
    at, _ = quiz_page(fake)
    no_exception(at)
    assert any("Không tạo được câu hỏi hợp lệ nào" in w.value for w in at.warning) and not [r for r in at.radio if r.key.startswith("quiz_")]


@pytest.mark.parametrize(
    ("error", "level"),
    [(ApiError("http", "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau.", status=429), "error"),
     (ApiError("http", "Không có nội dung nào khớp với khoảng trang hoặc chủ đề đã chọn.", status=422), "error"),
     (ApiError("http", "Mô hình AI trả về kết quả không đúng định dạng. Hãy thử lại.", status=502), "error"),
     (ApiError("timeout", "Máy chủ chưa trả lời sau 120 giây. Nếu máy chủ đã làm xong, bấm lại sẽ lấy kết quả đã lưu ngay."), "warning")],
)
def test_quiz_errors_are_shown_without_a_stack_trace(error: ApiError, level: str) -> None:
    fake = FakeClient([doc()])
    fake.quiz_reply = error
    at, _ = quiz_page(fake)
    no_exception(at)
    assert error.message in [e.value for e in getattr(at, level)] and "Traceback" not in texts(at)
    assert not [r for r in at.radio if r.key.startswith("quiz_")]


# ---------------------------------------------------------------------- quiz: answering


def answer(at, *choices: int | None):
    for qid, choice in enumerate(choices, start=1):
        if choice is not None:
            at.radio(key=f"quiz_1_{qid}").set_value(choice)
    return submit_button(at).click().run()


def test_checking_the_answers_scores_them_and_reveals_explanations_and_sources() -> None:
    at, _ = quiz_page()  # keys: Q1 -> C (2), Q2 -> A (0), Q3 -> D (3)
    at = answer(at, 2, 1, None)  # right, wrong, unanswered
    no_exception(at)
    page = texts(at)
    assert "Điểm: 1/3" in page
    assert any("Đúng (C)." in s.value for s in at.success)
    errors = [e.value for e in at.error]
    assert "Sai: bạn chọn B, đáp án đúng là A." in errors and "Chưa trả lời. Đáp án đúng là D." in errors
    for qid in (1, 2, 3):
        assert f"{SECRET_EXPLANATION}-{qid}" in page and f"{SECRET_QUOTE}-{qid} nguyên văn" in page
    assert "**Đáp án đúng:** C. phương án 1c" in page
    assert "Nguồn: bao\\_cao.pdf, trang 5" in page and "Nguồn: bao\\_cao.pdf, không có số trang" in page and "trang 7" in page
    assert page.count("Đã xác minh") == 3


def test_all_right_and_all_wrong() -> None:
    at, _ = quiz_page()
    assert "Điểm: 3/3" in texts(answer(at, 2, 0, 3))
    at, _ = quiz_page()
    assert "Điểm: 0/3" in texts(answer(at, 0, 1, 0))


def test_after_checking_the_answers_cannot_be_changed() -> None:
    at, _ = quiz_page()
    at = answer(at, 2, 0, 3)
    assert all(r.disabled for r in at.radio if r.key.startswith("quiz_")) and submit_button(at).disabled


def test_a_rerun_after_checking_keeps_the_result_and_does_not_call_the_backend() -> None:
    at, fake = quiz_page()
    at = answer(at, 2, 0, 3)
    at.run()
    at.run()
    assert "Điểm: 3/3" in texts(at) and len(fake.quiz_calls) == 1


def test_retrying_hides_the_key_again_and_clears_the_choices() -> None:
    at, fake = quiz_page()
    at = answer(at, 2, 1, 3)
    at = click(at, "quiz_retry")
    page = page_text(at)
    assert "Điểm:" not in page and SECRET_EXPLANATION not in page and "Đáp án đúng" not in page
    radios = [r for r in at.radio if r.key.startswith("quiz_")]
    assert len(radios) == 3 and all(r.value is None and not r.disabled for r in radios) and not submit_button(at).disabled
    assert all(r.key.startswith("quiz_2_") for r in radios)  # new widget keys, so no browser keeps the old selections
    assert len(fake.quiz_calls) == 1  # same questions, no new call


def test_a_new_set_replaces_the_quiz_and_its_answers() -> None:
    at, fake = quiz_page()
    at = answer(at, 2, 0, 3)
    fake.quiz_reply = quiz_of(question(1, 1, 3, question="Một câu hỏi hoàn toàn mới?"), requested=1)
    at = click(at, "quiz_new")
    page = page_text(at)
    assert "Một câu hỏi hoàn toàn mới?" in page and "Điểm:" not in page and "Nội dung câu hỏi số 3?" not in page


def test_the_quiz_belongs_to_the_document_it_was_made_for() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    fake.quiz_reply = quiz_of()
    at, _ = quiz_page(fake)
    assert "Câu 1." in page_text(at)
    at = at.selectbox(key="study_doc").set_value(B).run()
    assert "Câu 1." not in page_text(at)  # another document: the old quiz is hidden, not shown as if it were its own
    at = at.selectbox(key="study_doc").set_value(A).run()
    assert "Câu 1." in page_text(at)


def test_document_text_in_a_quiz_is_shown_literally() -> None:
    fake = FakeClient([doc()])
    fake.quiz_reply = quiz_of(question(1, 0, 2, question="Lệnh `EXPLAIN` có **tác dụng** gì $x$?", options=("[a](http://evil.test)", "b", "c", "d"),
                                       explanation="Xem *tài liệu*", source=QuizSource(1, 2, "[x].pdf", "> trích **đậm**")))
    at, _ = quiz_page(fake)
    page = page_text(at)
    assert "Lệnh \\`EXPLAIN\\` có \\*\\*tác dụng\\*\\* gì \\$x\\$?" in page and "A. \\[a\\](http\\://evil.test)" in page
    at = answer(at, 0)
    page = texts(at)
    assert "Xem \\*tài liệu\\*" in page and "\\[x\\].pdf" in page and "\\> trích \\*\\*đậm\\*\\*" in page
