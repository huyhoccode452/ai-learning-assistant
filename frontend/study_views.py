"""The "Tóm tắt" (summary) and "Trắc nghiệm" (quiz) sections of the Streamlit app.

Both call the backend only when the user presses a button; the result is kept in `st.session_state`, so a rerun never
asks again. The backend stores results too (SQLite), so even a new session or a retry after a timeout is free.
The quiz answer key is kept in session_state but is never written to the page until the user presses "Kiểm tra".
"""

import streamlit as st

from frontend import view
from frontend.api_client import ApiClient, ApiError, Document, Quiz

SPINNER = "Đang tạo… có thể mất vài chục giây (tối đa khoảng 2 phút). Kết quả được lưu để lần sau không tốn thêm lượt gọi."
STYLE_LABELS = {"short": "Ngắn gọn", "detailed": "Chi tiết"}


def _pick_document(ready: list[Document]) -> Document:
    ids = [d.doc_id for d in ready]
    if st.session_state.get("study_doc") not in ids:  # the chosen document was deleted or is gone
        st.session_state.study_doc = ids[0]
    names = {d.doc_id: d.filename for d in ready}
    chosen = st.selectbox("Tài liệu", ids, format_func=lambda i: names[i], key="study_doc")
    return next(d for d in ready if d.doc_id == chosen)


def _show_error(key: str) -> None:
    error = st.session_state.get(key)
    if error:
        getattr(st, "warning" if error[0] == "timeout" else "error")(error[1])


# ---------------------------------------------------------------------- summary


def summary_area(client: ApiClient, ready: list[Document]) -> None:
    ss = st.session_state
    if not ready:
        st.info("Hãy tải lên một tài liệu và đợi trạng thái “Sẵn sàng” để tóm tắt.")
        return
    doc = _pick_document(ready)
    style = st.radio("Kiểu tóm tắt", list(STYLE_LABELS), format_func=STYLE_LABELS.get, horizontal=True, key="summary_style")
    fresh, regenerate = st.columns(2)
    make = fresh.button("Tạo tóm tắt", key="summary_make", type="primary")
    again = regenerate.button("Tạo lại (bỏ qua kết quả đã lưu)", key="summary_again", help="Tốn thêm lượt gọi mô hình")
    if make or again:
        ss.summary_error = None
        with st.spinner(SPINNER):
            try:
                ss.summaries[(doc.doc_id, style)] = client.summary(doc.doc_id, style, refresh=again)
            except ApiError as exc:
                ss.summary_error = (exc.kind, exc.message)
    _show_error("summary_error")
    result = ss.summaries.get((doc.doc_id, style))
    if result:
        st.markdown(view.answer_markdown(result.summary))
        if result.truncated and result.warning:
            st.warning(result.warning)
        st.caption(view.summary_caption(result))


# ---------------------------------------------------------------------- quiz


def _generate_quiz(client: ApiClient, doc: Document, refresh: bool) -> None:
    ss = st.session_state
    pages = [ss.quiz_page_from or None, ss.quiz_page_to or None]  # 0 = no limit
    ss.quiz_error = None
    with st.spinner(SPINNER):
        try:
            quiz = client.quiz(doc.doc_id, num_questions=int(ss.quiz_count), topic=(ss.quiz_topic.strip() or None),
                               page_from=pages[0], page_to=pages[1], verify=bool(ss.quiz_verify), refresh=refresh)
        except ApiError as exc:
            ss.quiz_error = (exc.kind, exc.message)
            return
    ss.quiz_run += 1  # new widget keys: the radios of a new quiz start empty
    ss.quiz = {"doc_id": doc.doc_id, "data": quiz, "graded": None}


def _render_questions(quiz: Quiz, run: int, locked: bool) -> dict[int, int | None]:
    answers: dict[int, int | None] = {}
    for q in quiz.questions:
        st.markdown(f"**Câu {q.id}.** {view.escape_markdown(q.question)}")
        answers[q.id] = st.radio(
            f"Câu {q.id}", [0, 1, 2, 3], index=None, key=f"quiz_{run}_{q.id}", label_visibility="collapsed", disabled=locked,
            format_func=lambda i, q=q: view.option_label(i, q.options[i]))
    return answers


def _render_results(quiz: Quiz, graded: list[view.Graded], score: int) -> None:
    st.subheader(f"Điểm: {score}/{len(graded)}")
    for q, g in zip(quiz.questions, graded):
        with st.container(border=True):
            st.markdown(f"**Câu {q.id}.** {view.escape_markdown(q.question)}  \n{view.verified_badge(q.verified)}")
            (st.success if g.is_correct else st.error)(view.verdict_text(g))
            st.markdown(f"**Đáp án đúng:** {view.option_label(q.answer_index, q.options[q.answer_index])}")
            st.markdown(f"**Giải thích:** {view.escape_markdown(q.explanation)}")
            st.caption(f"Nguồn: {view.escape_markdown(q.source.filename)}, {view.page_text(q.source.page)}")
            st.markdown("> " + view.escape_markdown(q.source.quote))


def quiz_area(client: ApiClient, ready: list[Document]) -> None:
    ss = st.session_state
    if not ready:
        st.info("Hãy tải lên một tài liệu và đợi trạng thái “Sẵn sàng” để tạo câu hỏi.")
        return
    doc = _pick_document(ready)
    left, middle, right = st.columns(3)
    left.number_input("Số câu hỏi", 1, 10, 5, key="quiz_count")
    middle.number_input("Từ trang (0 = không giới hạn)", 0, 10_000, 0, key="quiz_page_from")
    right.number_input("Đến trang (0 = không giới hạn)", 0, 10_000, 0, key="quiz_page_to")
    st.text_input("Chủ đề (tùy chọn)", key="quiz_topic", max_chars=200, placeholder="Ví dụ: replication và sao lưu")
    st.checkbox("Kiểm tra lại từng câu bằng một lần gọi mô hình nữa (chậm hơn, ít câu sai hơn)", value=True, key="quiz_verify")
    make, renew = st.columns(2)
    start = make.button("Tạo bộ câu hỏi", key="quiz_make", type="primary")
    new_set = renew.button("Tạo bộ mới (bỏ qua kết quả đã lưu)", key="quiz_new", help="Tốn thêm lượt gọi mô hình")
    if start or new_set:
        _generate_quiz(client, doc, refresh=new_set)
    _show_error("quiz_error")

    state = ss.quiz
    if not state or state["doc_id"] != doc.doc_id:
        return
    quiz: Quiz = state["data"]
    if quiz.message:
        st.warning(quiz.message)
    for warning in quiz.warnings:
        st.warning(warning)
    st.caption(view.quiz_caption(quiz))
    if quiz.dropped:
        with st.expander("Các câu đã bị loại"):
            st.markdown(view.drop_summary(quiz.dropped))
            for decision in (d for d in quiz.decisions if not d.kept):
                st.caption(f"{view.escape_markdown(decision.question)} — {view.escape_markdown(decision.reason)}")
    if not quiz.questions:
        return

    graded = state["graded"]
    with st.form(f"quiz_form_{ss.quiz_run}"):
        answers = _render_questions(quiz, ss.quiz_run, locked=graded is not None)
        checked = st.form_submit_button("Kiểm tra", disabled=graded is not None, type="primary")
    if checked:
        state["graded"] = view.grade_quiz(quiz.questions, answers)
        st.rerun()
    if graded is not None:
        _render_results(quiz, *graded)
        if st.button("Làm lại bộ câu hỏi này", key="quiz_retry"):
            ss.quiz_run += 1
            state["graded"] = None
            st.rerun()
