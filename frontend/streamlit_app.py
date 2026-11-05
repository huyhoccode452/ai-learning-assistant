"""Streamlit UI for the AI Learning Assistant: a thin, stateless client of the FastAPI backend.

    API_BASE_URL=http://127.0.0.1:8000 streamlit run frontend/streamlit_app.py

Streamlit re-runs this whole script on every interaction, so anything that must survive lives in
`st.session_state` (chat history, the pending question, upload guards). The server keeps the truth
(documents and their status); the UI only shows it. The browser never sees any API key: the key stays in
the backend, and this file only knows the backend's URL.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root: `frontend.*` imports

from frontend import view  # noqa: E402
from frontend.history_views import apply_pending_load, history_section, reset_conversation  # noqa: E402
from frontend.api_client import DEFAULT_BASE_URL, ApiClient, ApiError, ChatResult, Document  # noqa: E402
from frontend.study_views import quiz_area, summary_area  # noqa: E402

MAX_QUESTION_CHARS = 2000  # same limit as the backend (app/rag.py)
SECTIONS = ("Hỏi đáp", "Tóm tắt", "Trắc nghiệm")
INTERRUPTED = ("Yêu cầu trước đó bị gián đoạn khi trang được làm mới nên chưa có câu trả lời. "
               "Máy chủ có thể vẫn đang xử lý; nếu nó đã xong, cuộc trò chuyện đã lưu trong mục Lịch sử sẽ có câu trả lời. "
               "Hãy hỏi lại nếu cần.")


def upload_limit_mb() -> int:
    try:
        return max(1, int(os.environ.get("UPLOAD_MAX_MB", "20")))  # keep equal to the backend's MAX_UPLOAD_MB
    except ValueError:
        return 20


def init_state() -> None:
    defaults = {"messages": [], "pending": None, "uploader_n": 0, "handled": set(), "notice": None, "confirm_delete": None,
                "first_seen": {}, "docs": [], "scope": [], "known_ready": set(), "docs_sig": None, "polling_active": False,
                "fresh_docs": False, "docs_error": None, "summaries": {}, "summary_error": None, "quiz": None, "quiz_run": 0,
                "quiz_error": None, "conversation_id": None, "pending_load": None, "history_limit": 10,
                "confirm_delete_conversation": None, "history_missing": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "client" not in st.session_state:  # tests put a fake client here before the first run
        try:
            st.session_state.client = ApiClient(os.environ.get("API_BASE_URL", DEFAULT_BASE_URL))
        except ValueError:
            st.error("Biến môi trường API_BASE_URL không hợp lệ. Nó phải có dạng http://127.0.0.1:8000.")
            st.stop()


# ---------------------------------------------------------------------- sidebar: upload


def handle_upload(client: ApiClient, uploaded) -> None:
    """Send a newly chosen file once. The uploader keeps its file across reruns, so every file is guarded by a key
    and the widget is then reset (new key), which prevents a re-upload loop."""
    ss = st.session_state
    key = (getattr(uploaded, "file_id", None), uploaded.name, uploaded.size)
    if key in ss.handled:
        return
    ss.handled.add(key)
    with st.spinner("Đang gửi tệp lên máy chủ…"):
        try:
            result = client.upload(uploaded.name, uploaded.getvalue())
        except ApiError as exc:
            ss.notice = ("error", f"Không tải lên được «{uploaded.name}»: {exc.message}")
        else:
            if result.already_indexed:
                ss.notice = ("info", f"«{uploaded.name}» đã có sẵn và đã sẵn sàng, không cần tải lại.")
            else:
                ss.first_seen.setdefault(result.document.doc_id, time.monotonic())
                ss.notice = ("success", f"Đã nhận «{uploaded.name}». Hệ thống đang xử lý ở nền; "
                                        "bạn có thể hỏi các tài liệu khác trong lúc chờ.")
    ss.uploader_n += 1
    st.rerun()


def show_notice() -> None:
    notice = st.session_state.notice
    if notice:
        getattr(st, notice[0])(notice[1])
        st.session_state.notice = None  # shown once


# ---------------------------------------------------------------------- sidebar: documents (polled fragment)


def delete_document(client: ApiClient, doc: Document) -> None:
    ss = st.session_state
    try:
        removed = client.delete_document(doc.doc_id)
    except ApiError as exc:
        ss.notice = ("error", f"Không xóa được «{doc.filename}»: {exc.message}")
    else:
        ss.notice = ("success", f"Đã xóa «{doc.filename}» ({removed} đoạn).")
    ss.confirm_delete = None
    st.rerun()


def document_row(client: ApiClient, doc: Document) -> None:
    ss = st.session_state
    with st.container(border=True):
        st.markdown(f"**{view.escape_markdown(doc.filename)}**  \n{view.STATUS_BADGES.get(doc.status, doc.status)} "
                    f"{view.document_summary(doc)}")
        if doc.status == "failed":
            st.caption((doc.error_message or "Xử lý thất bại.") + " Hãy tải lại tệp để thử lại.")
        if ss.confirm_delete == doc.doc_id:
            st.warning("Xóa tài liệu này cùng dữ liệu đã lập chỉ mục?")
            yes, no = st.columns(2)
            if yes.button("Xác nhận xóa", key=f"yes_{doc.doc_id}", type="primary"):
                delete_document(client, doc)
            if no.button("Hủy", key=f"no_{doc.doc_id}"):
                ss.confirm_delete = None
                st.rerun()
        elif st.button("Xóa", key=f"del_{doc.doc_id}", disabled=doc.status == "processing",
                       help="Không xóa được khi tài liệu đang xử lý" if doc.status == "processing" else None):
            ss.confirm_delete = doc.doc_id
            st.rerun()


def refresh_documents(client: ApiClient) -> ApiError | None:
    """Fetch the document list into session_state. Returns the error if it failed (the old list is kept)."""
    ss = st.session_state
    try:
        ss.docs = client.list_documents()
    except ApiError as exc:
        return exc
    now = time.monotonic()
    processing = {d.doc_id for d in ss.docs if d.status == "processing"}
    for doc_id in processing:
        ss.first_seen.setdefault(doc_id, now)
    for doc_id in [i for i in ss.first_seen if i not in processing]:
        del ss.first_seen[doc_id]
    return None


def documents_panel(client: ApiClient) -> None:
    """Called inline by every full run (which has just fetched the list) and, while something is processing, run
    by itself every few seconds (st.fragment). A change seen by itself triggers one full rerun."""
    ss = st.session_state
    standalone = not ss.fresh_docs  # a polling tick or a click inside the panel: the list must be fetched here
    ss.fresh_docs = False
    before = ss.docs_sig
    error = refresh_documents(client) if standalone else ss.docs_error
    docs, now = ss.docs, time.monotonic()
    ss.docs_sig = tuple((d.doc_id, d.status) for d in docs)
    if error:
        st.caption(f"Không cập nhật được danh sách tài liệu: {error.message}")

    if not docs:
        st.caption("Chưa có tài liệu nào.")
    for doc in docs:
        document_row(client, doc)
    phases = [view.poll_phase(ss.first_seen.get(d.doc_id, now), now) for d in docs if d.status == "processing"]
    if "fresh" in phases and "slow" not in phases:
        st.caption("Đang xử lý; danh sách tự cập nhật.")
    elif "slow" in phases:
        st.warning("Vẫn đang xử lý. Embedding của gói miễn phí chậm (khoảng 100 đoạn mỗi phút) nên tài liệu dài có thể mất vài phút. "
                   "Bạn vẫn hỏi được các tài liệu đã sẵn sàng.")
    elif phases and set(phases) == {"stopped"}:
        st.warning("Đã dừng tự động kiểm tra sau 3 phút. Tài liệu có thể vẫn đang được xử lý ở máy chủ.")
        if st.button("Kiểm tra lại", key="recheck"):
            ss.first_seen = {}
            st.rerun()

    # A full rerun refreshes the chat scope and recomputes whether polling is still needed.
    if standalone and (ss.docs_sig != before or (ss.polling_active and not view.still_polling(docs, ss.first_seen, now))):
        st.rerun()


# ---------------------------------------------------------------------- main area: chat


def render_result(result: ChatResult) -> None:
    if result.used_fallback:
        st.info("**Không đủ bằng chứng trong tài liệu để trả lời.**\n\n" + view.escape_markdown(result.answer))
        st.caption(view.fallback_score_text(result.best_score))
    else:
        st.markdown(view.answer_markdown(result.answer))
        if result.truncated:
            st.warning("Câu trả lời có thể đã bị cắt do giới hạn độ dài.")
        if result.sources:
            with st.expander(f"Nguồn ({len(result.sources)})"):
                if not any(s.cited for s in result.sources):
                    st.caption("Câu trả lời không trích dẫn nguồn nào; các đoạn dưới đây chỉ là phần được truy xuất.")
                for source in result.sources:
                    st.markdown(view.source_header(source))
                    if source.excerpt:
                        st.markdown(view.excerpt_quote(source))
    if result.latency:
        st.caption(view.latency_caption(result.latency, result.used_fallback))
    if not result.saved:
        st.warning(view.NOT_SAVED_NOTE)


def render_body(message: dict) -> None:
    if message["role"] == "user":
        st.markdown(view.escape_markdown(message["content"]))
    elif "result" in message:
        render_result(message["result"])
    else:
        getattr(st, message.get("level", "error"))(message["error"])


def chat_area(client: ApiClient, scope: list[str], disabled_reason: str | None, show_hint: bool = True) -> None:
    ss = st.session_state
    if ss.pending and ss.pending["sent"]:  # the previous run was cut off while waiting for the server
        ss.messages.append({"role": "assistant", "error": INTERRUPTED, "level": "warning"})
        ss.pending = None
    st.caption(view.FOLLOW_UP_NOTE)
    if ss.history_missing:
        st.warning(view.missing_documents_note(ss.history_missing))
    question = st.chat_input("Đặt câu hỏi về tài liệu đã chọn…" if not disabled_reason else disabled_reason,
                             disabled=disabled_reason is not None, max_chars=MAX_QUESTION_CHARS)
    if question and not ss.pending:  # a disabled input returns nothing, so no separate check is needed
        ss.messages.append({"role": "user", "content": question})
        ss.pending = {"question": question, "doc_ids": list(scope), "sent": False}
    if not ss.messages and (show_hint or not disabled_reason):
        st.info(disabled_reason or "Hãy đặt câu hỏi ở ô bên dưới. Câu trả lời sẽ kèm nguồn trích dẫn [n].")
    for message in ss.messages:
        with st.chat_message(message["role"]):
            render_body(message)

    if ss.pending and not ss.pending["sent"]:
        ss.pending["sent"] = True  # set BEFORE the call: if this run is interrupted we must not send it twice
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm trong tài liệu và hỏi mô hình… (tối đa khoảng 60 giây)"):
                try:
                    result = client.chat(ss.pending["question"], ss.pending["doc_ids"], conversation_id=ss.conversation_id)
                    message = {"role": "assistant", "result": result}
                    ss.conversation_id = result.conversation_id or ss.conversation_id
                except ApiError as exc:
                    message = {"role": "assistant", "error": exc.message, "level": "warning" if exc.kind == "timeout" else "error"}
                    if exc.code == "conversation_not_found":  # deleted elsewhere: the next question starts a new conversation
                        ss.conversation_id = None
                ss.messages.append(message)  # stored before any further Streamlit call
                ss.pending = None
            render_body(message)
        if "result" in message and message["result"].saved:
            st.rerun()  # the server saved it: redraw so the sidebar list shows the new or updated conversation


# ---------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="AI Learning Assistant", page_icon="📚", layout="wide")
    init_state()
    ss, client = st.session_state, st.session_state.client

    try:
        health, down = client.health(), None
    except ApiError as exc:
        health, down = None, exc
    backend_ok = health is not None
    ss.docs_error = refresh_documents(client) if backend_ok else None
    ss.fresh_docs = backend_ok  # tells the panel that this run already fetched the list
    ss.docs_sig = tuple((d.doc_id, d.status) for d in ss.docs)
    if backend_ok:  # before any widget: it may change the chat scope and the open section
        apply_pending_load(client, [d.doc_id for d in ss.docs if d.status == "ready"])
    ss.polling_active = backend_ok and view.still_polling(ss.docs, ss.first_seen, time.monotonic())
    panel = st.fragment(documents_panel, run_every=view.POLL_INTERVAL_S if ss.polling_active else None)

    with st.sidebar:
        st.header("Tài liệu")
        uploaded = st.file_uploader("Tải tài liệu lên (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key=f"uploader_{ss.uploader_n}",
                                    disabled=not backend_ok, max_upload_size=upload_limit_mb(),
                                    help=f"Tối đa {upload_limit_mb()} MB. Tài liệu được xử lý ở nền; hãy đợi trạng thái “Sẵn sàng”.")
        if uploaded is not None and backend_ok:
            handle_upload(client, uploaded)
        show_notice()
        if backend_ok:
            panel(client)
        ready = [d for d in ss.docs if d.status == "ready"]
        ready_ids = [d.doc_id for d in ready]
        ss.scope = view.selection_after_refresh(ss.scope, ready_ids, ss.known_ready)
        ss.known_ready = set(ready_ids)
        names = {d.doc_id: d.filename for d in ready}
        st.multiselect("Hỏi trong các tài liệu", options=ready_ids, format_func=lambda i: names.get(i, i), key="scope",
                       disabled=not ready_ids, placeholder="Chưa có tài liệu sẵn sàng")
        if st.button("Xóa hội thoại", help="Chỉ xóa phần đang hiển thị và bắt đầu cuộc trò chuyện mới. Cuộc trò chuyện đã lưu vẫn còn "
                                          "trong mục Lịch sử; muốn xóa hẳn, dùng nút Xóa ở đó."):
            reset_conversation()
            st.rerun()
        st.caption("“Xóa hội thoại” chỉ làm sạch màn hình; bản đã lưu vẫn nằm trong Lịch sử.")
        if backend_ok:
            history_section(client)
        st.caption("Bản demo chạy cục bộ: không có đăng nhập, mọi người truy cập đều thấy cùng một bộ tài liệu và cùng một lịch sử.")

    st.title("AI Learning Assistant")
    st.caption("Cách hoạt động: tài liệu được chia nhỏ, chuyển thành vector và lưu vào Qdrant (xử lý nền). Câu hỏi được so với các "
               "đoạn gần nhất; mô hình chỉ trả lời dựa trên các đoạn đó và ghi nguồn [n]. Nếu bằng chứng không đủ, hệ thống từ chối "
               "thay vì đoán.")
    if down:
        st.error(down.message)
        st.button("Thử kết nối lại")
    elif health and not health.qdrant_reachable:
        st.warning("Cơ sở dữ liệu vector (Qdrant) không kết nối được: tải lên và hỏi đáp có thể báo lỗi cho tới khi Qdrant chạy lại.")

    if down:
        reason = "Không kết nối được máy chủ API"
    elif not ready_ids:
        reason = "Hãy tải lên một tài liệu và đợi trạng thái “Sẵn sàng”"
    elif not ss.scope:
        reason = "Hãy chọn ít nhất một tài liệu ở thanh bên"
    else:
        reason = None
    if down:
        chat_area(client, list(ss.scope), reason, show_hint=False)  # the red banner already explains an outage
        return
    section = st.radio("Chức năng", SECTIONS, horizontal=True, key="section", label_visibility="collapsed")
    if section == "Tóm tắt":
        summary_area(client, ready)
    elif section == "Trắc nghiệm":
        quiz_area(client, ready)
    else:
        chat_area(client, list(ss.scope), reason)


main()
