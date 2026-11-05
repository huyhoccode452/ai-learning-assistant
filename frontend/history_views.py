"""The "Lịch sử" (saved conversations) section of the sidebar.

Where things live:
  * the SERVER stores every chat exchange (the backend saves it while answering), so history survives a browser
    reload, a new tab and a restart of this Streamlit process;
  * `st.session_state` only says which conversation is open (`conversation_id`) and holds what is on screen (`messages`).
Opening a saved conversation only calls GET /conversations/{id}: it never asks the model anything. Because a Streamlit
widget cannot be changed after it was drawn in the same run, a click only records `pending_load`; the next run applies
it at the top of the page (see `apply_pending_load`), before the sidebar widgets exist.
"""

import streamlit as st

from frontend import view
from frontend.api_client import ApiClient, ApiError, ConversationInfo

HISTORY_PAGE = 10  # entries added by each "Xem thêm"
HISTORY_MAX = 100  # the API's page limit: older conversations are not listed (the note says so)
UNTITLED = "(không có tiêu đề)"


def reset_conversation() -> None:
    """Empty screen and no open conversation: the next question starts a NEW saved conversation. Nothing saved is deleted."""
    ss = st.session_state
    ss.conversation_id, ss.messages, ss.pending, ss.history_missing = None, [], None, 0


def apply_pending_load(client: ApiClient, ready_ids: list[str]) -> None:
    """Open the conversation chosen by the last click. Call before any sidebar widget is created."""
    ss = st.session_state
    conversation_id, ss.pending_load = ss.pending_load, None
    if not conversation_id:
        return
    try:
        detail = client.get_conversation(conversation_id)  # reading only: no LLM call
    except ApiError as exc:
        ss.notice = ("error", f"Không mở được cuộc trò chuyện: {exc.message}")
        return
    ss.messages = view.messages_from_history(detail)
    ss.pending = None
    ss.conversation_id = detail.info.id
    ss.history_missing = len(detail.info.missing_doc_ids)
    ss.section = "Hỏi đáp"
    still_there = [d for d in detail.info.doc_ids if d in ready_ids]
    if still_there:  # a follow-up searches the same collection, as far as it still exists
        ss.scope = still_there


def _delete(client: ApiClient, info: ConversationInfo) -> None:
    ss = st.session_state
    title = view.escape_markdown(info.title or UNTITLED)
    try:
        removed = client.delete_conversation(info.id)
    except ApiError as exc:
        ss.notice = ("error", f"Không xóa được cuộc trò chuyện «{title}»: {exc.message}")
    else:
        ss.notice = ("success", f"Đã xóa cuộc trò chuyện «{title}» ({removed} tin nhắn).")
        if ss.conversation_id == info.id:  # it was open: the screen no longer shows a saved conversation
            reset_conversation()
    ss.confirm_delete_conversation = None
    st.rerun()


def _entry(client: ApiClient, info: ConversationInfo) -> None:
    ss = st.session_state
    with st.container(border=True):
        is_open = info.id == ss.conversation_id
        st.markdown(f"**{view.escape_markdown(info.title or UNTITLED)}**" + ("  :blue-badge[Đang mở]" if is_open else ""))
        st.caption(view.history_caption(info))
        if ss.confirm_delete_conversation == info.id:
            st.warning("Xóa cuộc trò chuyện này khỏi lịch sử? Không thể hoàn tác.")
            yes, no = st.columns(2)
            if yes.button("Xác nhận xóa", key=f"conv_yes_{info.id}", type="primary"):
                _delete(client, info)
            if no.button("Hủy", key=f"conv_no_{info.id}"):
                ss.confirm_delete_conversation = None
                st.rerun()
            return
        open_, delete = st.columns(2)
        if open_.button("Mở", key=f"conv_open_{info.id}", disabled=is_open):
            ss.pending_load = info.id
            st.rerun()
        if delete.button("Xóa", key=f"conv_del_{info.id}"):
            ss.confirm_delete_conversation = info.id
            st.rerun()


def history_section(client: ApiClient) -> None:
    ss = st.session_state
    st.subheader("Lịch sử")
    if st.button("Cuộc trò chuyện mới", key="new_conversation"):
        reset_conversation()
        st.rerun()
    st.caption("Các cuộc trò chuyện được lưu trên máy chủ (không có đăng nhập nên mọi người dùng chung một lịch sử) và còn lại sau "
               "khi tải lại trang hoặc khởi động lại ứng dụng. Xóa tài liệu không xóa lịch sử.")
    limit = min(ss.history_limit, HISTORY_MAX)
    try:
        page = client.list_conversations(limit=limit, offset=0)
    except ApiError as exc:
        st.caption(f"Không tải được lịch sử: {exc.message}")
        return
    if not page.items:
        st.caption("Chưa có cuộc trò chuyện nào được lưu.")
        return
    for info in page.items:
        _entry(client, info)
    if page.total > len(page.items):
        if limit < HISTORY_MAX:
            if st.button("Xem thêm", key="history_more"):
                ss.history_limit += HISTORY_PAGE
                st.rerun()
        else:
            st.caption(f"Chỉ hiển thị {HISTORY_MAX} cuộc trò chuyện gần nhất (tổng {page.total}).")
