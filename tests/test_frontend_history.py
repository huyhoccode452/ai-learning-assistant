"""UI tests (AppTest) for saved conversations: save while chatting, list, open, new, clear, delete. Fake client only:
no browser, no server, no model."""

import re
from dataclasses import replace

import pytest

from frontend import view
from frontend.api_client import ApiError, HistoryMessage, Source
from tests.test_frontend_app import A, B, FakeClient, answer, ask, doc, no_exception, sidebar_button, source, start, texts

C1, C2, C3 = "1" * 32, "2" * 32, "3" * 32


def save(fake: FakeClient, conversation_id: str, title: str, *, docs=(A,), reply=None, questions: int = 1) -> None:
    """A conversation that is already saved on the 'server' when the page starts."""
    reply = reply or answer()
    fake._tick += 1
    stamp = "2026-11-05T08:00:00+00:00"
    messages = []
    for n in range(questions):
        question = title if n == 0 else f"{title} (câu {n + 1})"
        messages += [HistoryMessage("user", question, stamp, None),
                     HistoryMessage("assistant", reply.answer, stamp, replace(reply, conversation_id=conversation_id))]
    fake.conversations[conversation_id] = {"title": title, "doc_ids": list(docs), "tick": fake._tick, "messages": messages}


def open_(at, conversation_id: str):
    return sidebar_button(at, f"conv_open_{conversation_id}").click().run()


def sidebar_text(at) -> str:
    return "\n".join(e.value for group in (at.sidebar.markdown, at.sidebar.caption, at.sidebar.warning, at.sidebar.success, at.sidebar.error, at.sidebar.subheader) for e in group)


# ====================================================================== the section and the notes


def test_an_empty_history_says_so_and_the_notes_explain_what_history_is_and_is_not() -> None:
    at = start(FakeClient([doc()]))
    no_exception(at)
    text = sidebar_text(at)
    assert "Lịch sử" in text and "Chưa có cuộc trò chuyện nào được lưu." in text
    assert at.sidebar.button(key="new_conversation") is not None
    assert "Xóa tài liệu không xóa lịch sử" in text and "không có đăng nhập nên mọi người dùng chung một lịch sử" in text
    assert "“Xóa hội thoại” chỉ làm sạch màn hình; bản đã lưu vẫn nằm trong Lịch sử." in text
    assert "cùng một lịch sử" in text
    assert any(view.FOLLOW_UP_NOTE == c.value for c in at.caption)  # follow-ups do not use previous messages
    assert "không dùng các tin nhắn trước" in view.FOLLOW_UP_NOTE and "Lịch sử chỉ để xem lại" in view.FOLLOW_UP_NOTE


def test_the_follow_up_note_is_shown_in_the_chat_but_not_in_the_other_sections() -> None:
    at = start(FakeClient([doc()]))
    at.radio(key="section").set_value("Tóm tắt").run()
    assert not any(view.FOLLOW_UP_NOTE == c.value for c in at.caption)


def test_listed_conversations_show_title_date_size_and_documents() -> None:
    fake = FakeClient([doc(A), doc(B, "b.pdf")])
    save(fake, C1, "work_mem là gì?", docs=(A, B), questions=2)
    text = sidebar_text(start(fake))
    assert "**work\\_mem là gì?**" in text  # escaped: a title is the user's text, shown literally
    assert re.search(r"\d\d/\d\d/2026 \d\d:\d\d · 4 tin nhắn · 2 tài liệu", text)  # the time is shown in this computer's zone


def test_a_conversation_with_deleted_documents_is_listed_with_how_many_are_gone() -> None:
    fake = FakeClient([doc(A)])
    save(fake, C1, "cũ", docs=(A, B))  # B is not on the server any more
    assert "2 tài liệu · 1 đã bị xóa" in sidebar_text(start(fake))


def test_markdown_in_a_title_is_not_interpreted() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "**đậm** [x](http://evil.test) $a$")
    text = sidebar_text(start(fake))
    assert "\\*\\*đậm\\*\\* \\[x\\](http\\://evil.test) \\$a\\$" in text


# ====================================================================== saving while chatting


def test_asking_a_question_saves_a_conversation_and_lists_it_as_open() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "work_mem mặc định là gì?")
    no_exception(at)
    assert fake.chat_conversation_ids == [None] and len(fake.conversations) == 1
    (conversation_id,) = fake.conversations
    assert at.session_state["conversation_id"] == conversation_id
    text = sidebar_text(at)
    assert "work\\_mem mặc định là gì?" in text and "Đang mở" in text and "2 tin nhắn" in text
    assert sidebar_button_exists(at, f"conv_open_{conversation_id}") and sidebar_button(at, f"conv_open_{conversation_id}").disabled
    assert [m.name for m in at.chat_message] == ["user", "assistant"]  # the answer is still on screen after the redraw


def sidebar_button_exists(at, key: str) -> bool:
    return any(b.key == key for b in at.sidebar.button)


def test_the_next_question_goes_to_the_same_conversation() -> None:
    fake = FakeClient([doc()])
    at = ask(ask(start(fake), "câu một"), "câu hai")
    (conversation_id,) = fake.conversations
    assert fake.chat_conversation_ids == [None, conversation_id]
    assert "4 tin nhắn" in sidebar_text(at) and len(fake.conversations) == 1
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]


def test_reruns_do_not_save_the_same_question_again() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "chỉ một lần")
    for _ in range(4):
        at.run()
    assert len(fake.chat_calls) == 1 and len(fake.conversations) == 1
    assert len(next(iter(fake.conversations.values()))["messages"]) == 2


def test_a_question_cut_off_mid_request_is_not_resent_and_points_to_the_history() -> None:
    fake = FakeClient([doc()])
    state = {"messages": [{"role": "user", "content": "đang hỏi"}], "pending": {"question": "đang hỏi", "doc_ids": [A], "sent": True}}
    at = start(fake, **state)
    assert fake.chat_calls == [] and any("Lịch sử" in w.value and "bị gián đoạn" in w.value for w in at.warning)


def test_when_the_server_cannot_save_the_answer_is_shown_with_a_warning() -> None:
    fake = FakeClient([doc()])
    fake.save_chats = False
    at = ask(start(fake), "câu hỏi quan trọng")
    no_exception(at)
    assert "work_mem mặc định là 4MB [1]." in texts(at)  # the answer is there
    assert any(view.NOT_SAVED_NOTE == w.value for w in at.warning)
    assert fake.conversations == {} and at.session_state["conversation_id"] is None
    at = ask(at, "câu hai")
    assert fake.chat_conversation_ids == [None, None]  # nothing to continue: the next one tries a new conversation


def test_a_conversation_deleted_behind_the_page_is_reported_and_the_next_question_starts_a_new_one() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "câu một")
    fake.conversations.clear()  # deleted from another tab
    at = ask(at, "câu hai")
    no_exception(at)
    assert any("Không tìm thấy cuộc trò chuyện." == e.value for e in at.error) and at.session_state["conversation_id"] is None
    at = ask(at, "câu ba")
    assert fake.chat_conversation_ids[-1] is None and len(fake.conversations) == 1


# ====================================================================== opening a saved conversation


def test_opening_a_conversation_shows_its_messages_without_calling_the_model() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "câu hỏi đã lưu")
    at = open_(start(fake), C1)
    no_exception(at)
    assert fake.detail_calls == [C1] and fake.chat_calls == [] and fake.summary_calls == [] and fake.quiz_calls == []
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    page = texts(at)
    assert "câu hỏi đã lưu" in page and "work_mem mặc định là 4MB [1]." in page
    assert at.session_state["conversation_id"] == C1 and "Đang mở" in sidebar_text(at)
    assert sidebar_button(at, f"conv_open_{C1}").disabled


def test_a_loaded_answer_renders_exactly_like_the_live_one() -> None:
    fake = FakeClient([doc()])
    live = ask(start(fake), "câu hỏi")
    live_view = ([e.label for e in live.expander], [m.value for m in live.expander[0].markdown], [c.value for c in live.caption if "Thời gian" in c.value])
    (conversation_id,) = fake.conversations
    reopened = open_(sidebar_reset(start(fake)), conversation_id)
    loaded_view = ([e.label for e in reopened.expander], [m.value for m in reopened.expander[0].markdown], [c.value for c in reopened.caption if "Thời gian" in c.value])
    assert loaded_view == live_view and live_view[0] == ["Nguồn (2)"] and len(live_view[2]) == 1  # sources, cited flags, latency: same


def sidebar_reset(at):
    return sidebar_button(at, "new_conversation").click().run()


def test_a_loaded_fallback_answer_shows_the_info_box_and_score() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "pg_upgrade?", reply=answer(answer="Tôi không tìm thấy thông tin.", sources=(), used_fallback=True, best_score=0.5427))
    at = open_(start(fake), C1)
    assert any("Không đủ bằng chứng" in i.value for i in at.info) and "0.543" in texts(at)


def test_a_source_of_a_deleted_document_is_marked_not_hidden() -> None:
    fake = FakeClient([doc(B, "b.pdf")])  # document A is gone
    gone = replace(source(1, True), document_exists=False)
    save(fake, C1, "câu cũ", docs=(A, B), reply=answer(sources=(gone, source(2, False))))
    at = open_(start(fake), C1)
    no_exception(at)
    sources = [m.value for m in at.expander[0].markdown]
    assert any("[1]" in s and "Tài liệu đã bị xóa" in s for s in sources)  # flagged
    assert not any("[2]" in s and "Tài liệu đã bị xóa" in s for s in sources)  # the other one is not
    assert any("work\\_mem mặc định 4MB" in s for s in sources)  # and the excerpt is still there
    assert any("1 tài liệu của cuộc trò chuyện này đã bị xóa" in w.value for w in at.warning)


def test_opening_restores_the_collection_that_still_exists_and_the_chat_section() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    save(fake, C1, "chỉ tài liệu B", docs=(B,))
    at = start(fake)
    at = at.radio(key="section").set_value("Trắc nghiệm").run()  # opened from another section
    at = open_(at, C1)
    assert at.session_state["section"] == "Hỏi đáp" and at.multiselect[0].value == [B]


def test_the_scope_is_left_alone_when_none_of_its_documents_remain() -> None:
    fake = FakeClient([doc(A, "a.pdf")])
    save(fake, C1, "tài liệu đã xóa hết", docs=(B,))
    at = open_(start(fake), C1)
    assert at.multiselect[0].value == [A]


def test_a_follow_up_after_opening_is_saved_in_that_conversation() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "câu đầu")
    save(fake, C2, "cuộc khác")
    at = open_(start(fake), C1)
    at = ask(at, "câu tiếp theo")
    assert fake.chat_conversation_ids == [C1] and len(fake.conversations) == 2
    assert len(fake.conversations[C1]["messages"]) == 4 and len(fake.conversations[C2]["messages"]) == 2
    assert "Đang mở" in sidebar_text(at)


def test_opening_a_conversation_that_cannot_be_read_reports_it_and_keeps_the_screen() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "không mở được")
    at = ask(start(fake), "đang xem dở")
    fake.detail_error = ApiError("http", "Không tìm thấy cuộc trò chuyện.", status=404, code="conversation_not_found")
    at = open_(at, C1)
    no_exception(at)
    assert any("Không mở được cuộc trò chuyện: Không tìm thấy cuộc trò chuyện." in e.value for e in at.sidebar.error)
    assert [m.name for m in at.chat_message] == ["user", "assistant"] and "đang xem dở" in texts(at)  # the screen is untouched
    assert at.session_state["conversation_id"] != C1


def test_loading_history_makes_the_list_call_but_never_a_model_call() -> None:
    fake = FakeClient([doc()])
    for cid in (C1, C2, C3):
        save(fake, cid, f"cuộc {cid[0]}")
    at = start(fake)
    for cid in (C1, C2, C3, C1):
        at = open_(at, cid) if not sidebar_button(at, f"conv_open_{cid}").disabled else at
    assert fake.chat_calls == [] and fake.summary_calls == [] and fake.quiz_calls == [] and fake.list_conversation_calls


def test_opening_a_conversation_does_not_drag_a_cut_off_question_into_it() -> None:
    """A question left 'sent' by an interrupted run would be reported as interrupted in the NEXT chat view: after
    opening another conversation that message would be attached to it."""
    fake = FakeClient([doc()])
    save(fake, C1, "cuộc đã lưu")
    state = {"messages": [{"role": "user", "content": "câu dở dang"}], "pending": {"question": "câu dở dang", "doc_ids": [A], "sent": True}}
    at = start(fake, **state)  # the page reports the interruption once...
    at.session_state["pending"] = {"question": "lại dở dang", "doc_ids": [A], "sent": True}  # ...and another run is cut off right now
    at = open_(at, C1)
    assert at.session_state["pending"] is None
    assert not any("Yêu cầu trước đó bị gián đoạn" in w.value for w in at.warning)  # no stray interruption notice on this conversation
    assert [m.name for m in at.chat_message] == ["user", "assistant"] and all("lại dở dang" not in m.markdown[0].value for m in at.chat_message)


def test_the_history_never_asks_for_more_than_the_api_allows() -> None:
    fake = FakeClient([doc()])
    for n in range(120):
        save(fake, f"{n:032x}", f"cuộc {n}")
    at = start(fake, history_limit=150)
    assert fake.list_conversation_calls[-1] == (100, 0)  # the API's page limit, not 150
    assert len([b for b in at.sidebar.button if (b.key or "").startswith("conv_open_")]) == 100
    assert "Chỉ hiển thị 100 cuộc trò chuyện gần nhất (tổng 120)." in sidebar_text(at) and not sidebar_button_exists(at, "history_more")


def test_an_unreadable_loaded_message_does_not_break_the_page() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "câu hỏi")
    messages = fake.conversations[C1]["messages"]
    fake.conversations[C1]["messages"][1] = replace(messages[1], result=replace(messages[1].result, latency=None, sources=()))
    at = open_(start(fake), C1)
    no_exception(at)
    assert "work_mem mặc định là 4MB [1]." in texts(at) and not any("Thời gian" in c.value for c in at.caption)


# ====================================================================== new conversation, clear, delete


def test_a_new_conversation_empties_the_screen_and_the_next_question_is_saved_separately() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "câu một")
    (first,) = fake.conversations
    at = sidebar_reset(at)
    assert not at.chat_message and at.session_state["conversation_id"] is None
    assert fake.deleted_conversations == [] and first in fake.conversations  # the saved one is untouched
    at = ask(at, "câu hai")
    assert fake.chat_conversation_ids == [None, None] and len(fake.conversations) == 2


def test_clear_chat_only_clears_the_screen_and_keeps_the_saved_conversation() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "câu một")
    (conversation_id,) = fake.conversations
    at = next(b for b in at.sidebar.button if b.label == "Xóa hội thoại").click().run()
    assert not at.chat_message and fake.deleted_conversations == [] and conversation_id in fake.conversations
    assert "câu một" in sidebar_text(at)  # still listed in the history
    assert at.session_state["conversation_id"] is None  # the next question starts a new saved conversation
    reopened = open_(at, conversation_id)
    assert [m.name for m in reopened.chat_message] == ["user", "assistant"]  # and it can be opened again, complete


def test_delete_asks_for_confirmation_first() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "để xóa")
    at = sidebar_button(start(fake), f"conv_del_{C1}").click().run()
    assert fake.deleted_conversations == [] and any("Xóa cuộc trò chuyện này khỏi lịch sử?" in w.value for w in at.sidebar.warning)
    at = sidebar_button(at, f"conv_no_{C1}").click().run()  # cancel
    assert fake.deleted_conversations == [] and C1 in fake.conversations and not at.sidebar.warning
    at = sidebar_button(at, f"conv_del_{C1}").click().run()
    at = sidebar_button(at, f"conv_yes_{C1}").click().run()
    no_exception(at)
    assert fake.deleted_conversations == [C1] and C1 not in fake.conversations
    text = sidebar_text(at)
    assert "Đã xóa cuộc trò chuyện «để xóa» (2 tin nhắn)." in text and "Chưa có cuộc trò chuyện nào được lưu." in text


def test_deleting_another_conversation_keeps_the_open_one_on_screen() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "đang mở")
    save(fake, C2, "sẽ xóa")
    at = open_(start(fake), C1)
    at = sidebar_button(sidebar_button(at, f"conv_del_{C2}").click().run(), f"conv_yes_{C2}").click().run()
    assert fake.deleted_conversations == [C2] and [m.name for m in at.chat_message] == ["user", "assistant"]
    assert at.session_state["conversation_id"] == C1


def test_deleting_the_open_conversation_clears_the_screen_and_detaches() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "đang mở")
    at = open_(start(fake), C1)
    at = sidebar_button(sidebar_button(at, f"conv_del_{C1}").click().run(), f"conv_yes_{C1}").click().run()
    assert not at.chat_message and at.session_state["conversation_id"] is None and fake.deleted_conversations == [C1]
    at = ask(at, "câu mới")
    assert fake.chat_conversation_ids == [None]  # not into the deleted conversation


def test_a_failed_delete_is_reported_and_the_conversation_stays() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "khó xóa")
    at = sidebar_button(start(fake), f"conv_del_{C1}").click().run()
    fake.delete_conversation_error = ApiError("http", "Dịch vụ tạm thời không khả dụng. Hãy thử lại sau.", status=503)
    at = sidebar_button(at, f"conv_yes_{C1}").click().run()
    no_exception(at)
    assert any("Không xóa được cuộc trò chuyện «khó xóa»: Dịch vụ tạm thời không khả dụng." in e.value for e in at.sidebar.error)
    assert C1 in fake.conversations and sidebar_button_exists(at, f"conv_open_{C1}")


# ====================================================================== paging, failures


def test_the_list_shows_ten_and_more_on_request() -> None:
    fake = FakeClient([doc()])
    for n in range(12):
        save(fake, f"{n:032x}", f"cuộc số {n}")
    at = start(fake)
    shown = [b for b in at.sidebar.button if (b.key or "").startswith("conv_open_")]
    assert len(shown) == 10 and sidebar_button_exists(at, "history_more")
    at = at.sidebar.button(key="history_more").click().run()
    assert len([b for b in at.sidebar.button if (b.key or "").startswith("conv_open_")]) == 12
    assert not any(b.key == "history_more" for b in at.sidebar.button)
    assert [call[0] for call in fake.list_conversation_calls if call[0] != 10] == [20]  # one request per run, 10 then 20


def test_the_newest_conversation_is_listed_first() -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "cũ nhất")
    save(fake, C2, "mới nhất")
    keys = [b.key for b in start(fake).sidebar.button if (b.key or "").startswith("conv_open_")]
    assert keys == [f"conv_open_{C2}", f"conv_open_{C1}"]


def test_a_history_that_cannot_be_loaded_does_not_break_the_rest_of_the_page() -> None:
    fake = FakeClient([doc()])
    fake.history_error = ApiError("http", "Dịch vụ tạm thời không khả dụng. Hãy thử lại sau.", status=503)
    at = start(fake)
    no_exception(at)
    assert "Không tải được lịch sử: Dịch vụ tạm thời không khả dụng." in sidebar_text(at)
    assert not at.chat_input[0].disabled  # chatting still works
    assert ask(at, "vẫn hỏi được").chat_message


def test_no_history_section_when_the_backend_is_down() -> None:
    fake = FakeClient([doc()], health=ApiError("unreachable", "Không kết nối được tới máy chủ API."))
    at = start(fake)
    no_exception(at)
    assert fake.list_conversation_calls == []  # the history is not even requested
    assert not sidebar_button_exists(at, "new_conversation") and "Chưa có cuộc trò chuyện nào được lưu." not in sidebar_text(at)


@pytest.mark.parametrize("section", ["Tóm tắt", "Trắc nghiệm"])
def test_the_history_stays_in_the_sidebar_in_every_section(section: str) -> None:
    fake = FakeClient([doc()])
    save(fake, C1, "luôn thấy")
    at = start(fake).radio(key="section").set_value(section).run()
    assert "luôn thấy" in sidebar_text(at)
