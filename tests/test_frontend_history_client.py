"""frontend/api_client.py and frontend/view.py for saved conversations (fake HTTP layer, no network)."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from frontend.api_client import ConversationDetail, ConversationInfo, HistoryMessage, Latency, Source
from frontend.view import (
    FOLLOW_UP_NOTE,
    NOT_SAVED_NOTE,
    format_time,
    history_caption,
    messages_from_history,
    missing_documents_note,
    source_header,
)
from tests.test_frontend_api_client import FakeResponse, client, error_body, failure

CID = "c" * 32
LATENCY = {"embed_ms": 100.0, "search_ms": 20.0, "llm_ms": 1500.0, "total_ms": 1620.0}
SOURCE = {"number": 1, "doc_id": "a" * 16, "filename": "a.pdf", "page": 5, "chunk_index": 3, "score": 0.83, "excerpt": "work_mem 4MB", "cited": True}
INFO = {"id": CID, "title": "work_mem là gì?", "doc_ids": ["a" * 16, "b" * 16], "missing_doc_ids": ["b" * 16],
        "created_at": "2026-11-05T08:00:00.123456+00:00", "updated_at": "2026-11-05T08:30:00.123456+00:00", "message_count": 4}
DETAIL = {**INFO, "messages": [
    {"id": 1, "role": "user", "content": "work_mem là gì?", "created_at": "2026-11-05T08:00:00+00:00", "sources": [], "used_fallback": None,
     "best_score": None, "truncated": None, "latency": None},
    {"id": 2, "role": "assistant", "content": "Là 4MB [1].", "created_at": "2026-11-05T08:00:01+00:00",
     "sources": [{**SOURCE, "document_exists": False}, {**SOURCE, "number": 2, "cited": False, "document_exists": True}],
     "used_fallback": False, "best_score": 0.83, "truncated": False, "latency": LATENCY},
]}
CHAT = {"answer": "Là 4MB [1].", "used_fallback": False, "best_score": 0.83, "truncated": False, "latency": LATENCY, "sources": [SOURCE]}


# ---------------------------------------------------------------------- chat carries the conversation


def test_chat_sends_the_conversation_id_only_when_there_is_one() -> None:
    api, session = client(FakeResponse(200, CHAT), FakeResponse(200, CHAT), FakeResponse(200, CHAT))
    api.chat("q")
    api.chat("q", ["a" * 16])
    api.chat("q", None, conversation_id=CID)
    assert [c["json"] for c in session.calls] == [{"question": "q"}, {"question": "q", "doc_ids": ["a" * 16]}, {"question": "q", "conversation_id": CID}]


def test_the_chat_reply_reports_where_it_was_saved() -> None:
    saved = client(FakeResponse(200, {**CHAT, "conversation_id": CID, "saved": True}))[0].chat("q")
    unsaved = client(FakeResponse(200, {**CHAT, "conversation_id": None, "saved": False}))[0].chat("q")
    assert (saved.conversation_id, saved.saved) == (CID, True) and (unsaved.conversation_id, unsaved.saved) == (None, False)
    legacy = client(FakeResponse(200, CHAT))[0].chat("q")  # a server without history: nothing to warn about
    assert (legacy.conversation_id, legacy.saved) == (None, True) and legacy.sources[0].document_exists is True


# ---------------------------------------------------------------------- list, open, delete


def test_listing_conversations_sends_the_paging_and_parses_the_page() -> None:
    api, session = client(FakeResponse(200, {"items": [INFO], "total": 31, "limit": 10, "offset": 20}))
    page = api.list_conversations(limit=10, offset=20)
    call = session.calls[0]
    assert (call["method"], call["url"], call["params"], call["timeout"]) == ("GET", "http://api.test:8000/conversations", {"limit": 10, "offset": 20}, (5.0, 10.0))
    assert (page.total, page.limit, page.offset) == (31, 10, 20)
    info = page.items[0]
    assert (info.id, info.title, info.doc_ids, info.missing_doc_ids, info.message_count) == (CID, "work_mem là gì?", ("a" * 16, "b" * 16), ("b" * 16,), 4)
    assert info.updated_at.startswith("2026-11-05T08:30")


def test_listing_defaults_to_the_first_twenty() -> None:
    api, session = client(FakeResponse(200, {"items": [], "total": 0, "limit": 20, "offset": 0}))
    assert api.list_conversations().items == ()
    assert session.calls[0]["params"] == {"limit": 20, "offset": 0}


def test_opening_a_conversation_returns_the_messages_in_the_live_shape() -> None:
    api, session = client(FakeResponse(200, DETAIL))
    detail = api.get_conversation(CID)
    assert (session.calls[0]["method"], session.calls[0]["url"]) == ("GET", f"http://api.test:8000/conversations/{CID}")
    assert detail.info.id == CID and [m.role for m in detail.messages] == ["user", "assistant"]
    user, assistant = detail.messages
    assert (user.content, user.result) == ("work_mem là gì?", None)
    result = assistant.result
    assert result is not None and (result.answer, result.used_fallback, result.best_score, result.truncated) == ("Là 4MB [1].", False, 0.83, False)
    assert result.latency == Latency(100.0, 20.0, 1500.0, 1620.0) and result.saved is True and result.conversation_id == CID
    assert [(s.number, s.cited, s.document_exists) for s in result.sources] == [(1, True, False), (2, False, True)]
    assert result.sources[0].filename == "a.pdf" and result.sources[0].excerpt == "work_mem 4MB" and result.sources[0].page == 5


def test_a_message_without_latency_or_sources_is_still_readable() -> None:
    bare = {**DETAIL, "messages": [{"id": 2, "role": "assistant", "content": "x", "created_at": "t", "sources": [], "used_fallback": True,
                                    "best_score": None, "truncated": None, "latency": None}]}
    result = client(FakeResponse(200, bare))[0].get_conversation(CID).messages[0].result
    assert result is not None and (result.latency, result.sources, result.used_fallback, result.best_score, result.truncated) == (None, (), True, None, False)


@pytest.mark.parametrize("broken", [{"unexpected": 1}, {**DETAIL, "messages": [{"role": "assistant"}]}, {**DETAIL, "message_count": "many"}])
def test_an_unusable_conversation_reply_is_a_bad_response(broken: dict) -> None:
    assert failure(lambda api: api.get_conversation(CID), FakeResponse(200, broken)).kind == "bad_response"
    assert failure(lambda api: api.list_conversations(), FakeResponse(200, {"items": [{"id": 1}], "total": 1, "limit": 1, "offset": 0})).kind == "bad_response"


def test_deleting_returns_how_many_messages_went_with_it() -> None:
    api, session = client(FakeResponse(200, {"conversation_id": CID, "deleted_messages": 4}))
    assert api.delete_conversation(CID) == 4
    assert (session.calls[0]["method"], session.calls[0]["url"], session.calls[0]["timeout"]) == ("DELETE", f"http://api.test:8000/conversations/{CID}", (5.0, 10.0))


@pytest.mark.parametrize("call", [lambda api: api.get_conversation(CID), lambda api: api.delete_conversation(CID)])
def test_a_missing_conversation_shows_the_backends_message(call) -> None:
    error = failure(call, FakeResponse(404, error_body("conversation_not_found", "Không tìm thấy cuộc trò chuyện.")))
    assert (error.kind, error.status, error.code, error.message) == ("http", 404, "conversation_not_found", "Không tìm thấy cuộc trò chuyện.")


def test_nothing_secret_is_sent_with_history_requests() -> None:
    api, session = client(FakeResponse(200, {"items": [], "total": 0, "limit": 20, "offset": 0}), FakeResponse(200, DETAIL),
                          FakeResponse(200, {"conversation_id": CID, "deleted_messages": 0}))
    api.list_conversations()
    api.get_conversation(CID)
    api.delete_conversation(CID)
    assert all("headers" not in c and "auth" not in c and "json" not in c for c in session.calls)


# ---------------------------------------------------------------------- presentation helpers


def test_format_time_uses_the_given_zone_and_survives_garbage() -> None:
    assert format_time("2026-11-05T08:30:00.123456+00:00", UTC) == "05/11/2026 08:30"
    assert format_time("2026-11-05T08:30:00+00:00", timezone(timedelta(hours=7))) == "05/11/2026 15:30"
    assert format_time("2026-11-05T23:30:00+00:00", timezone(timedelta(hours=7))) == "06/11/2026 06:30"  # the date moves too
    assert format_time("not a date") == "not a date" and format_time("") == ""
    local = format_time("2026-11-05T08:30:00+00:00")  # this computer's zone: only the shape is stable
    assert len(local) == 16 and local[2] == "/" and datetime.strptime(local, "%d/%m/%Y %H:%M")


def info(**changes) -> ConversationInfo:
    values = dict(id=CID, title="t", doc_ids=("a" * 16, "b" * 16), missing_doc_ids=(), created_at="x", updated_at="2026-11-05T08:30:00+00:00", message_count=4)
    values.update(changes)
    return ConversationInfo(**values)


def test_history_caption() -> None:
    assert history_caption(info(), UTC) == "05/11/2026 08:30 · 4 tin nhắn · 2 tài liệu"
    assert history_caption(info(missing_doc_ids=("b" * 16,)), UTC) == "05/11/2026 08:30 · 4 tin nhắn · 2 tài liệu · 1 đã bị xóa"
    assert history_caption(info(doc_ids=(), message_count=0), UTC) == "05/11/2026 08:30 · 0 tin nhắn · 0 tài liệu"


def test_a_deleted_document_is_marked_in_the_source_line_and_only_there() -> None:
    kept = Source(1, "a.pdf", 5, 0.83, "x", True)
    gone = Source(1, "a.pdf", 5, 0.83, "x", True, document_exists=False)
    assert source_header(kept) == "**[1]** a.pdf · trang 5 · điểm 0.830 · :green-badge[Được trích dẫn]"
    assert source_header(gone) == source_header(kept) + " · :red-badge[Tài liệu đã bị xóa]"


def test_the_notes_say_what_history_does_not_do() -> None:
    assert "độc lập" in FOLLOW_UP_NOTE and "không dùng các tin nhắn trước" in FOLLOW_UP_NOTE and "câu hỏi nối tiếp" in FOLLOW_UP_NOTE
    assert "chưa được lưu" in NOT_SAVED_NOTE and "vẫn dùng được" in NOT_SAVED_NOTE
    assert "3 tài liệu" in missing_documents_note(3) and "Tài liệu đã bị xóa" in missing_documents_note(3)


def test_saved_messages_become_live_messages() -> None:
    detail = client(FakeResponse(200, DETAIL))[0].get_conversation(CID)
    messages = messages_from_history(detail)
    assert [m["role"] for m in messages] == ["user", "assistant"] and messages[0] == {"role": "user", "content": "work_mem là gì?"}
    assert messages[1]["result"] is detail.messages[1].result and set(messages[1]) == {"role", "result"}
    assert messages_from_history(ConversationDetail(detail.info, ())) == []
    odd = ConversationDetail(detail.info, (HistoryMessage("assistant", "x", "t", None),))  # cannot happen, but must not crash
    assert messages_from_history(odd) == [{"role": "user", "content": "x"}]
