"""UI tests with streamlit.testing.v1.AppTest and a fake API client: no browser, no server, no network.

The app reads its client from st.session_state["client"], so each test puts a FakeClient there before the
first run and then checks what the page shows and what the client was asked to do."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import ApiError, ChatResult, Document, Health, Latency, Source, Upload

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py")
A, B, C = "a" * 16, "b" * 16, "c" * 16


def doc(doc_id: str = A, name: str = "bao_cao.pdf", status: str = "ready", **fields) -> Document:
    values = dict(error_code=None, error_message=None, num_pages=9, num_chunks=29)
    values.update(fields)
    return Document(doc_id, name, status, **values)


def source(number: int, cited: bool, page: int | None = 5, name: str = "bao_cao.pdf", excerpt: str = "work_mem mặc định 4MB", score: float = 0.83):
    return Source(number, name, page, score, excerpt, cited)


LATENCY = Latency(embed_ms=120.0, search_ms=15.0, llm_ms=1800.0, total_ms=1950.0)


def answer(**fields) -> ChatResult:
    values = dict(answer="work_mem mặc định là 4MB [1].", sources=(source(1, True), source(2, False, page=None, name="ghi_chu.txt")),
                  used_fallback=False, best_score=0.83, truncated=False, latency=LATENCY)
    values.update(fields)
    return ChatResult(**values)


FALLBACK = answer(answer="Tôi không tìm thấy thông tin đủ liên quan.", sources=(), used_fallback=True, best_score=0.5427,
                  latency=Latency(100.0, 10.0, 0.0, 300.0))


class FakeClient:
    def __init__(self, docs=(), *, health: "Health | ApiError" = Health("ok", True), reply: "ChatResult | ApiError" = None) -> None:
        self.docs = list(docs)
        self.health_value = health
        self.reply = reply or answer()
        self.chat_calls: list[tuple[str, list[str] | None]] = []
        self.deleted: list[str] = []
        self.delete_error: ApiError | None = None
        self.uploads: list[tuple[str, bytes]] = []
        self.list_calls = 0
        self.pending_sent_during_call: list[bool] = []
        self.upload_value: "Upload | ApiError" = Upload(doc(C, "moi.pdf", "processing"), already_indexed=False)

    def health(self) -> Health:
        if isinstance(self.health_value, ApiError):
            raise self.health_value
        return self.health_value

    def list_documents(self) -> list[Document]:
        self.list_calls += 1
        return list(self.docs)

    def chat(self, question: str, doc_ids: list[str] | None = None) -> ChatResult:
        import streamlit as st  # we are inside the script run here, so session_state is the real one

        self.pending_sent_during_call.append(st.session_state["pending"]["sent"])
        self.chat_calls.append((question, doc_ids))
        if isinstance(self.reply, ApiError):
            raise self.reply
        return self.reply

    def delete_document(self, doc_id: str) -> int:
        if self.delete_error:
            raise self.delete_error
        self.deleted.append(doc_id)
        self.docs = [d for d in self.docs if d.doc_id != doc_id]
        return 29

    def upload(self, filename: str, content: bytes) -> Upload:
        self.uploads.append((filename, content))
        if isinstance(self.upload_value, ApiError):
            raise self.upload_value
        if not self.upload_value.already_indexed:  # like the real server: the new document shows up in the list
            self.docs.append(self.upload_value.document)
        return self.upload_value


def start(client: FakeClient, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=20)
    at.session_state["client"] = client
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def texts(at: AppTest) -> str:
    """All visible text of the page, flattened, for substring checks."""
    parts = [e.value for group in (at.markdown, at.caption, at.info, at.warning, at.error, at.success, at.title, at.header) for e in group]
    return "\n".join(parts)


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


def no_exception(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


# ---------------------------------------------------------------------- empty state


def test_empty_state_disables_the_chat_and_explains_what_to_do() -> None:
    at = start(FakeClient())
    no_exception(at)
    assert at.chat_input[0].disabled
    assert "Hãy tải lên một tài liệu" in texts(at) and "Chưa có tài liệu nào." in texts(at)
    assert at.multiselect[0].disabled and "Cách hoạt động" in texts(at) and "không có đăng nhập" in texts(at)


def test_only_processing_or_failed_documents_still_leave_the_chat_disabled() -> None:
    at = start(FakeClient([doc(A, "a.pdf", "processing"), doc(B, "b.pdf", "failed", error_message="Tệp bị hỏng.")]))
    no_exception(at)
    assert at.chat_input[0].disabled and at.multiselect[0].options == []
    assert "Tệp bị hỏng." in texts(at) and "Đang xử lý" in texts(at) and "Lỗi" in texts(at)


# ---------------------------------------------------------------------- backend problems


def test_backend_down_shows_a_friendly_message_and_disables_everything() -> None:
    down = ApiError("unreachable", "Không kết nối được tới máy chủ API tại http://127.0.0.1:8000. Hãy kiểm tra máy chủ đang chạy.")
    at = start(FakeClient([doc()], health=down))
    no_exception(at)
    assert any("Không kết nối được tới máy chủ API" in e.value for e in at.error)
    assert at.chat_input[0].disabled and at.sidebar.button[0].label != "" and at.get("file_uploader")[0].proto.disabled
    assert "Traceback" not in texts(at)
    assert not at.info  # one clear message, not the banner plus a second box saying the same thing


def test_degraded_backend_warns_about_qdrant_but_chat_stays_available() -> None:
    at = start(FakeClient([doc()], health=Health("degraded", False)))
    no_exception(at)
    assert any("Qdrant" in w.value for w in at.warning) and not at.chat_input[0].disabled


def test_an_invalid_base_url_is_explained_not_a_stack_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "not-a-url")
    at = AppTest.from_file(APP, default_timeout=20).run()
    no_exception(at)
    assert any("API_BASE_URL" in e.value for e in at.error)


# ---------------------------------------------------------------------- documents and scope


def test_all_ready_documents_are_selected_by_default_and_others_are_not_offered() -> None:
    at = start(FakeClient([doc(A, "a.pdf"), doc(B, "b.docx"), doc(C, "c.txt", "processing")]))
    no_exception(at)
    assert at.multiselect[0].options == ["a.pdf", "b.docx"] and at.multiselect[0].value == [A, B]
    assert not at.chat_input[0].disabled


def test_a_document_that_becomes_ready_is_added_to_the_selection_but_a_deselected_one_stays_deselected() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    at = start(fake)
    at.multiselect[0].unselect(B).run()
    assert at.multiselect[0].value == [A]
    fake.docs.append(doc(C, "c.pdf"))
    at.run()
    assert at.multiselect[0].value == [A, C]


def test_chat_searches_only_the_selected_documents() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    at = start(fake)
    at.multiselect[0].unselect(A).run()
    ask(at, "work_mem là gì?")
    assert fake.chat_calls == [("work_mem là gì?", [B])]


def test_deselecting_everything_disables_the_chat() -> None:
    at = start(FakeClient([doc(A)]))
    at.multiselect[0].unselect(A).run()
    assert at.chat_input[0].disabled and "chọn ít nhất một tài liệu" in texts(at)


def test_a_full_run_fetches_the_document_list_once_not_twice() -> None:
    fake = FakeClient([doc()])
    start(fake)
    assert fake.list_calls == 1


def test_the_notice_after_a_delete_stays_visible() -> None:
    at = start(FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")]))
    sidebar_button(at, f"del_{A}").click().run()
    sidebar_button(at, f"yes_{A}").click().run()
    assert any("Đã xóa «a.pdf»" in s.value for s in at.sidebar.success)  # it used to vanish in an extra rerun


# ---------------------------------------------------------------------- answers


def test_an_answer_shows_text_sources_and_latency() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "work_mem mặc định là bao nhiêu?")
    no_exception(at)
    assert fake.chat_calls == [("work_mem mặc định là bao nhiêu?", [A])]
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    page = texts(at)
    assert "work_mem mặc định là 4MB [1]." in page
    assert [e.label for e in at.expander] == ["Nguồn (2)"]
    assert "Được trích dẫn" in page and "Chỉ truy xuất" in page  # cited vs only retrieved
    assert "trang 5" in page and "không có số trang" in page and "0.830" in page and "work\\_mem mặc định 4MB" in page  # the excerpt is escaped, so it shows literally
    assert "tổng 1.9 s" in page and "mô hình 1.8 s" in page and "nhúng câu hỏi 120 ms" in page
    assert not at.info or all("Không đủ bằng chứng" not in i.value for i in at.info)


def test_a_fallback_answer_is_visibly_different_and_shows_the_score() -> None:
    at = ask(start(FakeClient([doc()], reply=FALLBACK)), "pg_upgrade là gì?")
    no_exception(at)
    assert any("Không đủ bằng chứng trong tài liệu" in i.value and "Tôi không tìm thấy" in i.value for i in at.info)
    page = texts(at)
    assert "0.543" in page and "không gọi mô hình" in page and "tổng 0.3 s" in page
    assert not at.expander  # nothing was retrieved for the model, so there is no source list


def test_a_fallback_without_any_hit_says_so() -> None:
    at = ask(start(FakeClient([doc()], reply=answer(used_fallback=True, best_score=None, sources=(), answer="Không có."))), "q")
    assert "Không tìm thấy đoạn nào" in texts(at)


def test_sources_none_of_which_is_cited_are_labelled_as_retrieved_only() -> None:
    reply = answer(answer="Tài liệu không đề cập pg_upgrade.", sources=(source(1, False), source(2, False)))
    page = texts(ask(start(FakeClient([doc()], reply=reply)), "pg_upgrade?"))
    assert "không trích dẫn nguồn nào" in page and "Được trích dẫn" not in page.replace("Chỉ truy xuất", "")


def test_a_truncated_answer_is_flagged() -> None:
    at = ask(start(FakeClient([doc()], reply=answer(truncated=True))), "q")
    assert any("bị cắt" in w.value for w in at.warning)


def test_document_text_is_shown_literally_not_interpreted() -> None:
    reply = answer(answer="Chi phí là $5 và $6.", sources=(source(1, True, name="[x](http://evil.test).pdf", excerpt="**to** $a$ [link](http://evil.test) # tiêu đề"),))
    page = texts(ask(start(FakeClient([doc()], reply=reply)), "q"))
    assert "\\[x\\]\\(http" not in page  # '(' is not special; only the '[' and ']' are escaped
    assert "\\[x\\](http\\://evil.test).pdf" in page  # no link: '[', ']' and the ':' that could start a directive are escaped
    assert "\\*\\*to\\*\\* \\$a\\$ \\[link\\](http\\://evil.test) \\# tiêu đề" in page
    assert "Chi phí là \\$5 và \\$6." in page  # no accidental LaTeX in the model's answer


# ---------------------------------------------------------------------- errors while chatting


@pytest.mark.parametrize(
    ("error", "level"),
    [
        (ApiError("http", "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau.", status=429), "error"),
        (ApiError("http", "Không kết nối được cơ sở dữ liệu vector (Qdrant). Hãy kiểm tra Qdrant đang chạy.", status=503), "error"),
        (ApiError("http", "Tài liệu chưa sẵn sàng để hỏi.", status=409), "error"),
        (ApiError("http", "Yêu cầu không hợp lệ. Hãy kiểm tra lại các trường được nêu.", status=422), "error"),
        (ApiError("unreachable", "Không kết nối được tới máy chủ API tại http://127.0.0.1:8000."), "error"),
        (ApiError("timeout", "Máy chủ chưa trả lời sau 60 giây. Yêu cầu có thể vẫn đang được xử lý ở phía máy chủ."), "warning"),
    ],
)
def test_chat_errors_show_the_mapped_message_in_the_conversation(error: ApiError, level: str) -> None:
    at = ask(start(FakeClient([doc()], reply=error)), "câu hỏi")
    no_exception(at)
    shown = [e.value for e in getattr(at, level)]
    assert error.message in shown
    assert "Traceback" not in texts(at) and [m.name for m in at.chat_message] == ["user", "assistant"]


def test_a_timeout_message_does_not_claim_the_server_failed() -> None:
    timeout = ApiError("timeout", "Máy chủ chưa trả lời sau 60 giây. Yêu cầu có thể vẫn đang được xử lý ở phía máy chủ, nên chưa chắc nó đã thất bại.")
    at = ask(start(FakeClient([doc()], reply=timeout)), "q")
    assert any("chưa chắc nó đã thất bại" in w.value for w in at.warning) and not at.error


# ---------------------------------------------------------------------- reruns, duplicates, clearing


def test_a_rerun_does_not_ask_again_and_the_history_is_kept() -> None:
    fake = FakeClient([doc()])
    at = ask(start(fake), "một câu hỏi")
    at.run()
    at.run()
    assert len(fake.chat_calls) == 1
    assert [m.name for m in at.chat_message] == ["user", "assistant"] and "work_mem mặc định là 4MB" in texts(at)


def test_two_questions_make_two_exchanges_in_order() -> None:
    fake = FakeClient([doc()])
    at = ask(ask(start(fake), "câu một"), "câu hai")
    assert [q for q, _ in fake.chat_calls] == ["câu một", "câu hai"]
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]


def test_a_question_cut_off_mid_request_is_not_sent_again() -> None:
    fake = FakeClient([doc()])
    state = {"messages": [{"role": "user", "content": "đang hỏi"}], "pending": {"question": "đang hỏi", "doc_ids": [A], "sent": True}}
    at = start(fake, **state)
    no_exception(at)
    assert fake.chat_calls == [] and any("bị gián đoạn" in w.value for w in at.warning)
    at.run()
    assert fake.chat_calls == [] and sum("bị gián đoạn" in w.value for w in at.warning) == 1  # reported once


def test_a_question_that_arrives_after_the_last_document_disappeared_is_not_sent() -> None:
    """The input was enabled in the browser, but by the time the script ran the only document was gone."""
    fake = FakeClient([doc()])
    at = start(fake)
    at.chat_input[0].set_value("câu hỏi muộn")
    fake.docs.clear()
    at.run()
    no_exception(at)
    assert fake.chat_calls == [] and not at.chat_message and at.chat_input[0].disabled


def test_the_request_is_marked_as_sent_before_the_call_is_made() -> None:
    """If the run is cut off during the call, the next run must see 'sent' and not ask again."""
    fake = FakeClient([doc()])
    ask(start(fake), "câu hỏi")
    assert fake.pending_sent_during_call == [True]


def test_file_names_in_the_document_list_are_shown_literally() -> None:
    at = start(FakeClient([doc(A, "[x](http://evil.test) **b**.pdf")]))
    sidebar_text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "\\[x\\](http\\://evil.test) \\*\\*b\\*\\*.pdf" in sidebar_text


def test_clear_chat_removes_the_conversation() -> None:
    at = ask(start(FakeClient([doc()])), "q")
    assert at.chat_message
    next(b for b in at.sidebar.button if b.label == "Xóa hội thoại").click().run()
    assert not at.chat_message and "Hãy đặt câu hỏi ở ô bên dưới" in texts(at)


# ---------------------------------------------------------------------- deleting


def sidebar_button(at: AppTest, key: str):
    return next(b for b in at.sidebar.button if b.key == key)


def test_delete_asks_for_confirmation_before_doing_anything() -> None:
    fake = FakeClient([doc(A, "a.pdf"), doc(B, "b.pdf")])
    at = start(fake)
    sidebar_button(at, f"del_{A}").click().run()
    assert fake.deleted == [] and any("Xóa tài liệu này" in w.value for w in at.sidebar.warning)
    sidebar_button(at, f"no_{A}").click().run()  # cancel
    assert fake.deleted == [] and not any("Xóa tài liệu này" in w.value for w in at.sidebar.warning)
    sidebar_button(at, f"del_{A}").click().run()
    sidebar_button(at, f"yes_{A}").click().run()
    no_exception(at)
    assert fake.deleted == [A] and "Đã xóa «a.pdf» (29 đoạn)." in texts(at)
    assert at.multiselect[0].options == ["b.pdf"] and at.multiselect[0].value == [B]


def test_a_delete_error_is_shown_with_the_backends_message() -> None:
    fake = FakeClient([doc()])
    fake.delete_error = ApiError("http", "Tài liệu đang được xử lý; hãy đợi xử lý xong rồi xóa.", status=409)
    at = start(fake)
    sidebar_button(at, f"del_{A}").click().run()
    sidebar_button(at, f"yes_{A}").click().run()
    assert any("Tài liệu đang được xử lý; hãy đợi xử lý xong rồi xóa." in e.value for e in at.error)
    assert fake.deleted == [] and at.multiselect[0].options == ["bao_cao.pdf"]


def test_a_processing_document_cannot_be_deleted() -> None:
    at = start(FakeClient([doc(A, "a.pdf", "processing")]))
    assert sidebar_button(at, f"del_{A}").disabled


# ---------------------------------------------------------------------- polling hints are bounded


def test_a_slow_document_gets_the_free_tier_hint() -> None:
    at = start(FakeClient([doc(A, "a.pdf", "processing")]), first_seen={A: time.monotonic() - 60})
    no_exception(at)
    assert any("gói miễn phí chậm" in w.value for w in at.sidebar.warning)


def test_polling_stops_after_the_limit_and_offers_a_manual_check() -> None:
    fake = FakeClient([doc(A, "a.pdf", "processing")])
    at = start(fake, first_seen={A: time.monotonic() - 400})
    no_exception(at)
    assert any("Đã dừng tự động kiểm tra" in w.value for w in at.sidebar.warning)
    sidebar_button(at, "recheck").click().run()
    assert not any("Đã dừng tự động kiểm tra" in w.value for w in at.sidebar.warning)  # the clock restarted


def test_a_fresh_document_is_not_flagged_as_slow() -> None:
    at = start(FakeClient([doc(A, "a.pdf", "processing")]))
    assert not at.sidebar.warning and "danh sách tự cập nhật" in texts(at)


# ---------------------------------------------------------------------- hygiene


def test_the_frontend_contains_no_secret_handling() -> None:
    root = Path(APP).parent
    source_text = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.py"))
    for forbidden in ("GEMINI", "api_key", "API_KEY", "dotenv", "Settings(", "read_text(\".env", "app.config"):
        assert forbidden not in source_text, forbidden
