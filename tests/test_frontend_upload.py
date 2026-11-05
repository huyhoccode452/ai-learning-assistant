"""Upload flow of the Streamlit app. `st.file_uploader` is replaced by a stand-in that, like the real widget,
keeps returning the same file on every rerun until the user removes it. That is exactly the situation that
could cause a re-upload loop."""

import pytest

from frontend.api_client import ApiError, Upload
from tests.test_frontend_app import A, FakeClient, doc, no_exception, start


class FakeUploaded:
    def __init__(self, name: str = "moi.pdf", data: bytes = b"%PDF-1.7 content", file_id: str = "id-1") -> None:
        self.name, self.size, self.file_id, self._data = name, len(data), file_id, data

    def getvalue(self) -> bytes:
        return self._data


def fake_uploader(monkeypatch: pytest.MonkeyPatch, uploaded: "FakeUploaded | None") -> dict:
    seen: dict = {"calls": 0}

    def file_uploader(label, **kwargs):
        seen["calls"] += 1
        seen["kwargs"] = kwargs
        return uploaded

    monkeypatch.setattr("streamlit.file_uploader", file_uploader)
    return seen


def test_an_uploaded_file_is_sent_once_even_though_the_widget_keeps_returning_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    seen = fake_uploader(monkeypatch, FakeUploaded())
    at = start(fake)
    at.run()
    at.run()
    no_exception(at)
    assert fake.uploads == [("moi.pdf", b"%PDF-1.7 content")]  # no re-upload loop
    assert seen["calls"] >= 3 and seen["kwargs"]["type"] == ["pdf", "docx", "txt"] and seen["kwargs"]["max_upload_size"] == 20


def test_after_an_upload_the_user_is_told_it_is_processing_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_uploader(monkeypatch, FakeUploaded())
    at = start(FakeClient())
    assert any("Đã nhận «moi.pdf»" in s.value and "xử lý ở nền" in s.value for s in at.sidebar.success)
    assert at.session_state["uploader_n"] == 1  # the widget was reset by changing its key
    assert at.session_state["first_seen"]  # the polling clock for the new document has started
    assert at.session_state["polling_active"] and "Đang xử lý" in "".join(m.value for m in at.sidebar.markdown)


def test_uploading_a_file_that_is_already_indexed_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient([doc(A, "moi.pdf")])
    fake.upload_value = Upload(doc(A, "moi.pdf"), already_indexed=True)
    fake_uploader(monkeypatch, FakeUploaded())
    at = start(fake)
    assert any("đã có sẵn và đã sẵn sàng" in i.value for i in at.sidebar.info)


@pytest.mark.parametrize(
    "message",
    ["Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .pdf, .docx và .txt.", "Tệp vượt quá dung lượng tối đa cho phép.",
     "PDF này là bản quét (chỉ có ảnh), không có lớp văn bản. Hệ thống chưa hỗ trợ OCR.",
     "Dịch vụ AI đang giới hạn số lượt gọi (gói miễn phí). Hãy thử lại sau ít phút."],
)
def test_upload_errors_show_the_backends_message_and_are_not_retried(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    fake = FakeClient()
    fake.upload_value = ApiError("http", message, status=422)
    fake_uploader(monkeypatch, FakeUploaded("anh.pdf"))
    at = start(fake)
    assert any(message in e.value and "«anh.pdf»" in e.value for e in at.sidebar.error)
    at.run()
    assert len(fake.uploads) == 1  # the same broken file is not sent again on the next rerun


def test_a_different_file_with_the_same_name_is_sent_again(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    chosen = {"file": FakeUploaded("a.pdf", b"one", "id-1")}
    monkeypatch.setattr("streamlit.file_uploader", lambda label, **kw: chosen["file"])
    at = start(fake)
    chosen["file"] = FakeUploaded("a.pdf", b"two!!", "id-2")
    at.run()
    assert [data for _, data in fake.uploads] == [b"one", b"two!!"]


def test_uploads_are_disabled_when_the_backend_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = fake_uploader(monkeypatch, FakeUploaded())
    fake = FakeClient(health=ApiError("unreachable", "Không kết nối được tới máy chủ API."))
    start(fake)
    assert seen["kwargs"]["disabled"] is True and fake.uploads == []
