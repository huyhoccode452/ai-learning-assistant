"""API tests for saved conversations: POST /chat with conversation_id, GET/DELETE /conversations (TestClient, fakes only)."""

import inspect
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import get_settings
from app.main import TITLE_MAX_CHARS, create_app
from app.rag import RagError, RagErrorCode
from tests.fakes import FakeLLM
from tests.test_api import TEXT, build_env, error_of, ready_document

OTHER = "Nội dung của một tài liệu khác. Chỉ mục B-tree phù hợp cho so sánh bằng, nhỏ hơn, lớn hơn và BETWEEN.".encode()


@pytest.fixture
def env(tmp_path: Path) -> Iterator[SimpleNamespace]:
    e = build_env(tmp_path)
    with TestClient(e.app) as client:
        e.client = client
        e.doc = ready_document(e)
        yield e


def chat(env, question: str = "work_mem là gì?", **body):
    return env.client.post("/chat", json={"question": question, **body})


def conversations(env, **params):
    return env.client.get("/conversations", params=params)


def add_failing_trigger(env) -> None:
    with sqlite3.connect(env.db.path) as conn:
        conn.execute("CREATE TRIGGER boom BEFORE INSERT ON messages WHEN NEW.role = 'assistant' BEGIN SELECT RAISE(ABORT, 'boom'); END;")


# ====================================================================== saving through /chat


def test_a_chat_without_a_conversation_id_starts_and_saves_a_conversation(env) -> None:
    response = chat(env)
    assert response.status_code == 200
    body = response.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["conversation_id"]) and body["saved"] is True
    assert body["answer"] == "Trả lời từ tài liệu [1]."  # the answer itself is unchanged by saving
    listing = conversations(env).json()
    assert listing["total"] == 1 and listing["limit"] == 20 and listing["offset"] == 0
    item = listing["items"][0]
    assert (item["id"], item["title"], item["doc_ids"], item["missing_doc_ids"], item["message_count"]) == (
        body["conversation_id"], "work_mem là gì?", [env.doc], [], 2)
    assert item["created_at"] == item["updated_at"]


def test_the_saved_messages_come_back_exactly_with_sources_and_flags(env) -> None:
    live = chat(env).json()
    detail = env.client.get(f"/conversations/{live['conversation_id']}").json()
    user, assistant = detail["messages"]
    assert (user["role"], user["content"], user["sources"], user["latency"], user["used_fallback"]) == ("user", "work_mem là gì?", [], None, None)
    assert (assistant["role"], assistant["content"]) == ("assistant", live["answer"])
    assert assistant["sources"] == [{**s, "document_exists": True} for s in live["sources"]] and live["sources"]
    assert (assistant["used_fallback"], assistant["best_score"], assistant["truncated"]) == (live["used_fallback"], live["best_score"], live["truncated"])
    assert assistant["latency"] == live["latency"]
    assert detail["doc_ids"] == [env.doc] and detail["missing_doc_ids"] == [] and detail["message_count"] == 2


def test_a_follow_up_with_the_conversation_id_is_added_to_the_same_conversation(env) -> None:
    first = chat(env, "câu hỏi đầu tiên").json()["conversation_id"]
    second = chat(env, "câu hỏi thứ hai", conversation_id=first).json()
    assert second["conversation_id"] == first and second["saved"] is True
    detail = env.client.get(f"/conversations/{first}").json()
    assert detail["title"] == "câu hỏi đầu tiên" and detail["message_count"] == 4
    assert [m["content"] for m in detail["messages"] if m["role"] == "user"] == ["câu hỏi đầu tiên", "câu hỏi thứ hai"]
    assert conversations(env).json()["total"] == 1


def test_the_conversation_remembers_every_document_it_searched(env) -> None:
    other = ready_document(env, OTHER, "khac.txt")
    first = chat(env, doc_ids=[env.doc]).json()["conversation_id"]
    chat(env, conversation_id=first, doc_ids=[other])
    assert env.client.get(f"/conversations/{first}").json()["doc_ids"] == [env.doc, other]
    everything = chat(env).json()["conversation_id"]  # no doc_ids = every ready document
    assert set(env.client.get(f"/conversations/{everything}").json()["doc_ids"]) == {env.doc, other}


def test_a_fallback_answer_is_saved_as_a_fallback_with_its_score(env) -> None:
    env.store.score = 0.1  # below the 0.6 threshold: nothing reaches the LLM
    env.store._rebuild()
    live = chat(env, "pg_upgrade là gì?").json()
    assert live["used_fallback"] is True and env.llm.calls == []
    assistant = env.client.get(f"/conversations/{live['conversation_id']}").json()["messages"][1]
    assert (assistant["used_fallback"], assistant["best_score"], assistant["sources"]) == (True, 0.1, [])


def test_the_title_is_the_first_question_on_one_line_and_cut_with_an_ellipsis(env) -> None:
    long_question = "Làm thế nào\n  để   cấu hình " + "rất dài " * 30
    response = chat(env, long_question)
    item = conversations(env).json()["items"][0]
    assert "\n" not in item["title"] and "  " not in item["title"]
    assert len(item["title"]) == TITLE_MAX_CHARS and item["title"].endswith("…") and item["title"].startswith("Làm thế nào để cấu hình")
    assert response.status_code == 200


def test_the_saved_question_is_the_full_text_not_the_shortened_title(env) -> None:
    question = "Làm thế nào để cấu hình bộ nhớ cho PostgreSQL,\nvà vì sao không nên đặt work_mem quá cao trên toàn hệ thống? " * 2
    live = chat(env, question).json()
    saved = env.client.get(f"/conversations/{live['conversation_id']}").json()
    assert saved["messages"][0]["content"] == question.strip() and len(saved["messages"][0]["content"]) > TITLE_MAX_CHARS
    assert len(saved["title"]) <= TITLE_MAX_CHARS  # only the title is cut


def test_one_request_saves_exactly_one_exchange_and_a_repeated_request_is_a_second_one(env) -> None:
    chat(env)
    assert conversations(env).json()["total"] == 1 and conversations(env).json()["items"][0]["message_count"] == 2
    chat(env)  # the same question again, without an id: a new conversation (the API does not guess duplicates)
    assert conversations(env).json()["total"] == 2


# ====================================================================== saving must never lose the answer


def test_when_saving_fails_the_answer_is_still_returned_with_saved_false(env, caplog: pytest.LogCaptureFixture) -> None:
    add_failing_trigger(env)
    with caplog.at_level("ERROR"):
        response = chat(env)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Trả lời từ tài liệu [1]." and body["saved"] is False and body["conversation_id"] is None
    assert body["sources"] and "boom" not in response.text  # the database error text does not reach the client
    assert any("Could not save the chat exchange" in r.getMessage() for r in caplog.records)  # but it is logged
    assert conversations(env).json()["total"] == 0  # and nothing half-written is left


def test_when_a_follow_up_cannot_be_saved_the_id_is_kept_so_the_client_can_continue(env) -> None:
    first = chat(env).json()["conversation_id"]
    add_failing_trigger(env)
    body = chat(env, "câu thất bại khi lưu", conversation_id=first).json()
    assert body["saved"] is False and body["conversation_id"] == first and body["answer"]
    assert env.client.get(f"/conversations/{first}").json()["message_count"] == 2  # unchanged


def test_a_conversation_deleted_while_the_model_was_answering_does_not_lose_the_answer(env) -> None:
    first = chat(env).json()["conversation_id"]

    def delete_during_the_call(system, prompt, **kwargs):
        env.db.delete_conversation(first)
        return "Câu trả lời vẫn đến tay người dùng [1]."

    class Slow(FakeLLM):
        def generate(self, system_instruction, prompt, **kwargs):
            from app.rag import LLMResult

            return LLMResult(text=delete_during_the_call(system_instruction, prompt))

    env.app.dependency_overrides[main.get_llm] = lambda: Slow()
    body = chat(env, "hỏi tiếp", conversation_id=first).json()
    assert body["answer"].startswith("Câu trả lời vẫn đến") and body["saved"] is False


def test_a_failed_llm_call_saves_nothing(env) -> None:
    env.app.dependency_overrides[main.get_llm] = lambda: FakeLLM(RagError(RagErrorCode.RATE_LIMITED, "quota"))
    response = chat(env)
    assert response.status_code == 429 and error_of(response)["code"] == "rag.rate_limited"
    assert conversations(env).json()["total"] == 0
    env.app.dependency_overrides[main.get_llm] = lambda: env.llm
    first = chat(env).json()["conversation_id"]
    env.app.dependency_overrides[main.get_llm] = lambda: FakeLLM(RagError(RagErrorCode.TIMEOUT, "slow"))
    assert chat(env, conversation_id=first).status_code == 504
    assert env.client.get(f"/conversations/{first}").json()["message_count"] == 2  # no question without an answer


# ====================================================================== ids are validated before anything costs money


def test_an_unknown_conversation_id_is_a_404_before_the_model_is_called(env) -> None:
    response = chat(env, conversation_id="f" * 32)
    assert response.status_code == 404 and error_of(response) == {"code": "conversation_not_found", "message": "Không tìm thấy cuộc trò chuyện."}
    assert env.llm.calls == [] and conversations(env).json()["total"] == 0


@pytest.mark.parametrize("bad", ["abc", "F" * 32, "g" * 32, "a" * 31, "a" * 33, "../etc/passwd", "' OR 1=1 --"])
def test_a_malformed_conversation_id_is_a_validation_error(env, bad: str) -> None:
    for response in (chat(env, conversation_id=bad), env.client.get(f"/conversations/{bad}"), env.client.delete(f"/conversations/{bad}")):
        assert response.status_code in (404, 422) and "Traceback" not in response.text
    assert chat(env, conversation_id=bad).status_code == 422 and env.llm.calls == []


def test_an_empty_conversation_id_in_the_chat_body_is_rejected(env) -> None:
    response = chat(env, conversation_id="")
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request" and env.llm.calls == []


# ====================================================================== listing


def test_conversations_are_listed_newest_activity_first(env) -> None:
    ids = [chat(env, f"câu {n}").json()["conversation_id"] for n in range(4)]
    assert [i["id"] for i in conversations(env).json()["items"]] == ids[::-1]
    chat(env, "thêm vào cuộc cũ nhất", conversation_id=ids[0])
    assert conversations(env).json()["items"][0]["id"] == ids[0]


def test_paging_returns_exact_slices_and_the_total(env) -> None:
    ids = [chat(env, f"câu {n}").json()["conversation_id"] for n in range(7)][::-1]
    page = conversations(env, limit=3, offset=0).json()
    assert [i["id"] for i in page["items"]] == ids[:3] and (page["total"], page["limit"], page["offset"]) == (7, 3, 0)
    assert [i["id"] for i in conversations(env, limit=3, offset=3).json()["items"]] == ids[3:6]
    assert [i["id"] for i in conversations(env, limit=3, offset=6).json()["items"]] == ids[6:]
    beyond = conversations(env, limit=3, offset=50).json()
    assert beyond["items"] == [] and beyond["total"] == 7


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"limit": -5}, {"offset": -1}, {"limit": "many"}, {"offset": "x"}])
def test_paging_parameters_out_of_bounds_are_rejected(env, params: dict) -> None:
    response = conversations(env, **params)
    assert response.status_code == 422 and error_of(response)["code"] == "invalid_request"


def test_the_largest_allowed_page_is_accepted(env) -> None:
    chat(env)
    assert conversations(env, limit=100).status_code == 200


def test_an_empty_history_is_an_empty_list_not_an_error(env) -> None:
    assert conversations(env).json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


# ====================================================================== reading, deleting, restart


def test_an_unknown_conversation_is_a_404_for_reading_and_deleting(env) -> None:
    for response in (env.client.get(f"/conversations/{'e' * 32}"), env.client.delete(f"/conversations/{'e' * 32}")):
        assert response.status_code == 404 and error_of(response)["code"] == "conversation_not_found"


def test_deleting_a_conversation_removes_its_messages(env) -> None:
    keep, gone = chat(env, "giữ").json()["conversation_id"], chat(env, "xóa").json()["conversation_id"]
    chat(env, "xóa thêm", conversation_id=gone)
    response = env.client.delete(f"/conversations/{gone}")
    assert response.status_code == 200 and response.json() == {"conversation_id": gone, "deleted_messages": 4}
    assert env.client.get(f"/conversations/{gone}").status_code == 404 and env.client.delete(f"/conversations/{gone}").status_code == 404
    with sqlite3.connect(env.db.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (gone,)).fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (keep,)).fetchone() == (2,)
    assert [i["id"] for i in conversations(env).json()["items"]] == [keep]


def test_reading_and_deleting_history_never_calls_the_model(env) -> None:
    first = chat(env).json()["conversation_id"]
    calls = len(env.llm.calls)
    conversations(env)
    env.client.get(f"/conversations/{first}")
    env.client.delete(f"/conversations/{first}")
    assert len(env.llm.calls) == calls


def test_history_survives_a_restart_of_the_application(env, tmp_path: Path) -> None:
    first = chat(env).json()["conversation_id"]
    restarted = create_app()  # a new application object on the same database file: what a server restart does
    restarted.dependency_overrides[get_settings] = lambda: env.settings
    with TestClient(restarted) as client:
        listing = client.get("/conversations").json()
        assert [i["id"] for i in listing["items"]] == [first]
        assert len(client.get(f"/conversations/{first}").json()["messages"]) == 2


# ====================================================================== a document deleted later


def test_a_conversation_whose_document_was_deleted_still_loads_and_says_so(env) -> None:
    other = ready_document(env, OTHER, "khac.txt")
    first = chat(env, doc_ids=[env.doc]).json()["conversation_id"]
    chat(env, "hỏi tài liệu khác", conversation_id=first, doc_ids=[other])
    assert env.client.delete(f"/documents/{env.doc}").status_code == 200
    detail = env.client.get(f"/conversations/{first}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["doc_ids"] == [env.doc, other] and body["missing_doc_ids"] == [env.doc]
    first_answer, second_answer = [m for m in body["messages"] if m["role"] == "assistant"]
    assert first_answer["sources"] and all(s["document_exists"] is False and s["doc_id"] == env.doc for s in first_answer["sources"])
    assert all(s["document_exists"] is True for s in second_answer["sources"])  # the other document is still there
    assert first_answer["sources"][0]["filename"] == "notes.txt" and first_answer["sources"][0]["excerpt"]  # nothing is hidden
    assert conversations(env).json()["items"][0]["missing_doc_ids"] == [env.doc]


def test_a_conversation_can_continue_after_one_of_its_documents_was_deleted(env) -> None:
    other = ready_document(env, OTHER, "khac.txt")
    first = chat(env, doc_ids=[env.doc]).json()["conversation_id"]
    env.client.delete(f"/documents/{env.doc}")
    follow_up = chat(env, "còn tài liệu kia?", conversation_id=first, doc_ids=[other])
    assert follow_up.status_code == 200 and follow_up.json()["saved"] is True
    assert env.client.get(f"/conversations/{first}").json()["message_count"] == 4


def test_a_conversation_of_a_deleted_document_is_still_listed_and_deletable(env) -> None:
    first = chat(env).json()["conversation_id"]
    env.client.delete(f"/documents/{env.doc}")
    assert conversations(env).json()["items"][0]["missing_doc_ids"] == [env.doc]
    assert env.client.delete(f"/conversations/{first}").status_code == 200


def test_damaged_saved_data_does_not_break_loading_the_conversation(env) -> None:
    first = chat(env).json()["conversation_id"]
    with sqlite3.connect(env.db.path) as conn:
        conn.execute("UPDATE messages SET sources_json = '{broken', latency_json = 'nope' WHERE role = 'assistant'")
    body = env.client.get(f"/conversations/{first}").json()
    assistant = body["messages"][1]
    assert assistant["sources"] == [] and assistant["latency"] is None and assistant["content"]  # that message loses its extras only
    with sqlite3.connect(env.db.path) as conn:
        conn.execute("UPDATE messages SET sources_json = ? WHERE role = 'assistant'", (json.dumps([{"number": "x"}]),))
    assert env.client.get(f"/conversations/{first}").json()["messages"][1]["sources"] == []


# ====================================================================== plumbing


def test_the_history_endpoints_are_plain_def_and_documented(env) -> None:
    routes = [r for r in env.app.routes if getattr(r, "path", "").startswith("/conversations")]
    assert len(routes) == 3 and all(not inspect.iscoroutinefunction(r.endpoint) for r in routes)
    paths = env.client.get("/openapi.json").json()["paths"]
    assert {"get"} == set(paths["/conversations"]) and {"get", "delete"} == set(paths["/conversations/{conversation_id}"])
    chat_schema = env.client.get("/openapi.json").json()["components"]["schemas"]["ChatRequest"]["properties"]
    assert "conversation_id" in chat_schema and "NOT used to understand" in chat_schema["conversation_id"]["description"]


def test_errors_of_the_history_endpoints_leak_nothing(env) -> None:
    add_failing_trigger(env)
    texts = [chat(env).text, conversations(env, limit=0).text, env.client.get(f"/conversations/{'e' * 32}").text]
    assert all("boom" not in t and "sqlite" not in t.lower() and "Traceback" not in t and env.settings.upload_dir.as_posix() not in t for t in texts)
