"""Tests for the saved chat history in app/db.py (real SQLite in a temporary folder)."""

import json
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from app.db import ConversationNotFound, Database, Role

A, B, C = "a" * 16, "b" * 16, "c" * 16
SOURCES = json.dumps([{"number": 1, "doc_id": A, "filename": "a.pdf", "page": 5, "chunk_index": 3, "score": 0.8, "excerpt": "x", "cited": True}])
LATENCY = json.dumps({"embed_ms": 100.0, "search_ms": 10.0, "llm_ms": 1500.0, "total_ms": 1610.0})


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "app.sqlite3")
    database.init_db()
    return database


def exchange(db: Database, conversation_id: str | None = None, question: str = "work_mem là gì?", docs=(A,), **changes) -> str:
    values = dict(conversation_id=conversation_id, title=question[:60], doc_ids=list(docs), question=question, answer="Là bộ nhớ [1].",
                  sources_json=SOURCES, used_fallback=False, best_score=0.83, latency_json=LATENCY, truncated=False)
    values.update(changes)
    return db.save_exchange(**values)


def raw(db: Database, sql: str, *params):
    with sqlite3.connect(db.path) as conn:
        return conn.execute(sql, params).fetchall()


def add_trigger(db: Database, role: str) -> None:
    with sqlite3.connect(db.path) as conn:
        conn.execute(f"CREATE TRIGGER boom BEFORE INSERT ON messages WHEN NEW.role = '{role}' BEGIN SELECT RAISE(ABORT, 'boom'); END;")


# ---------------------------------------------------------------------- saving one exchange


def test_the_first_exchange_creates_a_conversation_with_two_messages(db: Database) -> None:
    conversation_id = exchange(db, question="work_mem là gì?", docs=(A, B, A))
    assert re.fullmatch(r"[0-9a-f]{32}", conversation_id)
    conversation = db.get_conversation(conversation_id)
    assert conversation is not None
    assert (conversation.title, conversation.doc_ids, conversation.message_count) == ("work_mem là gì?", (A, B), 2)  # duplicates collapsed
    assert conversation.created_at == conversation.updated_at and conversation.created_at.endswith("+00:00")
    user, assistant = db.list_messages(conversation_id)
    assert (user.role, user.content) == (Role.USER, "work_mem là gì?")
    assert (user.sources_json, user.used_fallback, user.best_score, user.latency_json, user.truncated) == (None,) * 5
    assert (assistant.role, assistant.content) == (Role.ASSISTANT, "Là bộ nhớ [1].")
    assert (assistant.sources_json, assistant.used_fallback, assistant.best_score, assistant.truncated) == (SOURCES, False, 0.83, False)
    assert json.loads(assistant.latency_json)["llm_ms"] == 1500.0  # type: ignore[arg-type]
    assert user.id < assistant.id


def test_a_fallback_answer_and_a_truncated_one_keep_their_flags(db: Database) -> None:
    fallback = exchange(db, used_fallback=True, best_score=None, answer="Không đủ bằng chứng.", sources_json="[]")
    cut = exchange(db, truncated=True)
    assistant = db.list_messages(fallback)[1]
    assert (assistant.used_fallback, assistant.best_score, assistant.truncated, assistant.sources_json) == (True, None, False, "[]")
    assert db.list_messages(cut)[1].truncated is True


def test_a_follow_up_exchange_goes_to_the_same_conversation_and_moves_it_up(db: Database) -> None:
    first = exchange(db, question="câu một", docs=(A,))
    before = db.get_conversation(first)
    exchange(db, question="chuyện khác", docs=(C,))
    again = exchange(db, first, question="câu hai", docs=(B, A))
    after = db.get_conversation(first)
    assert again == first and after is not None and before is not None
    assert (after.title, after.message_count) == ("câu một", 4)  # the title stays the first question
    assert after.doc_ids == (A, B) and after.updated_at > before.updated_at and after.created_at == before.created_at
    assert [m.content for m in db.list_messages(first)][::2] == ["câu một", "câu hai"]
    assert [c.title for c in db.list_conversations()] == ["câu một", "chuyện khác"]  # newest activity first


def test_an_unknown_conversation_is_refused_and_nothing_is_written(db: Database) -> None:
    with pytest.raises(ConversationNotFound):
        exchange(db, "f" * 32)
    assert raw(db, "SELECT COUNT(*) FROM conversations") == [(0,)] and raw(db, "SELECT COUNT(*) FROM messages") == [(0,)]


# ---------------------------------------------------------------------- all or nothing


@pytest.mark.parametrize("failing_role", ["assistant", "user"])
def test_a_failure_in_the_middle_of_a_new_exchange_leaves_nothing_behind(db: Database, failing_role: str) -> None:
    add_trigger(db, failing_role)
    with pytest.raises(sqlite3.DatabaseError):
        exchange(db)
    # no conversation without messages, no question without its answer
    assert raw(db, "SELECT COUNT(*) FROM conversations") == [(0,)] and raw(db, "SELECT COUNT(*) FROM messages") == [(0,)]


def test_a_failure_in_a_follow_up_leaves_the_conversation_exactly_as_it_was(db: Database) -> None:
    conversation_id = exchange(db, docs=(A,))
    before = (db.get_conversation(conversation_id), db.list_messages(conversation_id))
    add_trigger(db, "assistant")
    with pytest.raises(sqlite3.DatabaseError):
        exchange(db, conversation_id, question="câu thất bại", docs=(B,))
    assert (db.get_conversation(conversation_id), db.list_messages(conversation_id)) == before  # no 3rd message, no new doc, no bump


def test_after_a_failed_save_the_database_still_works(db: Database) -> None:
    add_trigger(db, "assistant")
    with pytest.raises(sqlite3.DatabaseError):
        exchange(db)
    raw(db, "DROP TRIGGER boom")
    assert db.get_conversation(exchange(db)) is not None


def test_many_simultaneous_exchanges_into_one_conversation_are_all_saved(db: Database) -> None:
    conversation_id = exchange(db, docs=(A,))
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        barrier.wait()
        try:
            exchange(db, conversation_id, question=f"câu {n}", docs=(f"{n:016x}",))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conversation = db.get_conversation(conversation_id)
    assert not errors and conversation is not None and conversation.message_count == 18
    assert len(conversation.doc_ids) == 9 and A in conversation.doc_ids  # no doc id lost to a race on the merge


# ---------------------------------------------------------------------- listing, ordering, paging


def test_conversations_are_listed_most_recent_first_even_within_one_second(db: Database) -> None:
    ids = [exchange(db, question=f"câu {n}") for n in range(6)]
    assert [c.id for c in db.list_conversations()] == ids[::-1]
    exchange(db, ids[2], question="thêm vào cũ")
    assert [c.id for c in db.list_conversations()][0] == ids[2]


def test_paging_is_exact_and_bounded(db: Database) -> None:
    ids = [exchange(db, question=f"câu {n}") for n in range(25)][::-1]  # newest first
    assert db.count_conversations() == 25
    assert [c.id for c in db.list_conversations(limit=10)] == ids[:10]
    assert [c.id for c in db.list_conversations(limit=10, offset=10)] == ids[10:20]
    assert [c.id for c in db.list_conversations(limit=10, offset=20)] == ids[20:]  # a short last page
    assert db.list_conversations(limit=10, offset=25) == [] and db.list_conversations(limit=0) == []
    assert [c.id for c in db.list_conversations()] == ids  # no limit = everything


def test_the_list_carries_the_message_count(db: Database) -> None:
    one, two = exchange(db), exchange(db)
    exchange(db, two)
    counts = {c.id: c.message_count for c in db.list_conversations()}
    assert counts == {one: 2, two: 4}


# ---------------------------------------------------------------------- deleting


def test_deleting_a_conversation_removes_its_messages_and_only_its_own(db: Database) -> None:
    keep, gone = exchange(db, question="giữ lại"), exchange(db, question="xóa")
    exchange(db, gone)
    assert db.delete_conversation(gone) == 4
    assert db.get_conversation(gone) is None and db.list_messages(gone) == []
    assert raw(db, "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", gone) == [(0,)]  # cascade, no orphans
    assert db.get_conversation(keep) is not None and len(db.list_messages(keep)) == 2


def test_deleting_twice_or_deleting_an_unknown_id_is_none(db: Database) -> None:
    conversation_id = exchange(db)
    assert db.delete_conversation(conversation_id) == 2
    assert db.delete_conversation(conversation_id) is None and db.delete_conversation("f" * 32) is None


def test_the_cascade_works_without_going_through_delete_conversation(db: Database) -> None:
    conversation_id = exchange(db)
    with db._connect() as conn:  # the same connection settings every method uses (foreign keys ON)
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    assert raw(db, "SELECT COUNT(*) FROM messages") == [(0,)]


def test_deleting_a_document_does_not_delete_the_history_that_mentions_it(db: Database) -> None:
    db.claim_document(A, "a.pdf", 3, 6)
    db.mark_ready(A, 6)
    conversation_id = exchange(db, docs=(A,))
    assert db.delete_document(A) is True
    conversation = db.get_conversation(conversation_id)
    assert conversation is not None and conversation.doc_ids == (A,) and len(db.list_messages(conversation_id)) == 2


# ---------------------------------------------------------------------- safety and compatibility


def test_values_and_ids_are_never_interpreted_as_sql(db: Database) -> None:
    hostile = "x'); DROP TABLE messages; --"
    conversation_id = exchange(db, question=hostile, answer=hostile)
    assert db.get_conversation(conversation_id).title == hostile  # type: ignore[union-attr]
    assert db.list_messages(conversation_id)[0].content == hostile
    assert db.get_conversation("' OR '1'='1") is None and db.delete_conversation("' OR '1'='1") is None
    assert db.count_conversations() == 1 and len(db.list_messages(conversation_id)) == 2  # still there


def test_add_message_keeps_working_and_bumps_the_activity_time(db: Database) -> None:
    conversation = db.create_conversation("thủ công", doc_ids=[A, A, B])
    assert conversation.doc_ids == (A, B)
    message = db.add_message(conversation.id, Role.ASSISTANT, "trả lời", "[]", used_fallback=True, best_score=0.5, latency_json="{}", truncated=False)
    assert (message.used_fallback, message.best_score, message.truncated) == (True, 0.5, False)
    assert db.get_conversation(conversation.id).updated_at == message.created_at  # type: ignore[union-attr]
    with pytest.raises(sqlite3.IntegrityError):
        db.add_message("f" * 32, Role.USER, "mồ côi")


def test_a_damaged_doc_ids_value_does_not_break_reading(db: Database) -> None:
    conversation_id = exchange(db)
    raw(db, "UPDATE conversations SET doc_ids_json = ? WHERE id = ?", "{not json", conversation_id)
    assert db.get_conversation(conversation_id).doc_ids == ()  # type: ignore[union-attr]
    raw(db, "UPDATE conversations SET doc_ids_json = ? WHERE id = ?", '[1, "ok", null]', conversation_id)
    assert db.get_conversation(conversation_id).doc_ids == ("ok",)  # type: ignore[union-attr]
    exchange(db, conversation_id, docs=(B,))  # saving into it still works and repairs the list
    assert db.get_conversation(conversation_id).doc_ids == ("ok", B)  # type: ignore[union-attr]


def test_a_database_created_before_history_existed_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:  # the schema of the first version: no doc ids, no updated_at, no answer fields
        conn.executescript("""
            CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, created_at TEXT NOT NULL);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), content TEXT NOT NULL, sources_json TEXT, created_at TEXT NOT NULL);
            INSERT INTO conversations VALUES ('old', 'Cuộc trò chuyện cũ', '2026-01-01T00:00:00+00:00');
            INSERT INTO messages (conversation_id, role, content, sources_json, created_at) VALUES ('old', 'user', 'câu hỏi cũ', NULL, '2026-01-01T00:00:00+00:00');
            INSERT INTO conversations VALUES ('older', 'Cũ hơn', '2025-06-01T00:00:00+00:00');  -- earlier date, LATER row
        """)
    database = Database(path)
    database.init_db()
    database.init_db()  # and again: idempotent
    old = database.get_conversation("old")
    assert old is not None and (old.title, old.doc_ids, old.updated_at, old.message_count) == ("Cuộc trò chuyện cũ", (), "2026-01-01T00:00:00+00:00", 1)
    message = database.list_messages("old")[0]
    assert (message.content, message.used_fallback, message.best_score, message.latency_json, message.truncated) == ("câu hỏi cũ", None, None, None, None)
    new_id = exchange(database, docs=(A,))  # new exchanges work on the upgraded file, next to the old data
    assert [c.id for c in database.list_conversations()] == [new_id, "old", "older"]  # old rows ordered by their creation date
    assert exchange(database, "old", docs=(B,)) == "old" and database.get_conversation("old").doc_ids == (B,)  # type: ignore[union-attr]


def test_exchanges_in_the_same_clock_tick_still_sort_by_activity(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: on Windows `datetime.now()` can repeat for calls milliseconds apart, so `updated_at` tied."""
    import app.db as db_module
    from datetime import UTC, datetime

    frozen = datetime(2026, 11, 5, 8, 0, 0, tzinfo=UTC)

    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return frozen

    monkeypatch.setattr(db_module, "datetime", FrozenClock)
    monkeypatch.setattr(db_module, "_last_precise", None)  # restored afterwards, so other tests are unaffected
    first = exchange(db, question="câu một", docs=(A,))
    exchange(db, question="chuyện khác", docs=(C,))
    exchange(db, first, question="câu hai", docs=(A,))
    assert [c.title for c in db.list_conversations()] == ["câu một", "chuyện khác"]
