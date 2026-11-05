"""Tests for app/db.py (real SQLite in a temporary folder)."""

import sqlite3
import threading
from pathlib import Path

import pytest

from app.db import Database, DocStatus, Role


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "sub" / "app.sqlite3")
    database.init_db()
    return database


def claim(db: Database, doc_id: str = "a" * 16, filename: str = "a.txt", chunks: int = 3):
    return db.claim_document(doc_id, filename, None, chunks)


def test_init_creates_folder_tables_and_is_repeatable(tmp_path: Path) -> None:
    database = Database(tmp_path / "new" / "deep" / "app.sqlite3")
    database.init_db()
    database.init_db()  # second call must not fail or wipe anything
    with sqlite3.connect(database.path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"documents", "conversations", "messages"} <= names
    assert mode == "wal"


def test_claim_new_document_is_processing(db: Database) -> None:
    document, claimed = claim(db)
    assert claimed is True
    assert (document.status, document.filename, document.num_chunks) == (DocStatus.PROCESSING, "a.txt", 3)
    assert document.created_at.endswith("+00:00")
    assert db.get_document("a" * 16) == document


def test_claim_is_refused_while_processing_or_ready(db: Database) -> None:
    claim(db)
    _, claimed = claim(db)
    assert claimed is False  # someone is already indexing it
    assert db.mark_ready("a" * 16, 3)
    document, claimed = claim(db)
    assert claimed is False and document.status == DocStatus.READY


def test_claim_of_a_failed_document_retries_and_clears_the_error(db: Database) -> None:
    claim(db)
    db.mark_failed("a" * 16, "embedding.rate_limited", "hết quota")
    document, claimed = claim(db, chunks=7)
    assert claimed is True
    assert (document.status, document.error_code, document.error_message, document.num_chunks) == (
        DocStatus.PROCESSING, None, None, 7)


def test_only_one_of_many_simultaneous_claims_wins(db: Database) -> None:
    results: list[bool] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        results.append(claim(db)[1])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and len(results) == 8


def test_status_transitions_are_one_way_from_processing(db: Database) -> None:
    claim(db)
    assert db.mark_ready("a" * 16, 5) is True
    assert db.mark_failed("a" * 16, "x", "y") is False  # ready is not overwritten
    assert db.mark_ready("a" * 16, 9) is False
    document = db.get_document("a" * 16)
    assert document and document.status == DocStatus.READY and document.num_chunks == 5
    assert db.mark_ready("f" * 16, 1) is False  # unknown id


def test_mark_failed_stores_code_and_message(db: Database) -> None:
    claim(db)
    assert db.mark_failed("a" * 16, "embedding.rate_limited", "Hết quota")
    document = db.get_document("a" * 16)
    assert document and (document.status, document.error_code, document.error_message) == (
        DocStatus.FAILED, "embedding.rate_limited", "Hết quota")


def test_fail_stuck_processing_only_touches_processing_rows(db: Database) -> None:
    claim(db, "a" * 16)
    claim(db, "b" * 16)
    db.mark_ready("b" * 16, 1)
    claim(db, "c" * 16)
    assert db.fail_stuck_processing("interrupted", "restart") == 2
    statuses = {d.doc_id[0]: d.status for d in db.list_documents()}
    assert statuses == {"a": DocStatus.FAILED, "b": DocStatus.READY, "c": DocStatus.FAILED}


def test_delete_refuses_a_processing_document(db: Database) -> None:
    claim(db)
    assert db.delete_document("a" * 16) is False
    assert db.get_document("a" * 16) is not None
    db.mark_failed("a" * 16, "x", "y")
    assert db.delete_document("a" * 16) is True
    assert db.get_document("a" * 16) is None
    assert db.delete_document("a" * 16) is False


def test_list_ready_and_bulk_lookup(db: Database) -> None:
    for letter in "abc":
        claim(db, letter * 16)
    db.mark_ready("a" * 16, 1)
    db.mark_ready("c" * 16, 1)
    assert sorted(db.ready_doc_ids()) == ["a" * 16, "c" * 16]
    assert len(db.list_documents()) == 3
    found = db.get_documents(["a" * 16, "z" * 16])
    assert list(found) == ["a" * 16]
    assert db.get_documents([]) == {}


def test_values_are_never_interpreted_as_sql(db: Database) -> None:
    hostile = "x'); DROP TABLE documents; --"
    db.claim_document("a" * 16, hostile, None, 1)
    assert db.get_document("a" * 16).filename == hostile  # type: ignore[union-attr]
    assert db.get_documents([hostile, "a" * 16]).keys() == {"a" * 16}
    assert len(db.list_documents()) == 1  # the table still exists


def test_conversations_and_messages(db: Database) -> None:
    first = db.create_conversation("Hỏi về PostgreSQL")
    second = db.create_conversation()
    assert {c.id for c in db.list_conversations()} == {first.id, second.id}
    db.add_message(first.id, Role.USER, "work_mem là gì?")
    db.add_message(first.id, Role.ASSISTANT, "Là bộ nhớ... [1]", sources_json='[{"number": 1}]')
    db.add_message(second.id, Role.USER, "câu hỏi khác")
    messages = db.list_messages(first.id)
    assert [(m.role, m.content) for m in messages] == [(Role.USER, "work_mem là gì?"), (Role.ASSISTANT, "Là bộ nhớ... [1]")]
    assert messages[1].sources_json == '[{"number": 1}]' and messages[0].sources_json is None
    assert [m.id for m in db.list_messages(first.id, limit=1)] == [messages[1].id]  # the LAST one
    assert db.list_messages("missing") == []


def test_messages_need_a_real_conversation_and_a_valid_role(db: Database) -> None:
    conversation = db.create_conversation()
    with pytest.raises(sqlite3.IntegrityError):
        db.add_message("missing", Role.USER, "orphan")  # foreign key
    with sqlite3.connect(db.path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'admin', 'x', 'now')",
                     (conversation.id,))  # CHECK constraint


def test_deleting_a_conversation_cascades_to_its_messages(db: Database) -> None:
    conversation = db.create_conversation()
    db.add_message(conversation.id, Role.USER, "hi")
    with sqlite3.connect(db.path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation.id,))
    assert db.list_messages(conversation.id) == []
