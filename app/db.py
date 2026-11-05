"""SQLite (stdlib sqlite3, no ORM) storage for documents, conversations and messages.

Every operation opens its own short-lived connection and closes it. FastAPI runs `def` endpoints
and background tasks in different threads, and a sqlite3 connection must not be shared between
threads, so "one connection per operation" is the simplest correct choice. WAL mode lets readers
(status polling) work while a writer (the indexing job) commits.

All SQL is parametrized; values are never formatted into the query string.

`documents` is the source of truth for upload status. Conversations and messages are only schema
plus simple insert/list functions for now; the RAG flow does not use them yet.
"""

import logging
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_S = 10.0  # wait this long for a competing writer before "database is locked"


class DocStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    error_code    TEXT,
    error_message TEXT,
    num_pages     INTEGER,
    num_chunks    INTEGER,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources_json    TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);
"""


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    filename: str
    status: DocStatus
    error_code: str | None
    error_message: str | None
    num_pages: int | None
    num_chunks: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    conversation_id: str
    role: Role
    content: str
    sources_json: str | None
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        filename=row["filename"],
        status=DocStatus(row["status"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        num_pages=row["num_pages"],
        num_chunks=row["num_chunks"],
        created_at=row["created_at"],
    )


class Database:
    """A path plus short-lived connections. Cheap to create, safe to use from any thread."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error, and always close (sqlite3's own `with`
        only handles the transaction, it does not close the connection)."""
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")  # per connection, off by default
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create the folder, switch to WAL and create the tables. Safe to call repeatedly."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")  # persistent in the file
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ documents

    def claim_document(
        self, doc_id: str, filename: str, num_pages: int | None, num_chunks: int
    ) -> tuple[Document, bool]:
        """Atomically decide who indexes this document. Returns (row, claimed).

        claimed=True  -> the caller must index it: the row is new, or it was 'failed' and is now
                         'processing' again (retry).
        claimed=False -> it is already 'ready' or someone else is 'processing' it; do nothing.
        BEGIN IMMEDIATE takes the write lock first, so two simultaneous uploads of the same file
        cannot both claim it.
        """
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                claimed = row is None or row["status"] == DocStatus.FAILED
                if row is None:
                    conn.execute(
                        "INSERT INTO documents (doc_id, filename, status, num_pages, num_chunks, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (doc_id, filename, DocStatus.PROCESSING, num_pages, num_chunks, _now()),
                    )
                elif claimed:
                    conn.execute(
                        "UPDATE documents SET status = ?, filename = ?, error_code = NULL, "
                        "error_message = NULL, num_pages = ?, num_chunks = ? WHERE doc_id = ?",
                        (DocStatus.PROCESSING, filename, num_pages, num_chunks, doc_id),
                    )
                row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return _document(row), claimed
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _document(row) if row else None

    def get_documents(self, doc_ids: Sequence[str]) -> dict[str, Document]:
        """The existing documents among `doc_ids`, keyed by id (unknown ids are simply absent)."""
        if not doc_ids:
            return {}
        placeholders = ", ".join("?" for _ in doc_ids)  # only "?" characters are formatted in
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", tuple(doc_ids)
            ).fetchall()
        return {row["doc_id"]: _document(row) for row in rows}

    def list_documents(self) -> list[Document]:
        """Newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, filename, doc_id"
            ).fetchall()
        return [_document(row) for row in rows]

    def ready_doc_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE status = ? ORDER BY created_at, doc_id",
                (DocStatus.READY,),
            ).fetchall()
        return [row["doc_id"] for row in rows]

    def mark_ready(self, doc_id: str, num_chunks: int) -> bool:
        """processing -> ready. False if the row is gone or was not 'processing'."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = ?, num_chunks = ?, error_code = NULL, "
                "error_message = NULL WHERE doc_id = ? AND status = ?",
                (DocStatus.READY, num_chunks, doc_id, DocStatus.PROCESSING),
            )
        return cursor.rowcount == 1

    def mark_failed(self, doc_id: str, error_code: str, error_message: str) -> bool:
        """processing -> failed. False if the row is gone or was not 'processing'."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = ?, error_code = ?, error_message = ? "
                "WHERE doc_id = ? AND status = ?",
                (DocStatus.FAILED, error_code, error_message, doc_id, DocStatus.PROCESSING),
            )
        return cursor.rowcount == 1

    def fail_stuck_processing(self, error_code: str, error_message: str) -> int:
        """Startup recovery: nothing can still be indexing in a freshly started process, so every
        'processing' row is a leftover of a crash or restart. Returns how many were marked."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = ?, error_code = ?, error_message = ? WHERE status = ?",
                (DocStatus.FAILED, error_code, error_message, DocStatus.PROCESSING),
            )
        if cursor.rowcount:
            logger.warning("Marked %d document(s) stuck in 'processing' as failed", cursor.rowcount)
        return cursor.rowcount

    def delete_document(self, doc_id: str) -> bool:
        """Delete the row unless it is being indexed. True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND status != ?",
                (doc_id, DocStatus.PROCESSING),
            )
        return cursor.rowcount == 1

    # ------------------------------------------------------------------ conversations

    def create_conversation(self, title: str | None = None) -> Conversation:
        conversation = Conversation(id=uuid.uuid4().hex, title=title, created_at=_now())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                (conversation.id, conversation.title, conversation.created_at),
            )
        return conversation

    def list_conversations(self) -> list[Conversation]:
        """Newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY created_at DESC, id"
            ).fetchall()
        return [Conversation(r["id"], r["title"], r["created_at"]) for r in rows]

    def add_message(
        self, conversation_id: str, role: Role, content: str, sources_json: str | None = None
    ) -> Message:
        """Raises sqlite3.IntegrityError for an unknown conversation or a role that is not allowed."""
        created_at = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, str(role), content, sources_json, created_at),
            )
        return Message(cursor.lastrowid or 0, conversation_id, Role(role), content, sources_json, created_at)

    def list_messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        """Oldest first. With `limit`, the LAST `limit` messages (still oldest first)."""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                    "ORDER BY id DESC LIMIT ?) ORDER BY id",
                    (conversation_id, limit),
                ).fetchall()
        return [
            Message(r["id"], r["conversation_id"], Role(r["role"]), r["content"], r["sources_json"], r["created_at"])
            for r in rows
        ]
