"""SQLite (stdlib sqlite3, no ORM) storage for documents, conversations and messages.

Every operation opens its own short-lived connection and closes it. FastAPI runs `def` endpoints
and background tasks in different threads, and a sqlite3 connection must not be shared between
threads, so "one connection per operation" is the simplest correct choice. WAL mode lets readers
(status polling) work while a writer (the indexing job) commits.

All SQL is parametrized; values are never formatted into the query string.

`documents` is the source of truth for upload status. `conversations` + `messages` are the saved chat
history: one chat exchange (the question and the answer, and the conversation row itself when it is
new) is written in ONE transaction by `save_exchange`, so a failure can never leave half an exchange.
Messages reference their conversation with ON DELETE CASCADE. There is deliberately NO foreign key from
a conversation to `documents`: deleting a document must not delete the history that mentions it.
"""

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    id           TEXT PRIMARY KEY,
    title        TEXT,
    doc_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    updated_at   TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources_json    TEXT,
    created_at      TEXT NOT NULL,
    used_fallback   INTEGER CHECK (used_fallback IN (0, 1)),
    best_score      REAL,
    latency_json    TEXT,
    truncated       INTEGER CHECK (truncated IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);
CREATE TABLE IF NOT EXISTS generations (
    cache_key      TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN ('summary', 'quiz')),
    params_json    TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    llm_calls      INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generations_doc ON generations (doc_id);
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


# Columns added after the first version of the schema. `CREATE TABLE IF NOT EXISTS` does not add columns to a
# table that already exists, so init_db adds the missing ones. The names are constants of this file, never user input.
_ADDED_COLUMNS = {
    "conversations": {"doc_ids_json": "TEXT NOT NULL DEFAULT '[]'", "updated_at": "TEXT"},
    "messages": {
        "used_fallback": "INTEGER CHECK (used_fallback IN (0, 1))",
        "best_score": "REAL",
        "latency_json": "TEXT",
        "truncated": "INTEGER CHECK (truncated IN (0, 1))",
    },
}


class ConversationNotFound(LookupError):
    """The conversation id does not exist (never created, or deleted)."""


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str | None
    created_at: str
    updated_at: str = ""
    doc_ids: tuple[str, ...] = ()  # every document searched in this conversation, first use first
    message_count: int = 0


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    conversation_id: str
    role: Role
    content: str
    sources_json: str | None
    created_at: str
    # Only assistant messages carry these (None on a user message)
    used_fallback: bool | None = None
    best_score: float | None = None
    latency_json: str | None = None
    truncated: bool | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    """A cached summary or quiz. `llm_calls` is what producing it cost the first time."""

    cache_key: str
    doc_id: str
    kind: str
    result_json: str
    llm_calls: int
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_clock_lock = threading.Lock()
_last_precise: datetime | None = None


def _now_precise() -> str:
    """A strictly increasing UTC timestamp with microseconds, so conversations sort by activity.

    The system clock alone is not enough: on Windows it can return the same value for calls a few
    milliseconds apart, and two exchanges would then tie on `updated_at`. Within this process every
    call returns a later value than the previous one (bumped by 1 microsecond on a tie).
    """
    global _last_precise
    with _clock_lock:
        now = datetime.now(UTC)
        if _last_precise is not None and now <= _last_precise:
            now = _last_precise + timedelta(microseconds=1)
        _last_precise = now
        return now.isoformat(timespec="microseconds")


def _conversation(row: sqlite3.Row) -> Conversation:
    try:
        ids = json.loads(row["doc_ids_json"])
        doc_ids = tuple(i for i in ids if isinstance(i, str)) if isinstance(ids, list) else ()
    except ValueError:
        doc_ids = ()  # a damaged value must not make the whole history unreadable
    return Conversation(
        id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"] or row["created_at"],
        doc_ids=doc_ids, message_count=row["message_count"] if "message_count" in row.keys() else 0,  # noqa: SIM118 - sqlite3.Row
    )


def _message(row: sqlite3.Row) -> Message:
    flag = lambda value: None if value is None else bool(value)  # noqa: E731
    return Message(
        row["id"], row["conversation_id"], Role(row["role"]), row["content"], row["sources_json"], row["created_at"],
        used_fallback=flag(row["used_fallback"]), best_score=row["best_score"], latency_json=row["latency_json"],
        truncated=flag(row["truncated"]),
    )


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
            for table, columns in _ADDED_COLUMNS.items():
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608 - constants
                for name, definition in columns.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")  # noqa: S608 - constants
            conn.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at IS NULL")
            # after the migration: on an old database the column did not exist when the schema script ran
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at DESC)")

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

    # ------------------------------------------------------------------ generations (summary / quiz cache)

    def get_generation(self, cache_key: str) -> Generation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cache_key, doc_id, kind, result_json, llm_calls, created_at FROM generations WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return Generation(**dict(row)) if row else None

    def put_generation(
        self, cache_key: str, doc_id: str, kind: str, params_json: str, model: str, prompt_version: str,
        result_json: str, llm_calls: int,
    ) -> None:
        """Store (or replace) a result. Raises sqlite3.IntegrityError if the document no longer exists."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO generations (cache_key, doc_id, kind, params_json, model, prompt_version, "
                "result_json, llm_calls, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cache_key, doc_id, kind, params_json, model, prompt_version, result_json, llm_calls, _now()),
            )

    def count_generations(self, doc_id: str | None = None) -> int:
        with self._connect() as conn:
            if doc_id is None:
                return conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM generations WHERE doc_id = ?", (doc_id,)).fetchone()[0]

    # ------------------------------------------------------------------ conversations

    def create_conversation(self, title: str | None = None, doc_ids: Sequence[str] = ()) -> Conversation:
        now = _now_precise()
        conversation = Conversation(
            id=uuid.uuid4().hex, title=title, created_at=now, updated_at=now, doc_ids=tuple(dict.fromkeys(doc_ids))
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, doc_ids_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation.id, conversation.title, json.dumps(list(conversation.doc_ids)), now, now),
            )
        return conversation

    def list_conversations(self, limit: int | None = None, offset: int = 0) -> list[Conversation]:
        """Most recently active first (a new message moves a conversation to the top). `limit=None` = all."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ? OFFSET ?",
                (-1 if limit is None else limit, offset),  # SQLite: LIMIT -1 means no limit
            ).fetchall()
        return [_conversation(r) for r in rows]

    def count_conversations(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
        return _conversation(row) if row else None

    def delete_conversation(self, conversation_id: str) -> int | None:
        """Delete a conversation and (by ON DELETE CASCADE) its messages. Returns how many messages went with it,
        or None if there was no such conversation."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return count if cursor.rowcount == 1 else None

    def save_exchange(
        self, *, conversation_id: str | None, title: str, doc_ids: Sequence[str], question: str, answer: str,
        sources_json: str, used_fallback: bool, best_score: float | None, latency_json: str, truncated: bool,
    ) -> str:
        """Save one chat exchange and return the conversation id. ALL OR NOTHING.

        The conversation row (when `conversation_id` is None it is created here), the user message, the assistant
        message and the `updated_at` bump happen in one transaction (BEGIN IMMEDIATE takes the write lock first,
        so two exchanges cannot interleave their doc_ids update). If any statement fails nothing is kept: there is
        never a conversation without messages, nor a question without its answer.
        Raises ConversationNotFound for an id that does not exist; sqlite3 errors propagate.
        """
        now = _now_precise()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conversation_id is None:
                conversation_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO conversations (id, title, doc_ids_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, title, json.dumps(list(dict.fromkeys(doc_ids))), now, now),
                )
            else:
                row = conn.execute("SELECT doc_ids_json FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if row is None:
                    raise ConversationNotFound(conversation_id)
                try:
                    known = [i for i in json.loads(row["doc_ids_json"]) if isinstance(i, str)]
                except ValueError:
                    known = []
                merged = list(dict.fromkeys([*known, *doc_ids]))
                conn.execute(
                    "UPDATE conversations SET updated_at = ?, doc_ids_json = ? WHERE id = ?",
                    (now, json.dumps(merged), conversation_id),
                )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, str(Role.USER), question, now),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources_json, created_at, used_fallback, "
                "best_score, latency_json, truncated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, str(Role.ASSISTANT), answer, sources_json, now, int(used_fallback), best_score,
                 latency_json, int(truncated)),
            )
        return conversation_id

    def add_message(
        self, conversation_id: str, role: Role, content: str, sources_json: str | None = None, *,
        used_fallback: bool | None = None, best_score: float | None = None, latency_json: str | None = None,
        truncated: bool | None = None,
    ) -> Message:
        """Raises sqlite3.IntegrityError for an unknown conversation or a role that is not allowed."""
        created_at = _now_precise()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources_json, created_at, used_fallback, "
                "best_score, latency_json, truncated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, str(role), content, sources_json, created_at,
                 None if used_fallback is None else int(used_fallback), best_score, latency_json,
                 None if truncated is None else int(truncated)),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (created_at, conversation_id))
        return Message(cursor.lastrowid or 0, conversation_id, Role(role), content, sources_json, created_at,
                       used_fallback, best_score, latency_json, truncated)

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
        return [_message(r) for r in rows]
