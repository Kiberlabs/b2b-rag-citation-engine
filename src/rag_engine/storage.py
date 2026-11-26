"""Concurrent SQLite WAL + FTS5 BM25 index for deterministic, auditable RAG retrieval."""

from __future__ import annotations

import random
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from rag_engine.models import DocumentChunk, DocumentInput

_T = TypeVar("_T")
_SQLITE_IN_BATCH_SIZE = 500
_REGISTRY_LOCK = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _close_conn_box(box: list[sqlite3.Connection | None]) -> None:
    conn = box[0]
    if conn is not None:
        box[0] = None
        conn.close()


def _lock_for_path(db_path: Path) -> threading.RLock:
    if str(db_path) == ":memory:":
        return threading.RLock()
    key = db_path.resolve()
    with _REGISTRY_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _retry_locked(op: Callable[[], _T], attempts: int = 20) -> _T:
    for attempt in range(attempts):
        try:
            return op()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(0.02 * min(attempt + 1, 6) + random.uniform(0.0, 0.03))
    raise AssertionError("unreachable")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    language TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    section_ref TEXT NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    section_ref,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class CitationStore:
    """Thread-safe SQLite WAL + FTS5 store with idempotent document re-ingestion and BM25 search."""

    def __init__(self, db_path: str | Path) -> None:
        """Open SQLite in WAL mode (5s busy_timeout, foreign keys ON) and initialize FTS5 tables."""
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock_for_path(self.db_path)
        self._conn_box: list[sqlite3.Connection | None] = [None]
        self._finalizer = weakref.finalize(self, _close_conn_box, self._conn_box)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except BaseException:
            conn.close()
            raise
        return conn

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn_box[0] is None:
            self._conn_box[0] = self._connect()
            if not self._finalizer.alive:
                self._finalizer = weakref.finalize(self, _close_conn_box, self._conn_box)
        return self._conn_box[0]

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._ensure_conn()

    def _drop_conn(self) -> None:
        _close_conn_box(self._conn_box)

    def _init_schema(self) -> None:
        with self._lock:
            for attempt in range(20):
                try:
                    conn = self._ensure_conn()
                    row = conn.execute("PRAGMA journal_mode;").fetchone()
                    if not row or str(row[0]).lower() != "wal":
                        conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.executescript(SCHEMA_SQL)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 19:
                        self._drop_conn()
                        raise
                    time.sleep(0.02 + random.uniform(0.0, 0.03))
                except BaseException:
                    self._drop_conn()
                    raise

    def journal_mode(self) -> str:
        """Return active SQLite journal mode (expected: 'wal')."""
        with self._lock:
            row = _retry_locked(lambda: self._ensure_conn().execute("PRAGMA journal_mode;").fetchone())
        return str(row[0]).lower() if row else ""

    def upsert_document(self, doc: DocumentInput, chunks: list[DocumentChunk]) -> int:
        """Atomically replace a document's chunks and FTS5 index rows without leaving stale data.

        Returns the exact number of chunks committed for doc.doc_id.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            def _write() -> int:
                conn = self._ensure_conn()
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    old_ids = [
                        r["chunk_id"]
                        for r in conn.execute(
                            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc.doc_id,)
                        ).fetchall()
                    ]
                    if old_ids:
                        for start in range(0, len(old_ids), _SQLITE_IN_BATCH_SIZE):
                            batch = old_ids[start : start + _SQLITE_IN_BATCH_SIZE]
                            placeholders = ",".join("?" for _ in batch)
                            conn.execute(
                                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                                batch,
                            )
                    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))

                    conn.execute(
                        """
                        INSERT INTO documents (doc_id, title, category, language, chunk_count, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(doc_id) DO UPDATE SET
                            title = excluded.title,
                            category = excluded.category,
                            language = excluded.language,
                            chunk_count = excluded.chunk_count,
                            updated_at = excluded.updated_at
                        """,
                        (doc.doc_id, doc.title, doc.category, doc.language, len(chunks), now_iso),
                    )

                    for ch in chunks:
                        conn.execute(
                            """
                            INSERT INTO chunks (
                                chunk_id, doc_id, title, section_ref, language, text, char_count, chunk_hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ch.chunk_id,
                                ch.doc_id,
                                ch.title,
                                ch.section_ref,
                                ch.language,
                                ch.text,
                                ch.char_count,
                                ch.chunk_hash,
                            ),
                        )
                        conn.execute(
                            "INSERT INTO chunks_fts (chunk_id, title, section_ref, text) VALUES (?, ?, ?, ?)",
                            (ch.chunk_id, ch.title, ch.section_ref, ch.text),
                        )
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK;")
                    raise
                else:
                    if conn.in_transaction:
                        conn.execute("COMMIT;")
                return len(chunks)

            return _retry_locked(_write)

    def search_fts_bm25(self, fts_match_expr: str, top_k: int = 3) -> list[dict[str, object]]:
        """Query FTS5 using BM25 ranking (lower raw bm25() = stronger match) and return top_k chunks."""
        if not fts_match_expr or not fts_match_expr.strip():
            return []
        safe_top_k = max(1, min(int(top_k), 250))
        with self._lock:
            rows = _retry_locked(
                lambda: self._ensure_conn().execute(
                    """
                    SELECT
                        c.chunk_id,
                        c.doc_id,
                        c.title,
                        c.section_ref,
                        c.text,
                        c.chunk_hash,
                        bm25(chunks_fts, 1.0, 2.0, 1.0) AS raw_bm25
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY raw_bm25 ASC
                    LIMIT ?
                    """,
                    (fts_match_expr, safe_top_k),
                ).fetchall()
            )
        return [dict(r) for r in rows]

    def verify_citation_ids(self, chunk_ids: list[str]) -> bool:
        """Verify that every cited chunk_id exists in the persisted chunks table (anti-hallucination check)."""
        if not chunk_ids:
            return False
        unique_ids = list(dict.fromkeys(chunk_ids))
        total = 0
        with self._lock:
            for start in range(0, len(unique_ids), _SQLITE_IN_BATCH_SIZE):
                batch = unique_ids[start : start + _SQLITE_IN_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                row = _retry_locked(
                    lambda b=batch, ph=placeholders: self._ensure_conn().execute(
                        f"SELECT COUNT(*) AS cnt FROM chunks WHERE chunk_id IN ({ph})",
                        b,
                    ).fetchone()
                )
                total += int(row["cnt"])
        return total == len(unique_ids)

    def count_chunks(self) -> int:
        """Return total indexed chunks currently stored in SQLite."""
        with self._lock:
            row = _retry_locked(lambda: self._ensure_conn().execute("SELECT COUNT(*) FROM chunks").fetchone())
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection cleanly."""
        with self._lock:
            self._finalizer()

    def __enter__(self) -> CitationStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
