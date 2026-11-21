"""Concurrent SQLite WAL + FTS5 BM25 index for deterministic, auditable RAG retrieval."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rag_engine.models import DocumentChunk, DocumentInput

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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def journal_mode(self) -> str:
        """Return active SQLite journal mode (expected: 'wal')."""
        row = self._conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]).lower() if row else ""

    def upsert_document(self, doc: DocumentInput, chunks: list[DocumentChunk]) -> int:
        """Atomically replace a document's chunks and FTS5 index rows without leaving stale data.

        Returns the exact number of chunks committed for doc.doc_id.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._conn:
            old_ids = [
                r["chunk_id"]
                for r in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc.doc_id,)
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", old_ids
                )
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))

            self._conn.execute(
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
                self._conn.execute(
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
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, title, section_ref, text) VALUES (?, ?, ?, ?)",
                    (ch.chunk_id, ch.title, ch.section_ref, ch.text),
                )
        return len(chunks)

    def search_fts_bm25(self, fts_match_expr: str, top_k: int = 3) -> list[dict[str, object]]:
        """Query FTS5 using BM25 ranking (lower raw bm25() = stronger match) and return top_k chunks."""
        rows = self._conn.execute(
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
            (fts_match_expr, top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    def verify_citation_ids(self, chunk_ids: list[str]) -> bool:
        """Verify that every cited chunk_id exists in the persisted chunks table (anti-hallucination check)."""
        if not chunk_ids:
            return False
        unique_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        row = self._conn.execute(
            f"SELECT COUNT(*) AS cnt FROM chunks WHERE chunk_id IN ({placeholders})",
            unique_ids,
        ).fetchone()
        return int(row["cnt"]) == len(unique_ids)

    def count_chunks(self) -> int:
        """Return total indexed chunks currently stored in SQLite."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection cleanly."""
        self._conn.close()

    def __enter__(self) -> CitationStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
