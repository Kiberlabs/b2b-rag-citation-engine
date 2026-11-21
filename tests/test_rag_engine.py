"""Automated unit, security (FTS5 injection), SQLite WAL, and grounded citation tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_engine.chunker import split_into_section_chunks
from rag_engine.cli import ingest_corpus_file, main as cli_main
from rag_engine.models import DocumentInput
from rag_engine.retriever import build_grounded_response, extract_query_tokens
from rag_engine.storage import CitationStore


def _sample_corpus_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"


def test_article_aware_chunking_preserves_section_refs_and_hashes() -> None:
    doc = DocumentInput(
        doc_id="LAW-01",
        title="Sample Regulatory Code",
        language="en",
        content=(
            "Article 1: Definitions\nProcessors handle enterprise telemetry strictly under contract.\n\n"
            "Article 2: Breach SLA\nNotification must occur within 72 hours of confirmed detection."
        ),
    )
    chunks = split_into_section_chunks(doc, max_chars=400)
    assert len(chunks) == 2
    assert chunks[0].section_ref == "Article 1: Definitions"
    assert chunks[1].section_ref == "Article 2: Breach SLA"
    assert chunks[0].chunk_hash != chunks[1].chunk_hash
    assert len(chunks[0].chunk_hash) == 16


def test_document_input_rejects_short_or_null_byte_content() -> None:
    with pytest.raises(ValidationError):
        DocumentInput(doc_id="BAD-1", title="Empty", content="\x00   short   ")


def test_idempotent_reingestion_replaces_stale_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        assert store.journal_mode() == "wal"

        v1 = DocumentInput(
            doc_id="DOC-UP",
            title="Retention Policy",
            content="Article 1: Retention\nRecords are retained for 12 months before archival.",
        )
        store.upsert_document(v1, split_into_section_chunks(v1))
        assert store.count_chunks() == 1

        v2 = DocumentInput(
            doc_id="DOC-UP",
            title="Retention Policy v2",
            content=(
                "Article 1: Retention\nRecords are retained for 36 months.\n\n"
                "Article 2: Deletion\nExpired backups are cryptographically wiped."
            ),
        )
        store.upsert_document(v2, split_into_section_chunks(v2))
        assert store.count_chunks() == 2

        old_ans = build_grounded_response("archival", store)
        assert old_ans.status == "INSUFFICIENT_EVIDENCE"

        new_ans = build_grounded_response("cryptographically wiped expired backups", store)
        assert new_ans.status == "GROUNDED"
        assert "Article 2" in new_ans.citations[0].section_ref


def test_grounded_retrieval_english_legal_query(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        docs_ok, chunks_ok, rejected = ingest_corpus_file(_sample_corpus_path(), store)
        assert docs_ok == 3
        assert chunks_ok == 7
        assert rejected == []

        ans = build_grounded_response("What is the deadline for data breach notification?", store)
        assert ans.status == "GROUNDED"
        assert ans.verified is True
        assert ans.citations[0].doc_id == "REG-GDPR-2026"
        assert "Article 4" in ans.citations[0].section_ref
        assert "72 hours" in ans.citations[0].snippet


def test_grounded_retrieval_spanish_accent_insensitive(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        ingest_corpus_file(_sample_corpus_path(), store)

        # Query intentionally omits accents ('retencion', 'facturas', 'electronicas')
        ans = build_grounded_response("Plazo minimo de conservacion de facturas electronicas", store)
        assert ans.status == "GROUNDED"
        assert ans.verified is True
        assert ans.citations[0].doc_id == "NORM-ES-FACTURA"
        assert "Artículo 3" in ans.citations[0].section_ref


def test_zero_hallucination_abstention_on_out_of_corpus_query(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        ingest_corpus_file(_sample_corpus_path(), store)

        ans = build_grounded_response("How do I bake sourdough croissants at 200 degrees?", store)
        assert ans.status == "INSUFFICIENT_EVIDENCE"
        assert ans.citations == []
        assert ans.verified is True


@pytest.mark.parametrize(
    "malicious_query",
    [
        '"breach" OR NOT NEAR(a,b) * ^ (())',
        "'; DROP TABLE chunks; --",
        "*** ??? !!! :::",
    ],
)
def test_fts5_syntax_and_sql_injection_resilience(tmp_path: Path, malicious_query: str) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        ans = build_grounded_response(malicious_query, store)
        assert ans.status in ("GROUNDED", "INSUFFICIENT_EVIDENCE")
        assert store.count_chunks() == 7


def test_verify_citation_ids_blocks_fabricated_citations(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    with CitationStore(db_path) as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        assert store.verify_citation_ids(["REG-GDPR-2026::c001"]) is True
        assert store.verify_citation_ids(["FAKE-DOC::c999"]) is False
        assert store.verify_citation_ids([]) is False


def test_cli_json_output_and_wal_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "cli_rag.db"
    rc = cli_main(
        [
            "query",
            "--corpus",
            str(_sample_corpus_path()),
            "--db",
            str(db_path),
            "--q",
            "escrow milestone acceptance 5 business days",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ingested_docs"] == 3
    assert payload["result"]["status"] == "GROUNDED"
    assert payload["result"]["citations"][0]["doc_id"] == "POL-ESCROW-SLA"

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
