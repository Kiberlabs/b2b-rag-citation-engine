"""Automated unit, security (FTS5 injection), SQLite WAL, and grounded citation tests."""

from __future__ import annotations

import gc
import json
import random
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_engine.chunker import split_into_section_chunks
from rag_engine.cli import ingest_corpus_file, main as cli_main
from rag_engine.models import DocumentChunk, DocumentInput
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


def test_rare_token_dilution_retrieves_high_overlap_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "dilution.db"
    with CitationStore(db_path) as store:
        for i in range(5):
            noise = DocumentInput(
                doc_id=f"NOISE-{i}",
                title=f"Zebra Noise Doc {i}",
                content=f"Article 1: Noise\nZebras migrate across seasonal plains with repeated zebra {i} notes.",
            )
            store.upsert_document(noise, split_into_section_chunks(noise))

        target = DocumentInput(
            doc_id="TARGET-1",
            title="Retention Mandate",
            content=(
                "Article 9: Compliance\n"
                "Encrypted audit logs require mandatory verification retention for enterprise tenants."
            ),
        )
        store.upsert_document(target, split_into_section_chunks(target))

        ans = build_grounded_response(
            "zebra encrypted audit retention verification",
            store,
            top_k=2,
            min_token_overlap=0.5,
        )
        assert ans.status == "GROUNDED"
        assert ans.verified is True
        assert 1 <= len(ans.citations) <= 2
        assert ans.citations[0].doc_id == "TARGET-1"


def test_top_k_clamping_across_retriever_storage_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "clamp.db"
    with CitationStore(db_path) as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        assert store.search_fts_bm25("", top_k=3) == []
        assert len(store.search_fts_bm25('"data"', top_k=-1)) == 1
        assert len(store.search_fts_bm25('"data"', top_k=0)) == 1

        ans_neg = build_grounded_response("data breach notification", store, top_k=-1)
        assert ans_neg.status == "GROUNDED"
        assert len(ans_neg.citations) == 1

    rc = cli_main(
        [
            "query",
            "--corpus",
            str(_sample_corpus_path()),
            "--db",
            str(tmp_path / "cli_clamp.db"),
            "--q",
            "data breach notification",
            "--top-k",
            "-1",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["result"]["citations"]) == 1


@pytest.mark.parametrize("bad_max_chars", [0, -5, 49, True])
def test_chunker_rejects_max_chars_below_50(bad_max_chars: int) -> None:
    doc = DocumentInput(
        doc_id="DOC-CHK",
        title="Chunker Guard",
        content="Article 1: Scope\nMinimum chunk boundary verification for safety.",
    )
    with pytest.raises(ValueError, match="max_chars"):
        split_into_section_chunks(doc, max_chars=bad_max_chars)


def test_citation_store_multithreaded_upsert_and_search(tmp_path: Path) -> None:
    db_path = tmp_path / "mt_rag.db"
    with CitationStore(db_path) as store:
        seed = DocumentInput(
            doc_id="SEED-0",
            title="Seed Policy",
            content="Article 1: Audit\nCompliance telemetry retention verification policy.",
        )
        store.upsert_document(seed, split_into_section_chunks(seed))

        workers = 12
        barrier = threading.Barrier(workers)
        errors: list[BaseException] = []
        results: list[bool] = []
        lock = threading.Lock()

        def run_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5.0)
                if idx % 2 == 0:
                    doc = DocumentInput(
                        doc_id=f"MT-{idx}",
                        title=f"Worker Doc {idx}",
                        content=(
                            f"Article {idx}: Telemetry\n"
                            "Compliance telemetry retention verification policy applies continuously."
                        ),
                    )
                    store.upsert_document(doc, split_into_section_chunks(doc))
                else:
                    ans = build_grounded_response("compliance telemetry retention verification", store)
                    with lock:
                        results.append(ans.verified and ans.status == "GROUNDED")
            except BaseException as exc:  # pragma: no cover - captured for assertion
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=run_worker, args=(i,)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == []
        assert len(results) == 6
        assert all(results)
        assert store.count_chunks() == 7


def test_null_bytes_empty_queries_and_nan_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DocumentInput(doc_id="BAD-TITLE", title=" \x00  ", content="Valid document body with enough chars.")
    with pytest.raises(ValidationError):
        DocumentInput(doc_id="BAD-CAT", title="Valid Title", category=" \x00 ", content="Valid document body with enough chars.")

    with CitationStore(tmp_path / "edge.db") as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        for q in ("", "   ", "\x00\x00", "the and for with"):
            ans = build_grounded_response(q, store)
            assert ans.status == "INSUFFICIENT_EVIDENCE"
            assert ans.verified is True
            assert ans.citations == []

        assert extract_query_tokens("\x00breach\x00 notification") == ["breach", "notification"]
        with pytest.raises(ValueError, match="min_token_overlap"):
            build_grounded_response("breach", store, min_token_overlap=float("nan"))


@pytest.mark.parametrize(
    "query",
    [
        "cake in to is",
        "pan en el de",
        "to in of",
        "en el de la es un al",
        "the and for with",
        "on at by an is as",
    ],
)
def test_stopword_only_and_nonsense_stopword_queries_abstain(tmp_path: Path, query: str) -> None:
    with CitationStore(tmp_path / "stopwords.db") as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        ans = build_grounded_response(query, store)
        assert ans.status == "INSUFFICIENT_EVIDENCE"
        assert ans.citations == []
        assert ans.verified is True


def test_single_real_word_with_2letter_stopwords_still_grounds(tmp_path: Path) -> None:
    with CitationStore(tmp_path / "single_real.db") as store:
        ingest_corpus_file(_sample_corpus_path(), store)
        for q, expected_doc in [
            ("breach", "REG-GDPR-2026"),
            ("breach in to is", "REG-GDPR-2026"),
            ("facturas en el de", "NORM-ES-FACTURA"),
        ]:
            ans = build_grounded_response(q, store)
            assert ans.status == "GROUNDED"
            assert ans.verified is True
            assert ans.citations[0].doc_id == expected_doc
            assert ans.citations[0].token_overlap == 1.0


def test_33k_chunk_reingestion_batched_delete_and_fts_consistency(tmp_path: Path) -> None:
    db_path = tmp_path / "huge_33k.db"
    with CitationStore(db_path) as store:
        doc_v1 = DocumentInput(
            doc_id="HUGE-33K",
            title="Massive Regulatory Corpus",
            content="Article 1: Baseline\nLegacy archival policy placeholder text for 33k chunk stress.",
        )
        chunks_v1 = [
            DocumentChunk(
                chunk_id=f"HUGE-33K::c{i:05d}",
                doc_id="HUGE-33K",
                title=doc_v1.title,
                section_ref=f"Section {i}",
                language="en",
                text=f"Legacytoken clause number {i} for batched delete verification.",
                char_count=len(f"Legacytoken clause number {i} for batched delete verification."),
                chunk_hash=f"{i:016x}",
            )
            for i in range(1, 33_501)
        ]
        assert store.upsert_document(doc_v1, chunks_v1) == 33_500
        assert store.count_chunks() == 33_500
        assert store.verify_citation_ids([c.chunk_id for c in chunks_v1]) is True

        doc_v2 = DocumentInput(
            doc_id="HUGE-33K",
            title="Massive Regulatory Corpus v2",
            content="Article 1: Erasure\nCryptographic erasure verification replaces all stale clauses.",
        )
        chunks_v2 = split_into_section_chunks(doc_v2)
        assert store.upsert_document(doc_v2, chunks_v2) == 1
        assert store.count_chunks() == 1
        fts_count = store._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_count == 1

        assert build_grounded_response("legacytoken", store).status == "INSUFFICIENT_EVIDENCE"
        ans = build_grounded_response("cryptographic erasure verification", store)
        assert ans.status == "GROUNDED"
        assert ans.verified is True
        assert ans.citations[0].doc_id == "HUGE-33K"


def test_large_document_chunking_over_1000_sections(tmp_path: Path) -> None:
    sections = [
        f"Article {i}: Telemetry Clause {i}\nEncrypted compliance audit trail mandate number {i} applies globally."
        for i in range(1, 1201)
    ]
    doc = DocumentInput(
        doc_id="BIG-DOC",
        title="Global Telemetry Standard",
        content="\n\n".join(sections),
    )
    chunks = split_into_section_chunks(doc, max_chars=400)
    assert len(chunks) == 1200
    assert chunks[-1].chunk_id == "BIG-DOC::c1200"
    assert all(1 <= ch.char_count <= 400 for ch in chunks)

    with CitationStore(tmp_path / "big_doc.db") as store:
        assert store.upsert_document(doc, chunks) == 1200
        ans = build_grounded_response("Telemetry Clause 1199 encrypted compliance", store, top_k=5)
        assert ans.status == "GROUNDED"
        assert ans.verified is True
        assert any("1199" in c.section_ref for c in ans.citations)


def test_randomized_property_fuzzing_queries_and_chunker(tmp_path: Path) -> None:
    rng = random.Random(20261123)
    stopwords = [
        "to", "in", "of", "on", "at", "by", "an", "is", "as",
        "de", "en", "el", "la", "es", "un", "al",
        "the", "and", "for", "with", "los", "las", "para", "sobre",
    ]
    nonces = ["zxqcake", "qwerty99", "zumbopan", "flurbos", "kryptonbake"]
    corpus_terms = ["breach", "72", "hours", "facturas", "escrow", "gdpr", "electronicas"]
    fts_ops = ['"', "*", "^", "(", ")", "OR", "AND", "NOT", "NEAR", "\x00", "á", "é", "ñ"]

    with CitationStore(tmp_path / "fuzz.db") as store:
        ingest_corpus_file(_sample_corpus_path(), store)

        for _ in range(120):
            picked = rng.choices(stopwords, k=rng.randint(1, 5)) + rng.choices(nonces, k=rng.randint(0, 2))
            rng.shuffle(picked)
            ans = build_grounded_response(" ".join(picked), store)
            assert ans.status == "INSUFFICIENT_EVIDENCE"
            assert ans.citations == []
            assert ans.verified is True

        for _ in range(150):
            parts = rng.choices(stopwords + corpus_terms + nonces + fts_ops, k=rng.randint(1, 8))
            query = " ".join(parts)
            top_k = rng.randint(1, 5)
            min_ov = round(rng.uniform(0.1, 0.8), 2)
            ans = build_grounded_response(query, store, top_k=top_k, min_token_overlap=min_ov)
            assert ans.verified is True
            assert len(ans.citations) <= top_k
            assert all(c.token_overlap >= min_ov for c in ans.citations)

        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 áéíóúñ"
        for i in range(40):
            n_sections = rng.randint(1, 8)
            body = "\n\n".join(
                f"Article {s}: Topic {s}\n"
                + " ".join(
                    "".join(rng.choices(alphabet, k=rng.randint(3, 14))).strip() or "clause"
                    for _ in range(rng.randint(6, 30))
                )
                for s in range(1, n_sections + 1)
            )
            doc = DocumentInput(doc_id=f"FUZZ-{i}", title=f"Fuzz Doc {i}", content=body)
            max_chars = rng.randint(50, 800)
            chunks = split_into_section_chunks(doc, max_chars=max_chars)
            assert len(chunks) >= 1
            assert len({c.chunk_id for c in chunks}) == len(chunks)
            assert all(1 <= c.char_count == len(c.text) <= max_chars for c in chunks)
            assert store.upsert_document(doc, chunks) == len(chunks)
            assert store.upsert_document(doc, chunks) == len(chunks)


def test_unicode_nfd_nfkc_zero_width_chunking_and_retrieval(tmp_path: Path) -> None:
    for zw in ("\u200b\u200c\u200d", "\ufeff \u2060"):
        with pytest.raises(ValidationError):
            DocumentInput(doc_id="ZW-T", title=zw, content="Valid body text with enough characters.")
        with pytest.raises(ValidationError):
            DocumentInput(doc_id="ZW-C", title="Valid Title", category=zw, content="Valid body text with enough characters.")
        with pytest.raises(ValidationError):
            DocumentInput(doc_id="ZW-B", title="Valid Title", content=zw * 10)

    nfd_doc = DocumentInput(
        doc_id="NFD-ES",
        title="Normativa NFD",
        language="es",
        content="Arti\u0301culo 3: Conservacio\u0301n\nLas facturas electro\u0301nicas deben conservarse durante 4 an\u0303os.",
    )
    chunks = split_into_section_chunks(nfd_doc, max_chars=400)
    assert chunks[0].section_ref == "Artículo 3: Conservación"

    with CitationStore(tmp_path / "unicode.db") as store:
        store.upsert_document(nfd_doc, chunks)
        # Zero-width split + NFKC fullwidth query must still ground cleanly
        ans_zw = build_grounded_response("fact\u200buras electro\u200bnicas", store)
        assert ans_zw.status == "GROUNDED"
        assert ans_zw.citations[0].doc_id == "NFD-ES"

        ans_fw = build_grounded_response("\uff46\uff41\uff43\uff54\uff55\uff52\uff41\uff53", store)
        assert ans_fw.status == "GROUNDED"
        assert ans_fw.citations[0].doc_id == "NFD-ES"


def test_top_k_int64_bounds_and_16_thread_wal_stress_and_rmtree(tmp_path: Path) -> None:
    wal_dir = tmp_path / "wal16"
    wal_dir.mkdir()
    db_path = wal_dir / "stress16.db"
    barrier = threading.Barrier(16)
    errors: list[tuple[int, Exception]] = []
    err_lock = threading.Lock()

    with CitationStore(db_path) as seed_store:
        ingest_corpus_file(_sample_corpus_path(), seed_store)
        assert seed_store.journal_mode() == "wal"
        assert seed_store._ensure_conn().execute("PRAGMA busy_timeout;").fetchone()[0] == 5000

    def _worker(tid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            with CitationStore(db_path) as store:
                doc = DocumentInput(
                    doc_id=f"S16-{tid:02d}",
                    title=f"Concurrent Policy {tid}",
                    content=f"Article {tid}: Telemetry\nEncrypted telemetry compliance verification retention clause {tid}.",
                )
                store.upsert_document(doc, split_into_section_chunks(doc))
                ans = build_grounded_response("telemetry compliance verification retention", store)
                assert ans.status == "GROUNDED"
                assert ans.verified is True
        except Exception as exc:
            with err_lock:
                errors.append((tid, exc))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_worker, range(16)))

    assert errors == []
    with CitationStore(db_path) as store:
        for bound in (0, 2**63 - 1, 2**63, -(2**63) - 1):
            ans = build_grounded_response("data breach notification", store, top_k=bound)
            assert ans.status == "GROUNDED"
            assert ans.verified is True

    shutil.rmtree(wal_dir)
    assert not wal_dir.exists()

    # Unclosed CitationStore relies on weakref.finalize during GC before rmtree (WinError 32 check)
    leak_dir = tmp_path / "wal_gc"
    leak_dir.mkdir()
    leak_db = leak_dir / "gc.db"

    def _unclosed() -> None:
        s = CitationStore(leak_db)
        ingest_corpus_file(_sample_corpus_path(), s)

    _unclosed()
    gc.collect()
    shutil.rmtree(leak_dir)
    assert not leak_dir.exists()
