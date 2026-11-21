"""CLI runner for KiberLabs B2B RAG & Grounded Citation Engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from rag_engine.chunker import split_into_section_chunks
from rag_engine.models import DocumentInput
from rag_engine.retriever import build_grounded_response
from rag_engine.storage import CitationStore


def ingest_corpus_file(
    corpus_path: Path,
    store: CitationStore,
    max_chars: int = 600,
) -> tuple[int, int, list[dict[str, str]]]:
    """Load, validate, chunk, and idempotently index a JSON corpus into SQLite FTS5/WAL.

    Returns (ingested_docs_count, total_chunks_written, rejected_docs).
    """
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Corpus JSON must be a list of document objects.")

    ingested_docs = 0
    total_chunks = 0
    rejected: list[dict[str, str]] = []

    for idx, entry in enumerate(raw):
        try:
            doc = DocumentInput.model_validate(entry)
        except ValidationError as exc:
            doc_id = str(entry.get("doc_id", f"row_{idx}")) if isinstance(entry, dict) else f"row_{idx}"
            rejected.append({"doc_id": doc_id, "error": exc.errors()[0]["msg"]})
            continue
        chunks = split_into_section_chunks(doc, max_chars=max_chars)
        total_chunks += store.upsert_document(doc, chunks)
        ingested_docs += 1

    return ingested_docs, total_chunks, rejected


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: ingests a JSON corpus into SQLite WAL+FTS5 and executes a grounded query."""
    parser = argparse.ArgumentParser(
        prog="rag-engine",
        description="KiberLabs B2B RAG Ingestion & Zero-Hallucination Citation Engine.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("query", help="Ingest corpus JSON into SQLite FTS5/WAL and answer a query with citations")
    run_p.add_argument("--corpus", type=Path, required=True, help="Path to regulatory/policy JSON corpus")
    run_p.add_argument("--db", type=Path, default=Path("data/rag.db"), help="SQLite WAL+FTS5 database path")
    run_p.add_argument("--q", required=True, help="User question to retrieve grounded citations for")
    run_p.add_argument("--top-k", type=int, default=3, help="Maximum number of citation hits (default: 3)")
    run_p.add_argument("--json", action="store_true", help="Emit structured JSON output")

    args = parser.parse_args(argv)
    with CitationStore(args.db) as store:
        docs_ok, chunks_ok, rejected = ingest_corpus_file(args.corpus, store)
        answer = build_grounded_response(args.q, store, top_k=args.top_k)

    if args.json:
        payload = {
            "ingested_docs": docs_ok,
            "indexed_chunks": chunks_ok,
            "rejected_docs": rejected,
            "result": answer.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"[KiberLabs RAG] Docs={docs_ok} | Chunks={chunks_ok} | "
            f"Status={answer.status} | Verified={answer.verified}"
        )
        print(f"  Summary: {answer.answer_summary}")
        for hit in answer.citations:
            print(
                f"  -> [{hit.doc_id} | {hit.section_ref}] overlap={hit.token_overlap:.0%} "
                f"hash={hit.chunk_hash}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
