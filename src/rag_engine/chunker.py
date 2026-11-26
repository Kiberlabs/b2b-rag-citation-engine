"""Deterministic article- and section-aware document chunker with SHA-256 fingerprinting."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from rag_engine.models import DocumentChunk, DocumentInput

# Recognizes headings like "Article 4:", "Artículo 12 -", "Section 3.1", or "## Clause"
_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+|(?:Article|Art[ií]culo|Section|Secci[oó]n|Clause)\s+[0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)


def compute_chunk_hash(doc_id: str, section_ref: str, text: str) -> str:
    """Return a stable 16-char SHA-256 fingerprint for audit-trail citation verification."""
    payload = f"{doc_id}|{section_ref}|{text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Break oversized sections on sentence boundaries without cutting words mid-token."""
    if isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                parts.append(" ".join(buf))
                buf, buf_len = [], 0
            for start in range(0, len(sent), max_chars):
                parts.append(sent[start : start + max_chars].strip())
            continue

        if buf and buf_len + len(sent) + 1 > max_chars:
            parts.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += len(sent) + (1 if buf_len else 0)

    if buf:
        parts.append(" ".join(buf))
    return [p for p in parts if p]


def split_into_section_chunks(doc: DocumentInput, max_chars: int = 600) -> list[DocumentChunk]:
    """Segment a regulatory or policy document into deterministic, citation-addressed chunks.

    Flow: split by blank lines or article/section headings -> preserve exact section_ref
    headers -> subdivide oversized clauses on sentence boundaries -> attach SHA-256 hashes.
    """
    if isinstance(max_chars, bool) or max_chars < 50:
        raise ValueError("max_chars must be >= 50")
    sections: list[tuple[str, list[str]]] = []
    current_ref = "General"
    current_lines: list[str] = []

    for raw_line in unicodedata.normalize("NFC", doc.content).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            if current_lines:
                sections.append((current_ref, current_lines))
                current_lines = []
            current_ref = line.lstrip("#").strip().rstrip(":")
            current_lines.append(line.lstrip("#").strip())
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_ref, current_lines))

    chunks: list[DocumentChunk] = []
    seq = 0
    for section_ref, lines in sections:
        body = " ".join(lines).strip()
        for part_idx, segment in enumerate(_split_long_text(body, max_chars=max_chars), start=1):
            seq += 1
            ref_label = section_ref if part_idx == 1 else f"{section_ref} (part {part_idx})"
            chunk_id = f"{doc.doc_id}::c{seq:03d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section_ref=ref_label,
                    language=doc.language,
                    text=segment,
                    char_count=len(segment),
                    chunk_hash=compute_chunk_hash(doc.doc_id, ref_label, segment),
                )
            )
    return chunks
