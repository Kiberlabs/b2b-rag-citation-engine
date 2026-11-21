"""Pydantic v2 schemas for structured regulatory/B2B documents, chunks, and verified citations."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class DocumentInput(BaseModel):
    """Raw regulatory, legal, or corporate policy document submitted for indexed ingestion."""

    doc_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=3, max_length=200)
    category: str = Field(default="regulation", min_length=2, max_length=64)
    language: Literal["en", "es"] = "en"
    content: str = Field(min_length=20)

    @field_validator("content")
    @classmethod
    def _reject_null_bytes(cls, value: str) -> str:
        """Strip null bytes and reject whitespace-only documents at ingestion boundary."""
        cleaned = value.replace("\x00", "").strip()
        if len(cleaned) < 20:
            raise ValueError("Document content must contain at least 20 non-whitespace characters")
        return cleaned


class DocumentChunk(BaseModel):
    """Deterministic article- or section-scoped text unit stored in SQLite FTS5."""

    chunk_id: str
    doc_id: str
    title: str
    section_ref: str
    language: Literal["en", "es"]
    text: str
    char_count: int
    chunk_hash: str


class CitationHit(BaseModel):
    """Single verified evidence chunk returned by BM25 full-text retrieval."""

    chunk_id: str
    doc_id: str
    title: str
    section_ref: str
    bm25_score: float
    token_overlap: float
    snippet: str
    chunk_hash: str


class GroundedAnswer(BaseModel):
    """Auditable retrieval verdict that either cites verified chunks or explicitly abstains."""

    query: str
    status: Literal["GROUNDED", "INSUFFICIENT_EVIDENCE"]
    answer_summary: str
    citations: list[CitationHit]
    verified: bool
