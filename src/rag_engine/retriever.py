"""Safe FTS5 query sanitizer, BM25 retriever, and zero-hallucination citation verifier."""

from __future__ import annotations

import math
import re
import unicodedata

from rag_engine.models import CitationHit, GroundedAnswer
from rag_engine.storage import CitationStore

_STOPWORDS = frozenset(
    {
        "to", "in", "of", "on", "at", "by", "an", "is", "as",
        "de", "en", "el", "la", "es", "un", "al",
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "what", "when", "how",
        "does", "under", "about", "into", "your", "have", "must", "can", "will",
        "los", "las", "del", "una", "uno", "para", "con", "que", "como", "cual", "cuando",
        "sobre", "por", "son", "esta", "este", "puede", "debe", "entre",
    }
)


_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def _strip_accents(text: str) -> str:
    """Fold Unicode accents (e.g. 'artículo' -> 'articulo') for bilingual EN/ES lexical overlap."""
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    nfkd = unicodedata.normalize("NFKD", cleaned)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def extract_query_tokens(query: str) -> list[str]:
    """Extract sanitized alphanumeric query terms, stripping FTS5 operators and stopwords.

    Prevents FTS5 syntax injection crashes caused by quotes, asterisks, or boolean keywords.
    """
    folded = _strip_accents(query.replace("\x00", "")).lower()
    raw_tokens = re.findall(r"[a-z0-9]+", folded)
    filtered = [tok for tok in raw_tokens if len(tok) >= 2 and tok not in _STOPWORDS]
    # Deduplicate while preserving order, bounded to avoid oversized FTS5 expressions
    return list(dict.fromkeys(filtered))[:32]


def build_safe_fts_query(tokens: list[str]) -> str:
    """Quote every sanitized token and join with OR so FTS5 never interprets user text as operators."""
    return " OR ".join(f'"{tok}"' for tok in tokens)


def compute_token_overlap(query_tokens: list[str], candidate_text: str) -> float:
    """Compute fraction of query tokens present in a candidate chunk (0.0–1.0)."""
    if not query_tokens:
        return 0.0
    folded_text = _strip_accents(candidate_text).lower()
    cand_tokens = set(re.findall(r"[a-z0-9]+", folded_text))
    hits = sum(1 for tok in query_tokens if tok in cand_tokens)
    return round(hits / len(query_tokens), 4)


def build_grounded_response(
    query: str,
    store: CitationStore,
    top_k: int = 3,
    min_token_overlap: float = 0.25,
) -> GroundedAnswer:
    """Retrieve BM25-ranked evidence chunks and enforce zero-hallucination citation verification.

    Flow: sanitize query tokens -> execute FTS5 BM25 search -> filter hits below min_token_overlap ->
    verify every cited chunk_id exists in SQLite -> return GROUNDED answer or INSUFFICIENT_EVIDENCE.
    """
    if isinstance(min_token_overlap, bool) or not math.isfinite(min_token_overlap) or not (0.0 <= min_token_overlap <= 1.0):
        raise ValueError("min_token_overlap must be between 0.0 and 1.0")
    safe_top_k = max(1, min(int(top_k), 50))
    tokens = extract_query_tokens(query)
    if not tokens:
        return GroundedAnswer(
            query=query,
            status="INSUFFICIENT_EVIDENCE",
            answer_summary="Query contains no searchable domain terms; refusing to fabricate an answer.",
            citations=[],
            verified=True,
        )

    fts_expr = build_safe_fts_query(tokens)
    candidate_k = max(safe_top_k * 5, 25)
    rows = store.search_fts_bm25(fts_expr, top_k=candidate_k)

    hits: list[CitationHit] = []
    for row in rows:
        searchable = f"{row['title']} {row['section_ref']} {row['text']}"
        overlap = compute_token_overlap(tokens, searchable)
        if overlap < min_token_overlap:
            continue
        snippet = str(row["text"])
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        hits.append(
            CitationHit(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                section_ref=str(row["section_ref"]),
                bm25_score=round(abs(float(row["raw_bm25"])), 4),
                token_overlap=overlap,
                snippet=snippet,
                chunk_hash=str(row["chunk_hash"]),
            )
        )

    if not hits:
        return GroundedAnswer(
            query=query,
            status="INSUFFICIENT_EVIDENCE",
            answer_summary="No verified clauses in the indexed corpus meet the evidence threshold.",
            citations=[],
            verified=True,
        )

    hits.sort(key=lambda h: (h.token_overlap, h.bm25_score), reverse=True)
    hits = hits[:safe_top_k]
    verified = store.verify_citation_ids([h.chunk_id for h in hits])
    top_hit = hits[0]
    summary = (
        f"Grounded in [{top_hit.doc_id} · {top_hit.section_ref}]: {top_hit.snippet}"
        if verified
        else "Citation verification failed; output withheld."
    )

    return GroundedAnswer(
        query=query,
        status="GROUNDED" if verified else "INSUFFICIENT_EVIDENCE",
        answer_summary=summary,
        citations=hits if verified else [],
        verified=verified,
    )
