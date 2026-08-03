"""Parent document retrieval.

Retrieval and reranking operate on small, precise chunks. But small chunks
often lack the surrounding context an LLM needs to answer confidently
(a chunk may match a query yet be too narrow to answer from). This module
expands each reranked hit to its full parent section — all sibling chunks
sharing the same source_id + section — before synthesis.

Retrieval scoring is untouched (nDCG/MRR unchanged by construction); only
the text handed to the LLM grows. Matched-chunk page numbers are preserved
for citations, and hits from the same parent section are de-duplicated so
the same section is never sent twice.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import settings

_MAX_PARENT_CHARS = 4000


def _fetch_section_chunks(
    client: QdrantClient, source_id: str, section: str
) -> list[dict[str, Any]]:
    """All chunks for one source_id + section, ordered by chunk_index."""
    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id)),
            qm.FieldCondition(key="section", match=qm.MatchValue(value=section)),
        ]
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=flt,
        with_payload=True,
        limit=200,
    )
    payloads = [p.payload or {} for p in points]
    payloads.sort(key=lambda p: p.get("chunk_index", 0))
    return payloads


def expand_to_parents(
    retrieved: list[tuple[str, float, dict[str, Any]]],
    max_parent_chars: int = _MAX_PARENT_CHARS,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Swap each reranked hit's text for its full parent-section text.

    De-duplicates by (source_id, section): the highest-scoring hit for a
    section survives, carrying the concatenated section text. Hits with no
    section (empty string — e.g. taxonomy chunks) pass through unchanged.
    """
    client = QdrantClient(url=settings.qdrant_url)
    out: list[tuple[str, float, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()

    for pid, score, payload in retrieved:
        source_id = payload.get("source_id", "")
        section = payload.get("section") or ""

        # No section to expand into — keep the chunk as-is.
        if not section:
            out.append((pid, score, payload))
            continue

        key = (source_id, section)
        if key in seen:
            continue  # already emitted this parent via a higher-scoring hit
        seen.add(key)

        siblings = _fetch_section_chunks(client, source_id, section)
        if not siblings:
            out.append((pid, score, payload))
            continue

        parent_text = "\n\n".join(s.get("text", "") for s in siblings if s.get("text"))
        if len(parent_text) > max_parent_chars:
            parent_text = parent_text[:max_parent_chars] + "\n…(section truncated)"

        # Keep the matched chunk's payload (source_id, section, page, chunk_id
        # all preserved for citations) but replace text with the parent section.
        new_payload = dict(payload)
        new_payload["text"] = parent_text
        out.append((pid, score, new_payload))

    return out