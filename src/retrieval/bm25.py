"""BM25 keyword search — in-memory index built by scrolling Qdrant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from src.config import settings

_TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """In-memory BM25 index over Qdrant chunk payloads."""

    bm25: BM25Okapi
    point_ids: list[str]
    payloads: list[dict[str, Any]]

    def search(self, query: str, k: int = 30) -> list[tuple[str, float, dict[str, Any]]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            (self.point_ids[i], float(scores[i]), self.payloads[i])
            for i in top_idx
            if scores[i] > 0
        ]


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    """Scroll all points from Qdrant once and build a cached BM25 index."""
    print("  building BM25 index from Qdrant...")
    client = QdrantClient(url=settings.qdrant_url)
    point_ids: list[str] = []
    payloads: list[dict[str, Any]] = []
    corpus_tokens: list[list[str]] = []

    next_offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in result:
            payload = pt.payload or {}
            point_ids.append(str(pt.id))
            payloads.append(payload)
            corpus_tokens.append(_tokenize(payload.get("text", "")))
        if next_offset is None:
            break

    if not corpus_tokens:
        raise RuntimeError("No documents in Qdrant — run scripts/ingest first.")

    bm25 = BM25Okapi(corpus_tokens)
    print(f"  BM25 index built ({len(corpus_tokens)} documents)")
    return BM25Index(bm25=bm25, point_ids=point_ids, payloads=payloads)