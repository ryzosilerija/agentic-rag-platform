"""Dense vector search via Qdrant."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.config import settings
from src.llm.embeddings import embed_texts


def dense_search(query: str, k: int = 30) -> list[tuple[str, float, dict[str, Any]]]:
    """Return (point_id, score, payload) tuples for the top-k dense matches."""
    client = QdrantClient(url=settings.qdrant_url)
    qvec = embed_texts([query])[0]
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=qvec,
        limit=k,
        with_payload=True,
    ).points
    return [(str(h.id), float(h.score), h.payload or {}) for h in hits]