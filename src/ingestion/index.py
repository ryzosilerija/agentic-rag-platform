"""Qdrant collection setup and batch upsert."""

from __future__ import annotations

import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings
from src.ingestion.chunking import Chunk
from src.llm.embeddings import embed_texts, get_embedding_dim


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    """Create the dense-vector collection if it doesn't exist."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=get_embedding_dim(), distance=Distance.COSINE),
    )
    print(f"  created Qdrant collection: {settings.qdrant_collection}")


def _stable_point_id(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id (idempotent upsert)."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def upsert_chunks(chunks: list[Chunk], batch_size: int = 20) -> int:
    """Embed and upsert chunks in batches. Returns count upserted."""
    if not chunks:
        return 0
    ensure_collection()
    client = get_qdrant_client()
    total = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  batch {batch_num}/{n_batches}: embedding {len(batch)} chunks...")
        vectors = embed_texts([c.text for c in batch], batch_size=len(batch))
        points = [
            PointStruct(
                id=_stable_point_id(c.chunk_id),
                vector=v,
                payload={
                    "text": c.text,
                    "chunk_id": c.chunk_id,
                    "source_id": c.source_id,
                    "source_path": c.source_path,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    **c.metadata,
                },
            )
            for c, v in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(batch)
    return total


def collection_stats() -> dict:
    """Basic stats about the current collection."""
    client = get_qdrant_client()
    info = client.get_collection(settings.qdrant_collection)
    count = client.count(collection_name=settings.qdrant_collection, exact=True).count
    return {
        "points": count,
        "status": str(info.status),
        "indexed": getattr(info, "indexed_vectors_count", None) or 0,
    }