"""Hybrid retrieval pipeline: dense + BM25 -> RRF fusion -> cross-encoder rerank.

Stages are togglable via RetrievalConfig so we can measure the incremental
lift from each — critical for the M6/M7 eval dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.bm25 import get_bm25_index
from src.retrieval.dense import dense_search
from src.retrieval.fusion import rrf_fuse
from src.retrieval.rerank import rerank


@dataclass
class RetrievalConfig:
    dense_k: int = 30
    bm25_k: int = 30
    fusion_k: int = 30
    rerank_top_k: int = 5
    use_bm25: bool = True
    use_rerank: bool = True


def retrieve(
    query: str,
    config: RetrievalConfig | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Run the full hybrid retrieval pipeline for a query."""
    cfg = config or RetrievalConfig()

    dense_hits = dense_search(query, k=cfg.dense_k)

    if cfg.use_bm25:
        bm25_hits = get_bm25_index().search(query, k=cfg.bm25_k)
        fused = rrf_fuse([dense_hits, bm25_hits], k=cfg.fusion_k)
    else:
        fused = dense_hits[: cfg.fusion_k]

    if cfg.use_rerank:
        return rerank(query, fused, top_k=cfg.rerank_top_k)
    return fused[: cfg.rerank_top_k]