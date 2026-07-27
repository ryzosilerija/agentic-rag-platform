"""Reciprocal Rank Fusion (RRF) — merge multiple ranked lists into one.

Standard formula: score(d) = sum over rankings R of 1 / (k + rank_R(d)).
Doc-agnostic score aggregation — no tuning, no per-list weights.
"""

from __future__ import annotations

from typing import Any


def rrf_fuse(
    rankings: list[list[tuple[str, float, dict[str, Any]]]],
    k: int = 30,
    rrf_k: int = 60,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Fuse multiple ranked lists via RRF."""
    fused_scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, (pid, _score, payload) in enumerate(ranking):
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if pid not in payloads:
                payloads[pid] = payload

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: -fused_scores[x])[:k]
    return [(pid, fused_scores[pid], payloads[pid]) for pid in sorted_ids]