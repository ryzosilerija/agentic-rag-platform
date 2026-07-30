"""Cross-encoder reranker (BGE bge-reranker-base by default).

Cross-encoders score (query, doc) pairs directly instead of comparing
independently-encoded vectors, so they're much more accurate at ranking
than bi-encoders — but ~100x slower per pair. Only ever call on the top-N
candidates from cheap retrieval.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# BAAI/bge-reranker-base: 278M params, ~1.1GB download, fast on CPU.
# Upgrade path when disk permits: BAAI/bge-reranker-v2-m3 (568M, ~2.3GB) —
# multilingual and slightly higher NDCG on long-doc retrieval.
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

# --- M16: LoRA adapter toggle ---------------------------------------------
# Lets eval swap between the stock reranker and a LoRA-fine-tuned variant.
# The adapter attaches to the CrossEncoder's underlying HF model (.model).
_LORA_PATH: str | None = None


def set_lora_adapter(path: str | None) -> None:
    """Toggle the LoRA adapter used by the reranker.

    path=None  -> use the stock base reranker.
    path=<dir> -> load the LoRA adapter from that directory onto the base model.
    Clears the cached model so the next rerank() rebuilds with the new setting.
    """
    global _LORA_PATH
    _LORA_PATH = path
    _get_reranker.cache_clear()


def _apply_lora_if_set(cross_encoder: Any) -> Any:
    """If a LoRA path is set, wrap the CrossEncoder's inner model with it."""
    if not _LORA_PATH:
        return cross_encoder
    from peft import PeftModel
    print(f"  loading LoRA adapter from {_LORA_PATH}...")
    cross_encoder.model = PeftModel.from_pretrained(cross_encoder.model, _LORA_PATH)
    cross_encoder.model.eval()
    return cross_encoder


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    """Lazy-load the cross-encoder (downloads on first call, ~1.1GB)."""
    from sentence_transformers import CrossEncoder

    print(f"  loading reranker {DEFAULT_RERANKER} (first call downloads ~1.1GB)...")
    ce = CrossEncoder(DEFAULT_RERANKER)
    return _apply_lora_if_set(ce)


def rerank(
    query: str,
    candidates: list[tuple[str, float, dict[str, Any]]],
    top_k: int = 5,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Rerank candidates with a cross-encoder; return top_k by rerank score."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c[2].get("text", "")) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: -float(x[1]),
    )
    return [(cand[0], float(score), cand[2]) for cand, score in ranked[:top_k]]