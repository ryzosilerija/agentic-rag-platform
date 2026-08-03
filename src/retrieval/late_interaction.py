"""Late-interaction (ColBERT-style MaxSim) reranking.

An A/B alternative to the cross-encoder reranker. Encodes the query and each
candidate chunk into token-level embeddings, then scores each candidate by
MaxSim: for every query token, take its max cosine similarity against any
document token, and sum. No PLAID index — scores the existing candidate pool
directly, so this is a controlled comparison against the cross-encoder on the
same pool.

Uses the general-domain colbertv2.0 checkpoint with no domain fine-tuning,
so this measures out-of-the-box late-interaction vs a purpose-built reranker.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "colbert-ir/colbertv2.0"
_MAX_Q_TOKENS = 32
_MAX_D_TOKENS = 180

_tokenizer = None
_model = None


def _load() -> None:
    global _tokenizer, _model
    if _model is None:
        print(f"  loading late-interaction model {_MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()


@torch.no_grad()
def _encode(texts: list[str], max_len: int) -> list[torch.Tensor]:
    """Return a list of (n_tokens, dim) L2-normalized token-embedding tensors."""
    enc = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    out = _model(**enc).last_hidden_state  # (batch, seq, dim)
    out = torch.nn.functional.normalize(out, p=2, dim=2)
    mask = enc["attention_mask"].bool()  # (batch, seq)
    # Keep only real (non-pad) token vectors per row.
    return [out[i][mask[i]] for i in range(out.size(0))]


@torch.no_grad()
def maxsim_rerank(
    query: str,
    pool: list[tuple[str, float, dict[str, Any]]],
    top_k: int = 5,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Rerank the candidate pool by ColBERT-style MaxSim late interaction."""
    _load()
    if not pool:
        return []

    q_tokens = _encode([query], _MAX_Q_TOKENS)[0]  # (nq, dim)

    doc_texts = [(p.get("text") or "") for _pid, _score, p in pool]
    scored: list[tuple[str, float, dict[str, Any]]] = []
    # Encode docs in small batches to keep CPU memory sane.
    batch = 8
    doc_vecs: list[torch.Tensor] = []
    for i in range(0, len(doc_texts), batch):
        doc_vecs.extend(_encode(doc_texts[i : i + batch], _MAX_D_TOKENS))

    for (pid, _old, payload), d_tokens in zip(pool, doc_vecs, strict=True):
        # sim: (nq, nd) → max over doc tokens → sum over query tokens
        sim = q_tokens @ d_tokens.T
        score = sim.max(dim=1).values.sum().item()
        scored.append((pid, float(score), payload))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]