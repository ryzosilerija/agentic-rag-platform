"""M16 verification: compare base reranker vs LoRA-fine-tuned reranker.

Runs the SAME IR eval (hybrid + rerank) twice — stock vs LoRA adapter — on the
golden set and prints the nDCG/MRR/P@5 delta. The honest before/after number.

Usage:
    python -m scripts.eval_reranker_compare
"""

from __future__ import annotations

from pathlib import Path

from eval.metrics_ir import compute_ir_metrics
from eval.schemas import load_dataset
from src.retrieval.hybrid import RetrievalConfig, retrieve

GOLDEN = Path("eval/golden/dataset.jsonl")
ADAPTER = Path("models/reranker-lora")


def _dedupe(hits):
    seen, out = set(), []
    for _pid, _s, p in hits:
        sid = p.get("source_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


_LORA_SCORER = None


def _get_lora_scorer():
    """Load base model + LoRA adapter as a raw transformers scorer (bypasses
    sentence-transformers CrossEncoder, which conflicts with PEFT wrapping)."""
    global _LORA_SCORER
    if _LORA_SCORER is not None:
        return _LORA_SCORER
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    base = "BAAI/bge-reranker-base"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()

    def score(query, passages):
        enc = tok([query] * len(passages), passages, padding=True,
                  truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
        return logits.tolist() if logits.dim() > 0 else [logits.item()]

    _LORA_SCORER = score
    return score


def _rerank_lora(query, candidates, top_k=5):
    scorer = _get_lora_scorer()
    passages = [c[2].get("text", "") for c in candidates]
    scores = scorer(query, passages)
    ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
    return [(c[0], float(s), c[2]) for c, s in ranked[:top_k]]


def _eval_all(use_lora: bool) -> dict:

    dataset = [d for d in load_dataset(GOLDEN) if not d.requires_tool]
    cfg = RetrievalConfig(use_bm25=True, use_rerank=(not use_lora), rerank_top_k=5)
    P = R = M = N = 0.0
    for item in dataset:
        if use_lora:
            # get fused candidates WITHOUT rerank, then rerank with LoRA scorer
            base_cfg = RetrievalConfig(use_bm25=True, use_rerank=False, rerank_top_k=30)
            fused = retrieve(item.question, base_cfg)
            hits = _rerank_lora(item.question, fused, top_k=5)
        else:
            hits = retrieve(item.question, cfg)
        ids = _dedupe(hits)
        m = compute_ir_metrics(ids, item.expected_source_ids, k=5)
        P += m.precision_at_k; R += m.recall_at_k; M += m.mrr; N += m.ndcg_at_k
    n = len(dataset)
    return {"P@5": P/n, "R@5": R/n, "MRR": M/n, "nDCG@5": N/n}


def main() -> int:
    if not ADAPTER.exists():
        print(f"ERROR: {ADAPTER} not found. Run scripts.train_reranker_lora first.")
        return 1

    print("Evaluating BASE reranker...")
    base = _eval_all(use_lora=False)
    print("Evaluating LoRA reranker...")
    lora = _eval_all(use_lora=True)

    print("\n" + "=" * 60)
    print(" Base vs LoRA-fine-tuned reranker (hybrid + rerank)")
    print("=" * 60)
    print(f"  {'metric':<10} {'base':>8} {'lora':>8} {'delta':>8}")
    for k in ["P@5", "R@5", "MRR", "nDCG@5"]:
        d = lora[k] - base[k]
        print(f"  {k:<10} {base[k]:>8.3f} {lora[k]:>8.3f} {d:>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())