"""Late-interaction A/B eval: fusion baseline vs cross-encoder vs ColBERT MaxSim.

Builds one candidate pool per golden question (dense+BM25+fusion, no rerank),
then ranks it three ways and scores each with the existing IR metrics.
Mirrors scripts/eval_retrieval.py's dedup + compute_ir_metrics usage exactly.

Usage:
    python -m scripts.eval_late_interaction
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from eval.metrics_ir import compute_ir_metrics
from eval.schemas import GoldenItem, load_dataset
from src.retrieval.hybrid import RetrievalConfig, retrieve
from src.retrieval.rerank import rerank as cross_encoder_rerank
from src.retrieval.late_interaction import maxsim_rerank

GOLDEN_PATH = Path("eval/golden/dataset.jsonl")
RESULTS_DIR = Path("eval/results")
TOP_K = 5


def _dedup_source_ids(hits: list[tuple[str, float, dict[str, Any]]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for _pid, _score, p in hits:
        sid = p.get("source_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found")
        return 1

    dataset = load_dataset(GOLDEN_PATH)
    doc_qs = [d for d in dataset if not d.requires_tool]
    print(f"Loaded {len(doc_qs)} doc-answerable golden questions.\n")

    # Pool config: full retrieval WITHOUT rerank; we rerank three ways below.
    pool_cfg = RetrievalConfig(use_bm25=True, use_rerank=False, rerank_top_k=30)

    configs = ["fusion baseline", "cross-encoder", "late-interaction"]
    acc: dict[str, dict[str, list[float]]] = {
        c: {"p": [], "r": [], "mrr": [], "ndcg": []} for c in configs
    }

    t0 = perf_counter()
    for item in doc_qs:
        pool = retrieve(item.question, pool_cfg)

        ranked = {
            "fusion baseline": sorted(pool, key=lambda x: -x[1])[:TOP_K],
            "cross-encoder": cross_encoder_rerank(item.question, pool, top_k=TOP_K),
            "late-interaction": maxsim_rerank(item.question, pool, top_k=TOP_K),
        }

        for cfg_name, hits in ranked.items():
            sids = _dedup_source_ids(hits)
            m = compute_ir_metrics(sids, item.expected_source_ids, k=TOP_K)
            acc[cfg_name]["p"].append(m.precision_at_k)
            acc[cfg_name]["r"].append(m.recall_at_k)
            acc[cfg_name]["mrr"].append(m.mrr)
            acc[cfg_name]["ndcg"].append(m.ndcg_at_k)

    elapsed = perf_counter() - t0

    print(f"{'config':<20} {'P@5':>7} {'R@5':>7} {'MRR':>7} {'nDCG@5':>7}")
    print("-" * 52)
    scorecard: dict[str, dict[str, float]] = {}
    for c in configs:
        p, r = _mean(acc[c]["p"]), _mean(acc[c]["r"])
        mr, nd = _mean(acc[c]["mrr"]), _mean(acc[c]["ndcg"])
        scorecard[c] = {"precision": p, "recall": r, "mrr": mr, "ndcg": nd}
        print(f"{c:<20} {p:>7.3f} {r:>7.3f} {mr:>7.3f} {nd:>7.3f}")

    print(f"\nEval ran in {elapsed:.0f}s over {len(doc_qs)} questions.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "late_interaction_scorecard.json"
    out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(f"Scorecard saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())