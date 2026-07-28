"""Retrieval evaluation CLI ā€” measures IR quality across 3 configs.

Usage:
    python -m scripts.eval_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from eval.metrics_ir import aggregate_ir_scores, compute_ir_metrics
from eval.schemas import GoldenItem, QuestionResult, load_dataset
from src.retrieval.hybrid import RetrievalConfig, retrieve

GOLDEN_PATH = Path("eval/golden/dataset.jsonl")
RESULTS_DIR = Path("eval/results")

CONFIGS: dict[str, RetrievalConfig] = {
    "dense only          ": RetrievalConfig(use_bm25=False, use_rerank=False, rerank_top_k=5),
    "hybrid (dense+BM25) ": RetrievalConfig(use_bm25=True,  use_rerank=False, rerank_top_k=5),
    "hybrid + rerank     ": RetrievalConfig(use_bm25=True,  use_rerank=True,  rerank_top_k=5),
}


def retrieve_and_score(item: GoldenItem, config: RetrievalConfig) -> QuestionResult:
    hits = retrieve(item.question, config)
    # Dedupe source_ids (order-preserving): metrics operate at DOCUMENT level.
    # Without this, multiple chunks from the same doc inflate DCG past 1.0.
    seen: set[str] = set()
    source_ids: list[str] = []
    for _pid, _score, p in hits:
        sid = p.get("source_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            source_ids.append(sid)
    metrics = compute_ir_metrics(source_ids, item.expected_source_ids, k=5)
    return QuestionResult(item=item, retrieved_source_ids=source_ids, ir_metrics=metrics)


def main() -> int:
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found")
        return 1

    dataset = load_dataset(GOLDEN_PATH)
    doc_qs = [d for d in dataset if not d.requires_tool]
    print(f"Loaded {len(dataset)} golden questions "
          f"({len(doc_qs)} doc-answerable, {len(dataset) - len(doc_qs)} tool-only skipped for IR).\n")

    scorecards: dict[str, dict] = {}
    for label, cfg in CONFIGS.items():
        print(f"[{label.strip()}]")
        t0 = perf_counter()
        results = [retrieve_and_score(item, cfg) for item in doc_qs]
        elapsed = perf_counter() - t0
        sc = aggregate_ir_scores(results, label.strip())
        scorecards[label] = {"scorecard": sc, "results": results, "elapsed_s": elapsed}
        print(f"  mean P@5={sc.mean_precision:.3f}  R@5={sc.mean_recall:.3f}  "
              f"MRR={sc.mean_mrr:.3f}  nDCG@5={sc.mean_ndcg:.3f}  ({elapsed:.1f}s)\n")

    print("=" * 78)
    print(" IR Metrics Comparison  (higher = better)")
    print("=" * 78)
    print(f"  {'Config':<22} {'P@5':>8} {'R@5':>8} {'MRR':>8} {'nDCG@5':>8}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for label, entry in scorecards.items():
        sc = entry["scorecard"]
        print(f"  {label:<22} {sc.mean_precision:>8.3f} {sc.mean_recall:>8.3f} "
              f"{sc.mean_mrr:>8.3f} {sc.mean_ndcg:>8.3f}")
    print()

    keys = list(CONFIGS.keys())
    if len(keys) >= 3:
        no_rr = scorecards[keys[1]]["scorecard"]
        rr = scorecards[keys[2]]["scorecard"]
        dense = scorecards[keys[0]]["scorecard"]
        d_ndcg = rr.mean_ndcg - no_rr.mean_ndcg
        d_mrr = rr.mean_mrr - no_rr.mean_mrr
        d_prec = rr.mean_precision - no_rr.mean_precision
        d_ndcg_hy = no_rr.mean_ndcg - dense.mean_ndcg
        print(f"  [Rerank delta]   nDCG@5: {d_ndcg:+.3f}   MRR: {d_mrr:+.3f}   P@5: {d_prec:+.3f}")
        print(f"  [Hybrid delta]   nDCG@5: {d_ndcg_hy:+.3f}  (dense -> hybrid)")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ir_scorecard.json"
    out.write_text(
        json.dumps(
            {
                label.strip(): {
                    "n_questions": entry["scorecard"].n_questions,
                    "mean_precision": entry["scorecard"].mean_precision,
                    "mean_recall": entry["scorecard"].mean_recall,
                    "mean_mrr": entry["scorecard"].mean_mrr,
                    "mean_ndcg": entry["scorecard"].mean_ndcg,
                    "per_category": entry["scorecard"].per_category,
                    "elapsed_s": entry["elapsed_s"],
                }
                for label, entry in scorecards.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Wrote scorecard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())