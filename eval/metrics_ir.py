"""IR metrics — precision@k, recall@k, MRR, nDCG@k over source_ids."""

from __future__ import annotations

import math
from collections import defaultdict

from eval.schemas import IRMetrics, IRScorecard, QuestionResult


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k == 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in expected)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for e in expected if e in top_k)
    return hits / len(expected)


def mrr(retrieved: list[str], expected: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        rel = 1.0 if r in expected else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def compute_ir_metrics(retrieved: list[str], expected: list[str], k: int = 5) -> IRMetrics:
    expected_set = set(expected)
    return IRMetrics(
        precision_at_k=precision_at_k(retrieved, expected_set, k),
        recall_at_k=recall_at_k(retrieved, expected_set, k),
        mrr=mrr(retrieved, expected_set),
        ndcg_at_k=ndcg_at_k(retrieved, expected_set, k),
    )


def aggregate_ir_scores(results: list[QuestionResult], config_name: str) -> IRScorecard:
    doc_results = [r for r in results if not r.item.requires_tool]
    n = len(doc_results)
    if n == 0:
        return IRScorecard(config_name, 0, 0.0, 0.0, 0.0, 0.0)

    mean_p = sum(r.ir_metrics.precision_at_k for r in doc_results) / n
    mean_r = sum(r.ir_metrics.recall_at_k for r in doc_results) / n
    mean_mrr = sum(r.ir_metrics.mrr for r in doc_results) / n
    mean_ndcg = sum(r.ir_metrics.ndcg_at_k for r in doc_results) / n

    by_cat: dict[str, list[IRMetrics]] = defaultdict(list)
    for r in doc_results:
        by_cat[r.item.category].append(r.ir_metrics)

    per_cat: dict[str, dict[str, float]] = {}
    for cat, ms in by_cat.items():
        n_cat = len(ms)
        per_cat[cat] = {
            "n": float(n_cat),
            "precision": sum(m.precision_at_k for m in ms) / n_cat,
            "recall": sum(m.recall_at_k for m in ms) / n_cat,
            "mrr": sum(m.mrr for m in ms) / n_cat,
            "ndcg": sum(m.ndcg_at_k for m in ms) / n_cat,
        }

    return IRScorecard(
        config_name=config_name,
        n_questions=n,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_mrr=mean_mrr,
        mean_ndcg=mean_ndcg,
        per_category=per_cat,
    )