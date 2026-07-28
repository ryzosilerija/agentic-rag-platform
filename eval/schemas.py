"""Data models for the evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    expected_answer_snippet: str
    expected_source_ids: list[str]
    requires_tool: Optional[str] = None


def load_dataset(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        items.append(GoldenItem(**d))
    return items


@dataclass
class IRMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass
class IRScorecard:
    config_name: str
    n_questions: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class QuestionResult:
    item: GoldenItem
    retrieved_source_ids: list[str]
    ir_metrics: IRMetrics
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)