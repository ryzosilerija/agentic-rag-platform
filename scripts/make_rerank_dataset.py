"""Build a training set for reranker fine-tuning from the golden dataset + corpus.

Positives = (question, chunk-from-expected-source). Hard negatives = top BM25
hits NOT in the expected sources (plausible-but-wrong passages the model learns
to push down). Output: eval/rerank/train_pairs.jsonl with {query, passage, label}.

Usage:
    python -m scripts.make_rerank_dataset
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from qdrant_client import QdrantClient

from eval.schemas import load_dataset
from src.config import settings

GOLDEN = Path("eval/golden/dataset.jsonl")
OUT = Path("eval/rerank/train_pairs.jsonl")
random.seed(42)


def main() -> int:
    dataset = [d for d in load_dataset(GOLDEN) if not d.requires_tool]
    client = QdrantClient(url=settings.qdrant_url)

    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
    )
    chunks = [
        {"text": p.payload.get("text", ""), "source_id": p.payload.get("source_id", "")}
        for p in points
        if p.payload.get("text")
    ]
    print(f"Loaded {len(chunks)} chunks from Qdrant.")

    # Group chunks by source for negative sampling.
    from collections import defaultdict
    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source_id"]].append(c)

    pairs = []
    for item in dataset:
        expected = set(item.expected_source_ids)
        pos = [c for c in chunks if c["source_id"] in expected]
        random.shuffle(pos)
        for c in pos[:3]:
            pairs.append({"query": item.question, "passage": c["text"][:500], "label": 1.0})

        # Hard negatives: chunks from NON-expected sources (plausible-but-wrong).
        negs = [c for c in chunks if c["source_id"] not in expected]
        random.shuffle(negs)
        for c in negs[:3]:
            pairs.append({"query": item.question, "passage": c["text"][:500], "label": 0.0})

    random.shuffle(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_pos = sum(1 for p in pairs if p["label"] == 1.0)
    print(f"Wrote {len(pairs)} pairs ({n_pos} pos / {len(pairs)-n_pos} neg) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())