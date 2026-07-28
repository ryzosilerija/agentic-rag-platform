"""CI gate: read eval/results/ir_scorecard.json and enforce metric floors.

Exits non-zero (fails the build) if any floor is violated.

Usage:
    python -m scripts.eval_gate                       # default floors
    python -m scripts.eval_gate --config "hybrid (dense+BM25)" --ndcg 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCORECARD = Path("eval/results/ir_scorecard.json")

DEFAULT_FLOORS = {
    "mean_ndcg": 0.80,
    "mean_recall": 0.85,
    "mean_mrr": 0.75,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hybrid (dense+BM25)",
                    help="Which config's scores to gate on")
    ap.add_argument("--ndcg", type=float, default=DEFAULT_FLOORS["mean_ndcg"])
    ap.add_argument("--recall", type=float, default=DEFAULT_FLOORS["mean_recall"])
    ap.add_argument("--mrr", type=float, default=DEFAULT_FLOORS["mean_mrr"])
    args = ap.parse_args()

    if not SCORECARD.exists():
        print(f"GATE FAIL: {SCORECARD} not found - did eval_retrieval run?")
        return 1

    data = json.loads(SCORECARD.read_text(encoding="utf-8"))
    if args.config not in data:
        print(f"GATE FAIL: config {args.config!r} not in scorecard. Have: {list(data)}")
        return 1

    scores = data[args.config]
    floors = {"mean_ndcg": args.ndcg, "mean_recall": args.recall, "mean_mrr": args.mrr}

    print(f"Gating on config: {args.config}")
    failed = False
    for metric, floor in floors.items():
        val = scores.get(metric, 0.0)
        status = "OK  " if val >= floor else "FAIL"
        if val < floor:
            failed = True
        print(f"  [{status}] {metric}: {val:.3f} (floor {floor:.3f})")

    if failed:
        print("\nGATE FAIL: retrieval quality regressed below floor(s).")
        return 1
    print("\nGATE PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())