"""M10 verification: routing accuracy + optional end-to-end runs.

Usage:
    python -m scripts.eval_routing
    python -m scripts.eval_routing --full
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.orchestrator.supervisor import Supervisor

GOLDEN = Path("eval/golden/routing.jsonl")
RESULTS = Path("eval/results/routing_scorecard.json")


def main() -> int:
    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} not found")
        return 1

    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Routing eval over {len(rows)} questions (classifier only, no agent runs).\n")

    sup = Supervisor()

    correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misroutes = []

    for r in rows:
        predicted = sup.classify(r["query"])
        expected = r["expected_route"]
        confusion[expected][predicted] += 1
        ok = predicted == expected
        correct += ok
        flag = "  " if ok else "XX"
        print(f"  [{flag}] {r['id']}  exp={expected}  got={predicted}   {r['query'][:50]}")
        if not ok:
            misroutes.append({"id": r["id"], "query": r["query"], "expected": expected, "got": predicted})

    n = len(rows)
    acc = correct / n
    print(f"\n  Routing accuracy: {correct}/{n} = {acc:.1%}")

    print("\n  Confusion matrix (rows=expected, cols=predicted):")
    labels = ["rag", "sql"]
    print("           " + "  ".join(f"{l:>5}" for l in labels))
    for exp_label in labels:
        cells = "  ".join(f"{confusion[exp_label][p]:>5}" for p in labels)
        print(f"    {exp_label:<6} {cells}")

    if misroutes:
        print("\n  Misrouted:")
        for m in misroutes:
            print(f"    {m['id']}: expected {m['expected']}, got {m['got']} — {m['query'][:60]}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {"n": n, "accuracy": acc,
             "confusion": {k: dict(v) for k, v in confusion.items()},
             "misroutes": misroutes},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Wrote {RESULTS}")

    if "--full" in sys.argv:
        print("\n" + "=" * 60)
        print(" End-to-end (router + agent):")
        print("=" * 60)
        for q in ["How do I prevent SQL injection?",
                  "How many Microsoft vulnerabilities are in the catalog?"]:
            resp = sup.run(q)
            print(f"\nQ: {q}")
            print(f"  routed_to: {resp.metadata.get('routed_to')}")
            print(f"  A: {resp.answer[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())