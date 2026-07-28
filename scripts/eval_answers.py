"""Answer-quality evaluation: run the FULL agent on the golden set, judge each answer.

Cost: per question = 1 agent run (2-3 LLM calls) + 1 judge call.
For 25 questions that is roughly 90-100 Gemini requests - run on a fresh
daily quota. Progress is saved incrementally so an interrupted run resumes.

Usage:
    python -m scripts.eval_answers            # full golden set
    python -m scripts.eval_answers q01 q02    # only specific question ids
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from eval.metrics_answer import judge_answer
from eval.schemas import load_dataset
from src.agents.rag_agent import RAGAgent

GOLDEN_PATH = Path("eval/golden/dataset.jsonl")
RESULTS_DIR = Path("eval/results")
OUT_PATH = RESULTS_DIR / "answer_scores.jsonl"


def _load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def main() -> int:
    dataset = load_dataset(GOLDEN_PATH)
    only_ids = set(sys.argv[1:])
    if only_ids:
        dataset = [d for d in dataset if d.id in only_ids]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids() if not only_ids else set()
    todo = [d for d in dataset if d.id not in done]
    print(f"{len(dataset)} questions, {len(done)} already scored, {len(todo)} to run.\n")
    if not todo:
        print("Nothing to do. Delete eval/results/answer_scores.jsonl to re-run.")
        _print_summary()
        return 0

    print("Initializing agent...")
    agent = RAGAgent()

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, item in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {item.id}: {item.question[:60]}...")
            t0 = time.perf_counter()
            try:
                resp = agent.run(item.question)
            except Exception as e:
                print(f"    ! agent failed: {e}")
                continue
            agent_s = time.perf_counter() - t0

            context = "\n\n".join(
                f"[{n}] {c.source_id}: {c.text}" for n, c in enumerate(resp.citations, 1)
            )
            try:
                scores = judge_answer(
                    item.question, context, resp.answer, item.expected_answer_snippet,
                )
            except Exception as e:
                print(f"    ! judge failed: {e}")
                continue

            row = {
                "id": item.id,
                "category": item.category,
                "faithfulness": scores.faithfulness,
                "correctness": scores.correctness,
                "reasoning": scores.reasoning,
                "tools_called": resp.metadata.get("tools_called", []),
                "agent_seconds": round(agent_s, 1),
                "answer": resp.answer[:500],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    faith={scores.faithfulness:.2f}  correct={scores.correctness:.2f}  ({agent_s:.0f}s)")

    _print_summary()
    return 0


def _print_summary() -> None:
    if not OUT_PATH.exists():
        return
    rows = [json.loads(l) for l in OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    n = len(rows)
    mean_f = sum(r["faithfulness"] for r in rows) / n
    mean_c = sum(r["correctness"] for r in rows) / n
    halluc = sum(1 for r in rows if r["faithfulness"] < 0.7)
    print()
    print("=" * 66)
    print(f" Answer Quality Summary  (n={n})")
    print("=" * 66)
    print(f"  mean faithfulness: {mean_f:.3f}")
    print(f"  mean correctness:  {mean_c:.3f}")
    print(f"  hallucination rate (faith < 0.7): {halluc}/{n} = {halluc / n:.1%}")
    low = [r for r in rows if r["correctness"] < 0.5 or r["faithfulness"] < 0.7]
    if low:
        print("\n  Flagged questions:")
        for r in low:
            print(f"    {r['id']}  faith={r['faithfulness']:.2f} correct={r['correctness']:.2f}  {r['reasoning'][:70]}")