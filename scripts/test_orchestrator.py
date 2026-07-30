"""M15 verification: planner-orchestrator with A2A delegation.

Usage:
    python -m scripts.test_orchestrator
"""

from __future__ import annotations

from src.orchestrator.a2a import A2AResult, A2AStatus, A2ATask
from src.orchestrator.planner import PlannerOrchestrator


def test_a2a_envelopes() -> None:
    print("=" * 66)
    print(" A2A protocol: envelope construction")
    print("=" * 66)
    task = A2ATask(recipient="sql", query="count Microsoft CVEs")
    print(f"  A2ATask  id={task.task_id} -> {task.recipient}: {task.query!r}")
    result = A2AResult(task_id=task.task_id, sender="sql", status=A2AStatus.OK, answer="382")
    print(f"  A2AResult id={result.task_id} status={result.status.value} answer={result.answer!r}")
    assert result.task_id == task.task_id
    print("  OK\n")


def main() -> int:
    test_a2a_envelopes()

    planner = PlannerOrchestrator()

    print("=" * 66)
    print(" Single-source question (expect 1 task)")
    print("=" * 66)
    q1 = "How do I prevent SQL injection?"
    print(f"Q: {q1}")
    r1 = planner.run(q1)
    print(f"  num_tasks: {r1.metadata.get('num_tasks')}")
    print(f"  plan: {r1.metadata.get('plan')}")
    print(f"  A: {r1.answer[:180]}\n")

    print("=" * 66)
    print(" Multi-source question (expect 2 tasks, sql -> rag/api dependency)")
    print("=" * 66)
    q2 = "Which vendor has the most known exploited vulnerabilities, and what is that vendor known for security-wise?"
    print(f"Q: {q2}")
    r2 = planner.run(q2)
    print(f"  num_tasks: {r2.metadata.get('num_tasks')}")
    print(f"  agents_used: {r2.metadata.get('agents_used')}")
    for step in r2.metadata.get("plan", []):
        print(f"    - {step['agent']}: {step['query']}  (depends_on={step['depends_on']})")
    print(f"  A: {r2.answer[:300]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())