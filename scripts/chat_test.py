"""M3 verification: end-to-end chat with the RAG agent (no server needed).

Usage:
    python -m scripts.chat_test
    python -m scripts.chat_test "your custom question"
"""

from __future__ import annotations

import sys

from src.agents.rag_agent import RAGAgent

DEFAULT_QUERIES = [
    "How can I prevent SQL injection in a web application?",
    "What is broken access control and how do I mitigate it?",
    "What password hashing algorithm does NIST recommend?",
]


def main() -> int:
    print("Initializing agent (loads reranker + BM25 index)...")
    agent = RAGAgent()
    queries = sys.argv[1:] or DEFAULT_QUERIES

    for q in queries:
        print(f"\n{'=' * 78}")
        print(f"Q: {q}")
        print("=" * 78)
        resp = agent.run(q)
        print(f"\n{resp.answer}")
        print("\n--- Citations ---")
        for i, c in enumerate(resp.citations, 1):
            section = (c.section or "-")[:60]
            print(f"[{i}] {c.source_id}  ({section})")
        rewritten = resp.metadata.get("rewritten_query", "")
        if rewritten and rewritten != q:
            print(f"\n(rewritten query used: {rewritten!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())