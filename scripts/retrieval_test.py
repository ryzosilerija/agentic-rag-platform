"""M2 verification: compare retrieval strategies on the same queries.

Runs each query through three progressively-richer pipelines:
  - dense only
  - dense + BM25 (RRF fusion)
  - dense + BM25 + rerank (full pipeline)

Prints top-3 for each so you can eyeball the quality delta.

Usage:
    python -m scripts.retrieval_test
    python -m scripts.retrieval_test "your custom query"
"""

from __future__ import annotations

import sys
import time

from src.retrieval.hybrid import RetrievalConfig, retrieve

DEFAULT_QUERIES = [
    "How can I prevent SQL injection in a web application?",
    "What is broken access control and how do I mitigate it?",
    "Cross-site scripting prevention with content security policy",
    "Password storage best practices with bcrypt",
    "NIST CSF Govern function",
]

CONFIGS = {
    "dense only        ": RetrievalConfig(use_bm25=False, use_rerank=False, rerank_top_k=3),
    "dense + BM25 (RRF)": RetrievalConfig(use_bm25=True,  use_rerank=False, rerank_top_k=3),
    "full pipeline     ": RetrievalConfig(use_bm25=True,  use_rerank=True,  rerank_top_k=3),
}


def _print_hits(hits, indent: int = 6) -> None:
    for i, (_pid, score, payload) in enumerate(hits, 1):
        src = payload.get("source_id", "?")
        section = (payload.get("section") or "-")[:50]
        snippet = (payload.get("text", "") or "").replace("\n", " ")[:140]
        print(f"{' ' * indent}[{i}] score={score:.3f}  {src}  ({section})")
        print(f"{' ' * (indent + 4)}{snippet}...")


def main() -> int:
    queries = sys.argv[1:] or DEFAULT_QUERIES
    for q in queries:
        print(f"\n{'=' * 78}")
        print(f"Q: {q}")
        print("=" * 78)
        for label, cfg in CONFIGS.items():
            t0 = time.perf_counter()
            hits = retrieve(q, cfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"\n  [{label}]  {elapsed_ms:.0f} ms")
            _print_hits(hits)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())