"""M1 verification: run a few real queries and show the top matches.

Usage:
    python -m scripts.search_test
    python -m scripts.search_test "your custom query"
"""

from __future__ import annotations

import sys

from src.config import settings
from src.ingestion.index import collection_stats, get_qdrant_client
from src.llm.embeddings import embed_texts

DEFAULT_QUERIES = [
    "How can I prevent SQL injection in a web application?",
    "What is broken access control and how do I mitigate it?",
    "NIST cybersecurity framework functions",
    "Cross-site scripting prevention with content security policy",
    "Password storage best practices",
]


def search(query: str, top_k: int = 3) -> None:
    client = get_qdrant_client()
    qvec = embed_texts([query])[0]
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=qvec,
        limit=top_k,
        with_payload=True,
    ).points

    print(f"\nQuery: {query!r}")
    print("-" * 78)
    for i, h in enumerate(hits, 1):
        payload = h.payload or {}
        src = payload.get("source_id", "?")
        section = payload.get("section", "") or "-"
        snippet = (payload.get("text", "") or "").replace("\n", " ")[:180]
        print(f"[{i}] score={h.score:.3f}  {src}   ({section[:60]})")
        print(f"     {snippet}...")


def main() -> int:
    stats = collection_stats()
    print(f"Collection stats: {stats}")
    if stats["points"] == 0:
        print("Collection is empty. Run  python -m scripts.ingest  first.")
        return 1

    queries = sys.argv[1:] or DEFAULT_QUERIES
    for q in queries:
        search(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())