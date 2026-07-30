"""Ingest the CWE corpus into Qdrant.

Usage:
    python -m scripts.ingest_cwe
    python -m scripts.ingest_cwe --from-file data/cache/cwec_latest.xml
    python -m scripts.ingest_cwe --limit 50   # quick test
"""

from __future__ import annotations

import argparse

from src.ingestion.cwe_loader import load_cwe_chunks
from src.ingestion.index import collection_stats, ensure_collection, upsert_chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Loading CWE weaknesses...")
    chunks = load_cwe_chunks(from_file=args.from_file, limit=args.limit)
    print(f"  parsed {len(chunks)} CWE weakness chunks")

    ensure_collection()
    n = upsert_chunks(chunks)
    print(f"  upserted {n} chunks")
    print("Collection stats:", collection_stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())