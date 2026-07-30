"""Ingest the MITRE ATT&CK Enterprise corpus into Qdrant.

Usage:
    python -m scripts.ingest_attack
    python -m scripts.ingest_attack --limit 20
"""

from __future__ import annotations

import argparse

from src.ingestion.attack_loader import load_attack_chunks
from src.ingestion.index import collection_stats, ensure_collection, upsert_chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Loading MITRE ATT&CK techniques (downloading ~30MB STIX on first run)...")
    chunks = load_attack_chunks(from_file=args.from_file, limit=args.limit)
    print(f"  parsed {len(chunks)} ATT&CK technique chunks")

    ensure_collection()
    n = upsert_chunks(chunks)
    print(f"  upserted {n} chunks")
    print("Collection stats:", collection_stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())