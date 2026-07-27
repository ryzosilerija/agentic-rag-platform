"""End-to-end ingestion: folder -> load -> chunk -> embed -> upsert to Qdrant.

Usage:
    python -m scripts.ingest                    # defaults to data/corpus
    python -m scripts.ingest path/to/folder
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.ingestion.chunking import chunk_document
from src.ingestion.index import collection_stats, upsert_chunks
from src.ingestion.loaders import load_folder


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/corpus")
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}")
        print("Hint: run  python -m scripts.download_corpus  first.")
        return 1

    print(f"[1/3] Loading documents from {folder}/")
    docs = load_folder(folder)
    print(f"      -> {len(docs)} documents")
    if not docs:
        print("Nothing to ingest.")
        return 0

    print("[2/3] Chunking...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        print(f"      {doc.source_id}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"      -> {len(all_chunks)} total chunks")

    print("[3/3] Embedding + upserting to Qdrant...")
    n = upsert_chunks(all_chunks)
    print(f"      -> upserted {n} points")

    print()
    stats = collection_stats()
    print(f"Collection stats: {stats}")
    print("Ingestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())