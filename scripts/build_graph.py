"""Build the security knowledge graph in Neo4j.

Usage:
    python -m scripts.build_graph
"""

from __future__ import annotations

from src.graph.builder import build_backbone


def main() -> int:
    print("Building knowledge-graph backbone from MITRE cross-references...")
    stats = build_backbone()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())