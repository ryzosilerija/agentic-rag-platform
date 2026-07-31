"""M17 verification: the payoff query — multi-hop traversal vector search can''t do.

Usage:
    python -m scripts.test_graph
"""

from __future__ import annotations

from src.graph.retrieval import (
    attack_chain_for_cwe,
    find_cwe_by_keyword,
    graph_stats,
    related_source_ids,
)


def main() -> int:
    print("Graph node counts:", graph_stats())

    print("\n" + "=" * 68)
    print(" Seed: find CWE nodes for 'SQL injection'")
    print("=" * 68)
    seeds = find_cwe_by_keyword("SQL injection")
    for s in seeds:
        print(f"  {s['id']}: {s['name']}")

    if not seeds:
        print("  (no seed found)")
        return 1

    cwe_id = next((s["id"] for s in seeds if s["id"] == "CWE-89"), seeds[0]["id"])
    print("\n" + "=" * 68)
    print(f" PAYOFF QUERY: {cwe_id} -> CAPEC attack patterns -> ATT&CK techniques")
    print(" (multi-hop reasoning pure vector search cannot do)")
    print("=" * 68)
    chain = attack_chain_for_cwe(cwe_id)
    seen = set()
    for row in chain:
        key = (row.get("capec"), row.get("technique"))
        if key in seen:
            continue
        seen.add(key)
        capec = f"{row.get('capec')}: {row.get('capec_name')}"
        tech = row.get("technique")
        if tech:
            print(f"  {capec}  ->  {tech}: {row.get('technique_name')}")
        else:
            print(f"  {capec}")

    print("\n" + "=" * 68)
    print(f" Blendable source_ids reachable from {cwe_id} (for hybrid+graph retrieval):")
    print("=" * 68)
    ids = related_source_ids(cwe_id)
    print("  " + ", ".join(ids[:15]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())