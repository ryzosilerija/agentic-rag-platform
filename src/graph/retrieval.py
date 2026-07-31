"""Graph retrieval — multi-hop traversal over the security taxonomy graph.

Answers relationship questions that vector search cannot: given a weakness,
find the attack patterns that exploit it and the ATT&CK techniques those map
to (CWE -> CAPEC -> Technique). Returns source_ids that resolve back to Qdrant
chunks, so graph hits can be blended into the retrieval pipeline.
"""

from __future__ import annotations

from src.graph.client import run_query


def attack_chain_for_cwe(cwe_id: str, limit: int = 25) -> list[dict]:
    """CWE -> CAPEC -> Technique traversal for a given weakness id (e.g. 'CWE-89')."""
    cypher = """
    MATCH (w:CWE {id: $cwe_id})-[:EXPLOITED_BY]->(c:CAPEC)
    OPTIONAL MATCH (c)-[:MAPS_TO]->(t:Technique)
    RETURN w.id AS cwe, w.name AS cwe_name,
           c.id AS capec, c.name AS capec_name,
           t.id AS technique, t.name AS technique_name
    LIMIT $limit
    """
    return run_query(cypher, cwe_id=cwe_id, limit=limit)


def related_source_ids(cwe_id: str, hops: int = 2, limit: int = 30) -> list[str]:
    """Return Qdrant source_ids reachable from a CWE within `hops` (for blending)."""
    cypher = f"""
    MATCH (w:CWE {{id: $cwe_id}})-[*1..{hops}]-(n)
    WHERE n.source_id IS NOT NULL
    RETURN DISTINCT n.source_id AS source_id
    LIMIT $limit
    """
    rows = run_query(cypher, cwe_id=cwe_id, limit=limit)
    return [r["source_id"] for r in rows if r.get("source_id")]


def find_cwe_by_keyword(keyword: str, limit: int = 5) -> list[dict]:
    """Find CWE nodes whose name matches a keyword (seed for traversal)."""
    cypher = """
    MATCH (w:CWE)
    WHERE toLower(w.name) CONTAINS toLower($kw)
    RETURN w.id AS id, w.name AS name
    LIMIT $limit
    """
    return run_query(cypher, kw=keyword, limit=limit)


def graph_stats() -> dict:
    rows = run_query("""
    MATCH (n) WITH labels(n)[0] AS label, count(*) AS c
    RETURN label, c ORDER BY c DESC
    """)
    return {r["label"]: r["c"] for r in rows}