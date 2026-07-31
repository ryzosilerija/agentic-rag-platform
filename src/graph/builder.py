"""Build the security knowledge graph in Neo4j from cached MITRE files."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from src.graph.client import get_driver, run_query
from src.graph.schema import CONSTRAINTS

CWE_XML = Path("data/cache/cwec_latest.xml")
CAPEC_XML = Path("data/cache/capec_latest.xml")
ATTACK_JSON = Path("data/cache/enterprise-attack.json")


def _attack_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def _apply_constraints() -> None:
    for c in CONSTRAINTS:
        run_query(c)


def _build_cwe() -> tuple[int, int, int]:
    root = ET.parse(CWE_XML).getroot()
    nodes, exploited, related = [], [], []
    for w in root.findall(".//{*}Weakness"):
        cid = w.get("ID")
        nodes.append({"id": f"CWE-{cid}", "name": w.get("Name", ""), "source_id": f"cwe-{cid}"})
        for ap in w.findall(".//{*}Related_Attack_Pattern"):
            capec = ap.get("CAPEC_ID")
            if capec:
                exploited.append({"cwe": f"CWE-{cid}", "capec": f"CAPEC-{capec}"})
        for rw in w.findall(".//{*}Related_Weakness"):
            target = rw.get("CWE_ID")
            if target:
                related.append({"a": f"CWE-{cid}", "b": f"CWE-{target}"})

    with get_driver().session() as s:
        s.run("UNWIND $rows AS r MERGE (n:CWE {id: r.id}) SET n.name = r.name, n.source_id = r.source_id", rows=nodes)
        s.run("UNWIND $rows AS r MATCH (c:CWE {id: r.cwe}) MERGE (p:CAPEC {id: r.capec}) MERGE (c)-[:EXPLOITED_BY]->(p)", rows=exploited)
        s.run("UNWIND $rows AS r MATCH (a:CWE {id: r.a}) MATCH (b:CWE {id: r.b}) MERGE (a)-[:RELATED_TO]->(b)", rows=related)
    return len(nodes), len(exploited), len(related)


def _build_capec() -> tuple[int, int]:
    root = ET.parse(CAPEC_XML).getroot()
    nodes, maps = [], []
    for p in root.findall(".//{*}Attack_Pattern"):
        cid = p.get("ID")
        nodes.append({"id": f"CAPEC-{cid}", "name": p.get("Name", ""), "source_id": f"capec-{cid}"})
        for tm in p.findall(".//{*}Taxonomy_Mapping"):
            if (tm.get("Taxonomy_Name") or "").upper() == "ATTACK":
                entry = tm.find("{*}Entry_ID")
                if entry is not None and entry.text:
                    maps.append({"capec": f"CAPEC-{cid}", "tech": f"T{entry.text.strip()}"})

    with get_driver().session() as s:
        s.run("UNWIND $rows AS r MERGE (n:CAPEC {id: r.id}) SET n.name = r.name, n.source_id = r.source_id", rows=nodes)
        s.run("UNWIND $rows AS r MATCH (c:CAPEC {id: r.capec}) MERGE (t:Technique {id: r.tech}) MERGE (c)-[:MAPS_TO]->(t)", rows=maps)
    return len(nodes), len(maps)


def _build_attack() -> tuple[int, int]:
    bundle = json.loads(ATTACK_JSON.read_text(encoding="utf-8"))
    objects = bundle.get("objects", [])
    stix_to_attack, tech_nodes = {}, []
    for o in objects:
        if o.get("type") == "attack-pattern" and not o.get("revoked"):
            tid = _attack_id(o)
            if tid:
                stix_to_attack[o["id"]] = tid
                tech_nodes.append({"id": tid, "name": o.get("name", ""), "source_id": f"attack-{tid}"})
    subtech = []
    for o in objects:
        if o.get("type") == "relationship" and o.get("relationship_type") == "subtechnique-of":
            src = stix_to_attack.get(o.get("source_ref"))
            tgt = stix_to_attack.get(o.get("target_ref"))
            if src and tgt:
                subtech.append({"a": src, "b": tgt})

    with get_driver().session() as s:
        s.run("UNWIND $rows AS r MERGE (n:Technique {id: r.id}) SET n.name = r.name, n.source_id = r.source_id", rows=tech_nodes)
        s.run("UNWIND $rows AS r MATCH (a:Technique {id: r.a}) MATCH (b:Technique {id: r.b}) MERGE (a)-[:SUBTECHNIQUE_OF]->(b)", rows=subtech)
    return len(tech_nodes), len(subtech)


def build_backbone() -> dict:
    _apply_constraints()
    cwe_n, exploited, related = _build_cwe()
    capec_n, maps = _build_capec()
    tech_n, subtech = _build_attack()
    return {
        "cwe_nodes": cwe_n, "capec_nodes": capec_n, "technique_nodes": tech_n,
        "cwe_exploited_by_capec": exploited, "cwe_related_to_cwe": related,
        "capec_maps_to_technique": maps, "technique_subtechnique_edges": subtech,
    }