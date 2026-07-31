"""Knowledge-graph schema for the security taxonomies.

Nodes: CWE (weakness), CAPEC (attack pattern), Technique (ATT&CK), Concept (LLM).
Backbone rels (official MITRE cross-references):
  (CWE)-[:EXPLOITED_BY]->(CAPEC)   (CWE Related_Attack_Pattern)
  (CWE)-[:RELATED_TO]->(CWE)       (CWE Related_Weakness)
  (CAPEC)-[:MAPS_TO]->(Technique)  (CAPEC ATTACK taxonomy mapping)
  (Technique)-[:SUBTECHNIQUE_OF]->(Technique)  (ATT&CK STIX)
Every node carries source_id matching the Qdrant chunk payload.
"""

from __future__ import annotations

CONSTRAINTS = [
    "CREATE CONSTRAINT cwe_id IF NOT EXISTS FOR (n:CWE) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT capec_id IF NOT EXISTS FOR (n:CAPEC) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT tech_id IF NOT EXISTS FOR (n:Technique) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
]

NODE_LABELS = ["CWE", "CAPEC", "Technique", "Concept"]
REL_TYPES = [
    "EXPLOITED_BY", "RELATED_TO", "MAPS_TO", "SUBTECHNIQUE_OF",
    "MENTIONS", "RELATES_TO",
]