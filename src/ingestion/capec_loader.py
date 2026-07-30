"""CAPEC corpus loader — MITRE Common Attack Pattern Enumeration.

Same approach as cwe_loader: fetch the MITRE CAPEC XML, turn each attack
pattern into a Chunk (source_id "capec-<id>") with name, description,
likelihood/severity, prerequisites, and mitigations.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

from src.ingestion.chunking import Chunk

CAPEC_XML_URL = "https://capec.mitre.org/data/xml/capec_latest.xml"
_CACHE = Path("data/cache/capec_latest.xml")


def _fetch_xml(from_file: str | None = None) -> bytes:
    if from_file:
        return Path(from_file).read_bytes()
    if _CACHE.exists():
        return _CACHE.read_bytes()
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(CAPEC_XML_URL, headers={"User-Agent": "agentic-rag-platform/0.1"})
        resp.raise_for_status()
    _CACHE.write_bytes(resp.content)
    return resp.content


def _text(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def load_capec_chunks(from_file: str | None = None, limit: int | None = None) -> list[Chunk]:
    xml_bytes = _fetch_xml(from_file)
    root = ET.fromstring(xml_bytes)

    patterns = root.findall(".//{*}Attack_Pattern")

    chunks: list[Chunk] = []
    for i, p in enumerate(patterns):
        if limit and i >= limit:
            break
        cid = p.get("ID", "")
        name = p.get("Name", "")
        status = p.get("Status", "")

        desc = _text(p.find("{*}Description"))
        likelihood = _text(p.find("{*}Likelihood_Of_Attack"))
        severity = _text(p.find("{*}Typical_Severity"))

        prereqs = [_text(x) for x in p.findall(".//{*}Prerequisite")]
        prereqs = [x for x in prereqs if x]

        mits = [_text(m) for m in p.findall(".//{*}Mitigation")]
        mits = [m for m in mits if m]

        parts = [f"CAPEC-{cid}: {name}"]
        if desc:
            parts.append(f"Description: {desc}")
        if likelihood:
            parts.append(f"Likelihood of attack: {likelihood}")
        if severity:
            parts.append(f"Typical severity: {severity}")
        if prereqs:
            parts.append("Prerequisites: " + "; ".join(prereqs[:4]))
        if mits:
            parts.append("Mitigations: " + " ".join(mits[:4]))

        text = "\n".join(parts)
        source_id = f"capec-{cid}"
        chunks.append(
            Chunk(
                text=text,
                source_id=source_id,
                source_path="capec:mitre",
                source_type="capec",
                chunk_id=f"{source_id}#0",
                chunk_index=0,
                section=f"CAPEC-{cid}: {name}",
                metadata={"corpus": "capec", "capec_id": cid, "status": status},
            )
        )
    return chunks