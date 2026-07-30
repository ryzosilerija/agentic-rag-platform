"""CWE corpus loader — MITRE Common Weakness Enumeration.

Fetches the CWE catalog (comprehensive XML from MITRE''s published ZIP) and
turns each weakness into a structured Chunk that flows through the existing
embed -> Qdrant pipeline (index.upsert_chunks). One chunk per weakness,
source_id "cwe-<id>", containing name, description, extended description,
common consequences, and mitigations.

Network: MITRE hosts CWE at cwe.mitre.org. If that host is not in the egress
allowlist, download the ZIP manually and use --from-file with the XML.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

from src.ingestion.chunking import Chunk

CWE_ZIP_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
_CACHE = Path("data/cache/cwec_latest.xml")


def _fetch_xml(from_file: str | None = None) -> bytes:
    if from_file:
        return Path(from_file).read_bytes()
    if _CACHE.exists():
        return _CACHE.read_bytes()
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(CWE_ZIP_URL, headers={"User-Agent": "agentic-rag-platform/0.1"})
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".xml")][0]
        xml_bytes = z.read(name)
    _CACHE.write_bytes(xml_bytes)
    return xml_bytes


def _text(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def load_cwe_chunks(from_file: str | None = None, limit: int | None = None) -> list[Chunk]:
    xml_bytes = _fetch_xml(from_file)
    root = ET.fromstring(xml_bytes)

    weaknesses = root.findall(".//{*}Weakness")

    chunks: list[Chunk] = []
    for i, w in enumerate(weaknesses):
        if limit and i >= limit:
            break
        cwe_id = w.get("ID", "")
        name = w.get("Name", "")
        abstraction = w.get("Abstraction", "")

        desc = _text(w.find("{*}Description"))
        ext = _text(w.find("{*}Extended_Description"))

        cons = []
        for c in w.findall(".//{*}Consequence"):
            scope = _text(c.find("{*}Scope"))
            impact = _text(c.find("{*}Impact"))
            if scope or impact:
                cons.append(f"{scope}: {impact}".strip(": "))

        mits = [_text(m.find("{*}Description")) for m in w.findall(".//{*}Mitigation")]
        mits = [m for m in mits if m]

        parts = [f"CWE-{cwe_id}: {name}"]
        if abstraction:
            parts.append(f"Abstraction: {abstraction}")
        if desc:
            parts.append(f"Description: {desc}")
        if ext:
            parts.append(f"Details: {ext}")
        if cons:
            parts.append("Common consequences: " + "; ".join(cons[:6]))
        if mits:
            parts.append("Mitigations: " + " ".join(mits[:4]))

        text = "\n".join(parts)
        source_id = f"cwe-{cwe_id}"
        chunks.append(
            Chunk(
                text=text,
                source_id=source_id,
                source_path="cwe:mitre",
                source_type="cwe",
                chunk_id=f"{source_id}#0",
                chunk_index=0,
                section=f"CWE-{cwe_id}: {name}",
                metadata={"corpus": "cwe", "cwe_id": cwe_id, "abstraction": abstraction},
            )
        )
    return chunks