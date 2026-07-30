"""MITRE ATT&CK corpus loader — Enterprise techniques via STIX JSON.

Fetches the Enterprise ATT&CK STIX bundle from MITRE''s GitHub and turns each
technique (attack-pattern object) into a Chunk (source_id "attack-<Txxxx>")
with name, description, tactics, platforms, and detection guidance.

Network: raw.githubusercontent.com (already in the egress allowlist).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.ingestion.chunking import Chunk

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
_CACHE = Path("data/cache/enterprise-attack.json")


def _fetch(from_file: str | None = None) -> dict:
    if from_file:
        return json.loads(Path(from_file).read_text(encoding="utf-8"))
    if _CACHE.exists():
        return json.loads(_CACHE.read_text(encoding="utf-8"))
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(ATTACK_URL, headers={"User-Agent": "agentic-rag-platform/0.1"})
        resp.raise_for_status()
    _CACHE.write_bytes(resp.content)
    return json.loads(resp.content)


def _attack_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def load_attack_chunks(from_file: str | None = None, limit: int | None = None) -> list[Chunk]:
    bundle = _fetch(from_file)
    objects = bundle.get("objects", [])

    chunks: list[Chunk] = []
    count = 0
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if limit and count >= limit:
            break

        tid = _attack_id(obj)
        if not tid:
            continue
        name = obj.get("name", "")
        desc = " ".join((obj.get("description", "") or "").split())

        tactics = [
            ph.get("phase_name", "")
            for ph in obj.get("kill_chain_phases", [])
            if ph.get("kill_chain_name") == "mitre-attack"
        ]
        platforms = obj.get("x_mitre_platforms", []) or []
        detection = " ".join((obj.get("x_mitre_detection", "") or "").split())

        parts = [f"ATT&CK {tid}: {name}"]
        if tactics:
            parts.append(f"Tactics: {', '.join(tactics)}")
        if platforms:
            parts.append(f"Platforms: {', '.join(platforms)}")
        if desc:
            parts.append(f"Description: {desc[:1200]}")
        if detection:
            parts.append(f"Detection: {detection[:600]}")

        text = "\n".join(parts)
        source_id = f"attack-{tid}"
        chunks.append(
            Chunk(
                text=text,
                source_id=source_id,
                source_path="attack:mitre",
                source_type="attack",
                chunk_id=f"{source_id}#0",
                chunk_index=0,
                section=f"ATT&CK {tid}: {name}",
                metadata={"corpus": "attack", "attack_id": tid, "tactics": ",".join(tactics)},
            )
        )
        count += 1
    return chunks