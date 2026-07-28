"""Look up CVE details via the NVD REST API (free, no auth required)."""

from __future__ import annotations

import re
from typing import Any

import httpx

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_TIMEOUT = 15.0


def lookup_cve(cve_id: str) -> dict[str, Any]:
    """Fetch structured CVE details from the NVD database."""
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.fullmatch(cve_id):
        return {"error": f"Invalid CVE ID format: {cve_id!r}. Expected 'CVE-YYYY-NNNNN'."}

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(
                _NVD_URL,
                params={"cveId": cve_id},
                headers={"User-Agent": "agentic-rag-platform/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"NVD API call failed: {e}"}

    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        return {"error": f"CVE {cve_id} not found in NVD."}

    cve = vulnerabilities[0]["cve"]
    desc = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
        "",
    )

    metrics = cve.get("metrics", {})
    severity = "unknown"
    score = None
    vector = ""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            cdata = entries[0].get("cvssData", {})
            severity = cdata.get("baseSeverity") or entries[0].get("baseSeverity", "unknown")
            score = cdata.get("baseScore")
            vector = cdata.get("vectorString", "")
            break

    cwes: list[str] = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            if d.get("lang") == "en":
                cwes.append(d.get("value", ""))

    return {
        "cve_id": cve_id,
        "description": desc,
        "severity": severity,
        "cvss_score": score,
        "cvss_vector": vector,
        "cwes": cwes[:3],
        "published_date": cve.get("published", ""),
        "last_modified": cve.get("lastModified", ""),
        "references": [r["url"] for r in cve.get("references", [])[:5]],
    }