"""Search CISA's Known Exploited Vulnerabilities (KEV) catalog.

CISA publishes a live CSV of vulnerabilities being actively exploited.
We cache the whole catalog for 24h and search it in-memory.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any

import httpx

_KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
_CACHE_PATH = Path("data/cache/kev.csv")
_CACHE_TTL_SEC = 24 * 3600


def _fetch_kev() -> list[dict[str, str]]:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    if _CACHE_PATH.exists() and (now - _CACHE_PATH.stat().st_mtime) < _CACHE_TTL_SEC:
        text = _CACHE_PATH.read_text(encoding="utf-8")
    else:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(_KEV_URL, headers={"User-Agent": "agentic-rag-platform/0.1"})
            resp.raise_for_status()
            text = resp.text
        _CACHE_PATH.write_text(text, encoding="utf-8")

    return list(csv.DictReader(io.StringIO(text)))


def search_kev(
    vendor: str = "",
    product: str = "",
    keyword: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    """Search CISA's Known Exploited Vulnerabilities catalog."""
    try:
        entries = _fetch_kev()
    except Exception as e:
        return {"error": f"CISA KEV fetch failed: {e}"}

    v_lc = vendor.lower().strip()
    p_lc = product.lower().strip()
    k_lc = keyword.lower().strip()

    matches: list[dict[str, str]] = []
    for row in entries:
        if v_lc and v_lc not in row.get("vendorProject", "").lower():
            continue
        if p_lc and p_lc not in row.get("product", "").lower():
            continue
        if k_lc:
            haystack = " ".join([
                row.get("vulnerabilityName", ""),
                row.get("shortDescription", ""),
                row.get("cveID", ""),
            ]).lower()
            if k_lc not in haystack:
                continue
        matches.append({
            "cve_id": row.get("cveID", ""),
            "vendor": row.get("vendorProject", ""),
            "product": row.get("product", ""),
            "name": row.get("vulnerabilityName", ""),
            "date_added": row.get("dateAdded", ""),
            "description": row.get("shortDescription", ""),
            "required_action": row.get("requiredAction", ""),
            "due_date": row.get("dueDate", ""),
            "known_ransomware_use": row.get("knownRansomwareCampaignUse", ""),
        })

    return {"total_matched": len(matches), "results": matches[:limit]}