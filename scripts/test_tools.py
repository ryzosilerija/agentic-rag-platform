"""Verify tools work end-to-end against the real NVD + CISA APIs.

Usage:
    python -m scripts.test_tools
"""

from __future__ import annotations

import json

from src.mcp_server.registry import invoke_tool


def _dump(name: str, result: dict) -> None:
    print(f"\n--- {name} ---")
    print(json.dumps(result, indent=2, default=str)[:1200])


def main() -> int:
    r = invoke_tool("lookup_cve", {"cve_id": "CVE-2021-44228"})
    _dump("lookup_cve(CVE-2021-44228)  # Log4Shell", r)
    assert "error" not in r, r.get("error")
    assert r.get("severity", "").lower() in ("critical", "high"), r

    r = invoke_tool("lookup_cve", {"cve_id": "not-a-cve"})
    _dump("lookup_cve('not-a-cve')  # expect error", r)
    assert "error" in r

    r = invoke_tool("search_kev", {"vendor": "Microsoft", "limit": 3})
    _dump("search_kev(vendor='Microsoft', limit=3)", r)
    assert "error" not in r, r.get("error")
    assert r.get("total_matched", 0) > 0, r

    r = invoke_tool("search_kev", {"keyword": "remote code execution", "limit": 3})
    _dump("search_kev(keyword='remote code execution', limit=3)", r)
    assert "error" not in r, r.get("error")

    print("\nAll tool tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())