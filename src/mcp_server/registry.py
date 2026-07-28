"""Tool registry.

Single source of truth for what tools exist. Used by:
  - the FastMCP server (exposes tools to MCP clients)
  - the RAG agent (calls tools directly via OpenAI function-calling)

Both paths share the SAME implementation — the tools in src/mcp_server/tools/.
"""

from __future__ import annotations

from typing import Any, Callable

from src.mcp_server.tools.cve import lookup_cve
from src.mcp_server.tools.kev import search_kev


ToolFn = Callable[..., dict[str, Any]]


TOOLS: dict[str, dict[str, Any]] = {
    "lookup_cve": {
        "function": lookup_cve,
        "schema": {
            "type": "function",
            "function": {
                "name": "lookup_cve",
                "description": (
                    "Look up detailed real-time information about a specific CVE "
                    "(Common Vulnerabilities and Exposures) identifier. Returns "
                    "description, CVSS severity score, affected CWEs, publication "
                    "date, and reference URLs from the NVD database. Use this when "
                    "the user asks about a specific CVE ID or a named vulnerability "
                    "you can map to one (e.g. 'Log4Shell' -> CVE-2021-44228)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cve_id": {
                            "type": "string",
                            "description": "CVE identifier in the format CVE-YYYY-NNNN, e.g. 'CVE-2021-44228'.",
                        },
                    },
                    "required": ["cve_id"],
                },
            },
        },
    },
    "search_kev": {
        "function": search_kev,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_kev",
                "description": (
                    "Search CISA's Known Exploited Vulnerabilities (KEV) catalog. "
                    "Returns vulnerabilities currently being exploited that match "
                    "the vendor, product, or keyword filter. Use this when the user "
                    "asks about actively exploited vulnerabilities, ransomware "
                    "campaigns, or current threats affecting specific vendors."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vendor": {"type": "string", "description": "Vendor filter, e.g. 'Microsoft'."},
                        "product": {"type": "string", "description": "Product filter."},
                        "keyword": {"type": "string", "description": "Keyword filter in name / description."},
                        "limit": {"type": "integer", "description": "Max results (default 10)."},
                    },
                    "required": [],
                },
            },
        },
    },
}


def get_tool_schemas() -> list[dict[str, Any]]:
    return [t["schema"] for t in TOOLS.values()]


def invoke_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    fn: ToolFn = TOOLS[name]["function"]
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}