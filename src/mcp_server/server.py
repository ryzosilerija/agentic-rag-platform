"""FastMCP server exposing our cybersecurity tools to any MCP client.

Standalone run:
    python -m src.mcp_server.server

Or plug into Claude Desktop / VS Code by adding to their MCP config:
    {
      "mcpServers": {
        "cybersec-tools": {
          "command": "python",
          "args": ["-m", "src.mcp_server.server"],
          "cwd": "D:/Documents/GitHub/agentic-rag-platform"
        }
      }
    }
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp_server.tools.cve import lookup_cve as _lookup_cve
from src.mcp_server.tools.kev import search_kev as _search_kev

mcp = FastMCP("agentic-rag-cybersec-tools")


@mcp.tool()
def lookup_cve(cve_id: str) -> dict[str, Any]:
    """Look up a CVE by ID.

    Args:
        cve_id: CVE identifier like 'CVE-2021-44228'.
    """
    return _lookup_cve(cve_id)


@mcp.tool()
def search_kev(
    vendor: str = "",
    product: str = "",
    keyword: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    """Search CISA's Known Exploited Vulnerabilities catalog.

    Args:
        vendor: Vendor filter, e.g. 'Microsoft'.
        product: Product filter.
        keyword: Keyword filter.
        limit: Max results (default 10).
    """
    return _search_kev(vendor=vendor, product=product, keyword=keyword, limit=limit)


if __name__ == "__main__":
    mcp.run()