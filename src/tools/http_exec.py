"""Guarded HTTP executor for the API agent."""

from __future__ import annotations

import httpx

from src.tools.http_guard import RequestValidationError, validate_request

_TIMEOUT = 10.0
_MAX_BYTES = 100_000


def guarded_fetch(url: str, method: str = "GET") -> dict:
    """Validate then perform a bounded HTTP request. Returns a result dict."""
    try:
        method, url = validate_request(url, method)
    except RequestValidationError as e:
        return {"error": f"blocked by safety guard: {e}", "blocked": True}

    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=False) as client:
            resp = client.request(method, url, headers={"User-Agent": "agentic-rag-platform/0.1"})
    except Exception as e:
        return {"error": f"request failed: {e}", "blocked": False}

    if resp.is_redirect:
        return {
            "status_code": resp.status_code,
            "note": "redirect not followed (safety)",
            "location": resp.headers.get("location", ""),
            "blocked": False,
        }

    body = resp.text[:_MAX_BYTES]
    truncated = len(resp.text) > _MAX_BYTES
    return {
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "body": body,
        "truncated": truncated,
        "blocked": False,
    }