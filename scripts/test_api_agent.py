"""M12 verification: API agent SSRF safety + a live functional call.

Usage:
    python -m scripts.test_api_agent
"""

from __future__ import annotations

from src.agents.api_agent import APIAgent
from src.tools.http_guard import RequestValidationError, validate_request

# SSRF / safety attacks — must ALL be blocked by validate_request.
SSRF_ATTACKS = [
    ("http://169.254.169.254/latest/meta-data/", "GET"),   # cloud metadata
    ("http://127.0.0.1/admin", "GET"),                      # loopback
    ("http://10.0.0.5/internal", "GET"),                    # private range
    ("http://192.168.1.1/", "GET"),                         # private range
    ("http://[::1]/", "GET"),                               # ipv6 loopback
    ("file:///etc/passwd", "GET"),                          # bad scheme
    ("http://example.com:22/", "GET"),                      # bad port
    ("https://example.com/", "POST"),                       # bad method
]


def main() -> int:
    print("=" * 68)
    print(" SSRF guard: internal / dangerous requests must be BLOCKED")
    print("=" * 68)
    all_blocked = True
    for url, method in SSRF_ATTACKS:
        try:
            validate_request(url, method)
            print(f"  [LEAK!] {method} {url}")
            all_blocked = False
        except RequestValidationError as e:
            print(f"  [blocked] {method} {url[:45]:<45} -> {str(e)[:38]}")
    print(f"\n  All SSRF attacks blocked: {all_blocked}\n")

    print("=" * 68)
    print(" Functional: API agent makes a real public API call")
    print("=" * 68)
    agent = APIAgent()
    q = "How many stargazers does the public GitHub repo langchain-ai/langchain have?"
    print(f"\nQ: {q}")
    resp = agent.run(q)
    print(f"  planned url: {resp.metadata.get('url', '')[:90]}")
    print(f"  status: {resp.metadata.get('status_code')}  blocked: {resp.metadata.get('blocked')}")
    print(f"  A: {resp.answer[:200]}")

    return 0 if all_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())