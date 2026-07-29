"""HTTP request safety guard for the API agent — SSRF defense-in-depth.

Layers: (1) scheme allowlist (http/https), (2) DNS resolution + reject any
private/loopback/link-local/reserved IP (blocks 127.0.0.1, 10.x, 192.168.x,
169.254.169.254 cloud metadata, ::1), (3) method allowlist (GET/HEAD),
(4) port allowlist (80/443/8080), (5) optional host allowlist.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_METHODS = {"GET", "HEAD"}
ALLOWED_PORTS = {80, 443, 8080}
DEFAULT_HOST_ALLOWLIST: set[str] = set()


class RequestValidationError(Exception):
    """Raised when an outbound HTTP request fails a safety check."""


def _resolved_ips(host: str) -> list[ipaddress._BaseAddress]:
    infos = socket.getaddrinfo(host, None)
    return [ipaddress.ip_address(sockaddr[0]) for _f, _t, _p, _c, sockaddr in infos]


def _is_public(ip: ipaddress._BaseAddress) -> bool:
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_request(
    url: str,
    method: str = "GET",
    host_allowlist: set[str] | None = None,
) -> tuple[str, str]:
    """Validate an outbound request. Returns (method, url) or raises."""
    method = (method or "GET").upper()
    if method not in ALLOWED_METHODS:
        raise RequestValidationError(
            f"Method {method!r} not allowed (only {sorted(ALLOWED_METHODS)})."
        )

    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise RequestValidationError(
            f"Scheme {parsed.scheme!r} not allowed (only {sorted(ALLOWED_SCHEMES)})."
        )

    host = parsed.hostname
    if not host:
        raise RequestValidationError("URL has no host.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise RequestValidationError(f"Port {port} not allowed (only {sorted(ALLOWED_PORTS)}).")

    allowlist = host_allowlist if host_allowlist is not None else DEFAULT_HOST_ALLOWLIST
    if allowlist:
        if not any(host == h or host.endswith("." + h) for h in allowlist):
            raise RequestValidationError(f"Host {host!r} not in allowlist {sorted(allowlist)}.")

    try:
        literal = ipaddress.ip_address(host)
        if not _is_public(literal):
            raise RequestValidationError(f"Host IP {host} is private/reserved (SSRF).")
    except ValueError:
        pass

    try:
        ips = _resolved_ips(host)
    except Exception as e:
        raise RequestValidationError(f"DNS resolution failed for {host!r}: {e}") from e

    if not ips:
        raise RequestValidationError(f"No IPs resolved for {host!r}.")
    for ip in ips:
        if not _is_public(ip):
            raise RequestValidationError(
                f"Host {host!r} resolves to non-public IP {ip} (SSRF blocked)."
            )

    return method, url