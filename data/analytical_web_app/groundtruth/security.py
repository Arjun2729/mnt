"""Security boundaries: URL allowlisting and read-only SQL enforcement.

Both guards here are deliberately conservative. The API fetcher refuses anything
that is not an explicitly permitted public host, and the SQL guard parses the
statement rather than prefix-matching it.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import duckdb

# Hosts the API connector may reach. Empty list means "nothing is permitted",
# which is the correct default for a tool that fetches URLs users type in.
DEFAULT_ALLOWED_HOSTS = [h.strip() for h in os.getenv("GT_ALLOWED_API_HOSTS", "").split(",") if h.strip()]

_BLOCKED_SCHEMES = {"file", "ftp", "gopher", "data", "dict"}


class SecurityError(Exception):
    """Raised when a request or query violates a security boundary."""


def _resolves_to_private_address(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise SecurityError(f"Host does not resolve: {host}")
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return True
    return False


def validate_api_url(url: str, allowed_hosts: list[str] | None = None, allow_private: bool = False) -> str:
    """Reject anything that is not an allowlisted public http(s) endpoint."""
    allowed = DEFAULT_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts
    parsed = urlparse(url)

    if parsed.scheme in _BLOCKED_SCHEMES:
        raise SecurityError(f"Scheme not permitted: {parsed.scheme}")
    if parsed.scheme not in {"http", "https"}:
        raise SecurityError("API URL must use http:// or https://")
    host = parsed.hostname
    if not host:
        raise SecurityError("API URL has no host")

    if not allowed:
        raise SecurityError(
            "No API hosts are allowlisted. Set GT_ALLOWED_API_HOSTS to a comma-separated "
            "list of hostnames before fetching remote data."
        )
    if host not in allowed:
        raise SecurityError(f"Host {host!r} is not in the allowlist: {', '.join(sorted(allowed))}")

    # An allowlisted name can still point at internal infrastructure (DNS rebinding,
    # a misconfigured internal record), so verify where it actually resolves.
    if not allow_private and _resolves_to_private_address(host):
        raise SecurityError(f"Host {host!r} resolves to a private address")
    return url


def assert_read_only(query: str) -> str:
    """Confirm a statement is a single read-only SELECT/WITH via the SQL parser.

    Prefix matching cannot see a second statement hiding behind a semicolon, and
    it cannot tell a CTE that ends in a SELECT from one that ends in a DELETE.
    DuckDB's own parser can, so we ask it.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise SecurityError("Empty query")

    try:
        statements = duckdb.extract_statements(stripped)
    except Exception as exc:  # unparseable is not runnable
        raise SecurityError(f"Could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise SecurityError(f"Only one statement may be run at a time (found {len(statements)})")

    # DuckDB labels SELECT and WITH-wrapped reads as SELECT statements.
    statement_type = str(statements[0].type).rsplit(".", 1)[-1].upper()
    if statement_type != "SELECT":
        raise SecurityError(f"Only read-only SELECT queries are allowed (got {statement_type})")

    lowered = stripped.lower()
    for forbidden in ("create ", "insert ", "update ", "delete ", "drop ", "alter ", "attach ", "copy ", "install ", "load ", "export "):
        if lowered.startswith(forbidden):
            raise SecurityError(f"Statement type not permitted: {forbidden.strip()}")
    return stripped
