"""Private HTTP helpers for optional loader integrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urljoin, urlparse


def safe_followup_url(
    candidate: str,
    base_url: str,
    *,
    allowed_hosts: set[str] | None = None,
    allowed_host_suffixes: tuple[str, ...] = (),
) -> str | None:
    """Return a same-origin or explicitly trusted follow-up URL."""
    base = urlparse(base_url)
    target = urlparse(urljoin(base_url, candidate))
    if base.scheme not in {"http", "https"} or target.scheme not in {"http", "https"}:
        return None
    if base.scheme == "https" and target.scheme != "https":
        return None
    if target.username or target.password or not target.hostname:
        return None

    base_host = base.hostname.lower() if base.hostname else ""
    target_host = target.hostname.lower()
    try:
        base_port = base.port or (443 if base.scheme == "https" else 80)
        target_port = target.port or (443 if target.scheme == "https" else 80)
    except ValueError:
        return None
    if base_host == target_host and base_port != target_port:
        return None
    hosts = {base_host, *(host.lower() for host in (allowed_hosts or set()))}
    suffix_match = any(
        target_host == suffix.lower() or target_host.endswith(f".{suffix.lower()}")
        for suffix in allowed_host_suffixes
    )
    if target_host not in hosts and not suffix_match:
        return None
    return target.geturl()


@contextmanager
def http_client(
    client: Any | None,
    timeout: float,
    install_extra: str,
) -> Iterator[Any]:
    """Yield an injected client or a lazily-created blocking httpx client."""
    if client is not None:
        yield client
        return

    try:
        import httpx
    except ImportError:
        raise ImportError(f"httpx required: pip install synapsekit[{install_extra}]") from None

    with httpx.Client(timeout=timeout, follow_redirects=True) as owned_client:
        yield owned_client
