"""Private HTTP helpers for optional loader integrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urljoin, urlparse

from ._url_guard import SSRFValidationError, assert_response_url_public, redirect_target

# Cap manual redirect following so a redirect loop can't spin forever.
_MAX_REDIRECTS = 10


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


class _GuardedHTTPClient:
    """Wrap an httpx-style client so every request is SSRF-validated.

    The loaders in this package fetch developer-configured API endpoints and, in
    some cases, URLs echoed back in a response body. Without a guard a fetch can
    be pointed (directly, or via an HTTP redirect from an otherwise-trusted
    endpoint) at ``http://169.254.169.254/`` (cloud metadata), ``127.0.0.1`` or an
    internal ``10.0.0.0/8`` service — a classic SSRF. This wrapper delegates
    ``get``/``post`` to the underlying client but:

    * disables the client's own auto-redirect (the owned client is created with
      ``follow_redirects=False``) and follows redirects manually so **every hop**
      is re-validated by the shared guard, and
    * optionally validates the very first URL too (``validate_initial``).

    For the library's own lazily-created client every URL is validated. When the
    caller injects their own client they own its request policy, so only the
    redirect hops (which come from the remote server and are therefore
    attacker-influenced) are validated — the caller-supplied initial URL is left
    to the caller. This keeps the guard effective on the default path while not
    second-guessing an explicitly injected client.
    """

    def __init__(self, inner: Any, *, validate_initial: bool) -> None:
        self._inner = inner
        self._validate_initial = validate_initial

    def get(self, url: str, **kwargs: Any) -> Any:
        return self._request("get", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        return self._request("post", url, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Proxy any other attribute/method (e.g. .stream, .headers) to the inner
        # client unchanged.
        return getattr(self._inner, name)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        current = url
        for hop in range(_MAX_REDIRECTS + 1):
            if hop > 0 or self._validate_initial:
                assert_response_url_public(current)
            response = getattr(self._inner, method)(current, **kwargs)
            next_url = redirect_target(response)
            if next_url is None:
                return response
            current = next_url
        raise SSRFValidationError(f"Exceeded {_MAX_REDIRECTS} redirects while fetching {url!r}")


@contextmanager
def http_client(
    client: Any | None,
    timeout: float,
    install_extra: str,
) -> Iterator[Any]:
    """Yield an SSRF-guarded client wrapping an injected or lazily-created one."""
    if client is not None:
        yield _GuardedHTTPClient(client, validate_initial=False)
        return

    try:
        import httpx
    except ImportError:
        raise ImportError(f"httpx required: pip install synapsekit[{install_extra}]") from None

    # follow_redirects=False so redirects are followed manually and each hop
    # re-validated by _GuardedHTTPClient (see _url_guard for the rationale).
    with httpx.Client(timeout=timeout, follow_redirects=False) as owned_client:
        yield _GuardedHTTPClient(owned_client, validate_initial=True)
