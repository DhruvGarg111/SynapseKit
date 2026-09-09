"""SSRF regression tests for the #889 HTTP loaders.

The loaders fetch developer-configured endpoints and, in some cases, URLs echoed
back in a response body. Every outbound request must go through the shared SSRF
guard so a URL (or a redirect from an otherwise-trusted endpoint) cannot be
pointed at loopback, RFC1918, or the ``169.254.169.254`` cloud-metadata address.

These tests are hermetic: they use IP literals (which the guard validates without
any DNS lookup) and hand-written fake clients, so no network access is required.
"""

from __future__ import annotations

import inspect

import pytest

from synapsekit.loaders._http_utils import _GuardedHTTPClient
from synapsekit.loaders._url_guard import SSRFValidationError
from synapsekit.loaders.graphql import GraphQLLoader


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.content = payload if isinstance(payload, bytes) else b""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTPClient:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url))
        return next(self._responses)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url))
        return next(self._responses)


# --- Owned-client path: the library's own client validates every URL ----------


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "http://127.0.0.1:8080/graphql",  # loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/graphql",  # RFC1918 private
        "http://[::1]:8080/graphql",  # IPv6 loopback
    ],
)
def test_owned_client_rejects_private_endpoint(endpoint_url: str) -> None:
    # No injected client -> the library creates its own guarded httpx client,
    # which validates the URL and raises before any socket is opened.
    loader = GraphQLLoader(endpoint_url=endpoint_url, query="{ viewer { id } }")
    with pytest.raises(SSRFValidationError):
        loader.load()


def test_owned_client_rejects_non_http_scheme() -> None:
    loader = GraphQLLoader(endpoint_url="file:///etc/passwd", query="{ x }")
    with pytest.raises(SSRFValidationError):
        loader.load()


# --- Redirect handling: attacker-influenced hops are validated on both paths ---


def test_injected_client_redirect_to_private_is_blocked() -> None:
    # A trusted-looking endpoint that 302s to loopback must not be followed,
    # even when the caller injects their own client.
    client = FakeHTTPClient(
        [FakeResponse({}, status_code=302, headers={"location": "http://127.0.0.1:6379/"})]
    )
    loader = GraphQLLoader(
        endpoint_url="http://93.184.216.34/graphql",
        query="{ x }",
        client=client,
    )
    with pytest.raises(SSRFValidationError):
        loader.load()


def test_injected_client_redirect_to_metadata_is_blocked() -> None:
    client = FakeHTTPClient(
        [FakeResponse({}, status_code=307, headers={"location": "http://169.254.169.254/latest/"})]
    )
    loader = GraphQLLoader(
        endpoint_url="http://93.184.216.34/graphql", query="{ x }", client=client
    )
    with pytest.raises(SSRFValidationError):
        loader.load()


def test_redirect_to_public_host_is_followed() -> None:
    # A redirect to another public address is allowed; the final response wins.
    client = FakeHTTPClient(
        [
            FakeResponse({}, status_code=302, headers={"location": "http://8.8.8.8/graphql"}),
            FakeResponse({"data": {"items": [{"id": "1", "name": "ok"}]}}),
        ]
    )
    loader = GraphQLLoader(
        endpoint_url="http://93.184.216.34/graphql",
        query="{ items { id name } }",
        record_path="items",
        client=client,
    )
    docs = loader.load()
    assert [c[1] for c in client.calls] == [
        "http://93.184.216.34/graphql",
        "http://8.8.8.8/graphql",
    ]
    assert len(docs) == 1


def test_guarded_client_caps_redirect_chain() -> None:
    # An endless public->public redirect loop is bounded, not spun forever.
    loop_response = FakeResponse({}, status_code=302, headers={"location": "http://8.8.8.8/next"})
    inner = FakeHTTPClient([loop_response] * 100)
    guarded = _GuardedHTTPClient(inner, validate_initial=True)
    with pytest.raises(SSRFValidationError, match="redirects"):
        guarded.get("http://8.8.8.8/start")


def test_guarded_client_passes_through_non_redirect_response() -> None:
    inner = FakeHTTPClient([FakeResponse({"ok": True})])
    guarded = _GuardedHTTPClient(inner, validate_initial=False)
    response = guarded.post("http://198.51.100.7/api", json={"q": 1})
    assert response.json() == {"ok": True}
    assert inner.calls == [("POST", "http://198.51.100.7/api")]


def test_aload_is_a_coroutine() -> None:
    loader = GraphQLLoader(endpoint_url="http://93.184.216.34/graphql", query="{ x }")
    assert inspect.iscoroutinefunction(loader.aload)
