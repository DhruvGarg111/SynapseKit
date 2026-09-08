from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urljoin

from ...loaders._url_guard import assert_response_url_public
from ..base import BaseTool, ToolResult

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 10


class HTTPRequestTool(BaseTool):
    """Make HTTP requests (GET, POST, PUT, DELETE, PATCH).

    A single ``aiohttp.ClientSession`` is created lazily on the first request
    and reused for all subsequent calls on the same tool instance.  This
    preserves TCP connection pooling and avoids the overhead of a new TLS
    handshake on every call.

    Call ``await tool.aclose()`` (or use as an async context manager) to
    release the underlying connection pool when the tool is no longer needed.
    """

    name = "http_request"
    description = (
        "Make an HTTP request to a URL. "
        "Input: method (GET/POST/PUT/DELETE/PATCH), url, optional body and headers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to request"},
            "method": {
                "type": "string",
                "description": "HTTP method (default: GET)",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "default": "GET",
            },
            "body": {
                "type": "string",
                "description": "Request body (for POST/PUT/PATCH)",
                "default": "",
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs",
                "default": {},
            },
        },
        "required": ["url"],
    }

    def __init__(self, max_response_length: int = 10000, timeout: int = 30) -> None:
        self._max_length = max_response_length
        self._timeout = timeout
        self._session: Any | None = None  # aiohttp.ClientSession, created lazily

    async def _get_session(self) -> Any:
        """Return the shared session, creating it on first use."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for HTTPRequestTool: pip install aiohttp") from None

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> HTTPRequestTool:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def run(
        self,
        url: str = "",
        method: str = "GET",
        body: str = "",
        headers: dict | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        url = url or kwargs.get("input", "")
        if not url:
            return ToolResult(output="", error="No URL provided.")

        method = method.upper()
        req_headers = headers or {}
        loop = asyncio.get_running_loop()

        # Validate the initial URL against the SSRF guard (fail-closed) before
        # doing anything else — this also rejects a bad URL without needing the
        # aiohttp client. Without this, an LLM-supplied URL — or a redirect from
        # a public host — could reach 169.254.169.254, 127.0.0.1, an internal
        # service, or a file:// target.
        try:
            await loop.run_in_executor(None, assert_response_url_public, url)
        except ValueError as e:
            # SSRFValidationError subclasses ValueError.
            return ToolResult(output="", error=str(e))

        # _get_session raises ImportError if aiohttp is missing — let it propagate
        session = await self._get_session()

        try:
            # Disable aiohttp's automatic redirect following and follow manually
            # so every hop is re-validated against the SSRF guard.
            req_kwargs: dict[str, Any] = {"headers": req_headers, "allow_redirects": False}
            if method in ("POST", "PUT", "PATCH") and body:
                req_kwargs["data"] = body

            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                # The guard resolves DNS (blocking); offload off the event loop.
                await loop.run_in_executor(None, assert_response_url_public, current)
                async with session.request(method, current, **req_kwargs) as resp:
                    location = resp.headers.get("Location")
                    if resp.status in _REDIRECT_CODES and location:
                        current = urljoin(current, location)
                        continue
                    status = resp.status
                    text = await resp.text()
                    if len(text) > self._max_length:
                        text = text[: self._max_length] + "\n... (truncated)"
                    return ToolResult(output=f"HTTP {status}\n{text}")
            return ToolResult(output="", error="Too many redirects.")
        except ValueError as e:
            # SSRFValidationError subclasses ValueError.
            return ToolResult(output="", error=str(e))
        except Exception as e:
            return ToolResult(output="", error=f"HTTP request failed: {e}")
