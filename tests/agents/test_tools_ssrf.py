"""SSRF regression tests for agent HTTP tools and the RSS loader.

All targets use IP literals or non-http schemes so the shared SSRF guard
rejects them without any DNS lookup or network access — these tests never
touch the network. Regression for #1022.
"""

from __future__ import annotations

import pytest

from synapsekit.agents.tools.api_builder import APIBuilderTool
from synapsekit.agents.tools.graphql import GraphQLTool
from synapsekit.agents.tools.http_request import HTTPRequestTool
from synapsekit.agents.tools.web_scraper import WebScraperTool
from synapsekit.loaders.rss import RSSLoader

# (label, url) pairs that must be rejected everywhere.
BLOCKED_HTTP = [
    ("cloud-metadata", "http://169.254.169.254/latest/meta-data/"),
    ("loopback", "http://127.0.0.1/admin"),
    ("rfc1918", "http://10.0.0.5/internal"),
]


class TestHTTPRequestToolSSRF:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("label,url", BLOCKED_HTTP)
    async def test_blocks_private_targets(self, label, url):
        r = await HTTPRequestTool().run(url=url)
        assert r.is_error
        assert "private/internal" in r.error

    @pytest.mark.asyncio
    async def test_blocks_file_scheme(self):
        r = await HTTPRequestTool().run(url="file:///etc/passwd")
        assert r.is_error
        assert "not allowed" in r.error


class TestGraphQLToolSSRF:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("label,url", BLOCKED_HTTP)
    async def test_blocks_private_targets(self, label, url):
        r = await GraphQLTool().run(url=url, query="{ __typename }")
        assert r.is_error
        assert "private/internal" in r.error

    @pytest.mark.asyncio
    async def test_blocks_file_scheme(self):
        r = await GraphQLTool().run(url="file:///etc/passwd", query="{ x }")
        assert r.is_error
        assert "not allowed" in r.error


class TestAPIBuilderToolSSRF:
    @pytest.mark.asyncio
    async def test_blocks_private_spec_url(self):
        r = await APIBuilderTool().run(
            intent="fetch data", openapi_url="http://169.254.169.254/openapi.json"
        )
        assert r.is_error

    @pytest.mark.asyncio
    async def test_blocks_private_request_via_server_url(self):
        r = await APIBuilderTool().run(
            intent="list users", path="/users", method="GET", server_url="http://127.0.0.1"
        )
        assert r.is_error

    @pytest.mark.asyncio
    async def test_blocks_file_scheme_spec_url(self):
        r = await APIBuilderTool().run(intent="x", openapi_url="file:///etc/passwd")
        assert r.is_error


class TestWebScraperToolSSRF:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("label,url", BLOCKED_HTTP)
    async def test_blocks_private_targets(self, label, url):
        r = await WebScraperTool().run(url=url)
        assert r.is_error
        assert "private/internal" in r.error

    @pytest.mark.asyncio
    async def test_blocks_file_scheme(self):
        r = await WebScraperTool().run(url="file:///etc/passwd")
        assert r.is_error
        assert "not allowed" in r.error


class TestRSSLoaderSSRF:
    def test_blocks_file_scheme(self):
        with pytest.raises(ValueError, match="not allowed"):
            RSSLoader("file:///etc/passwd").load()

    def test_blocks_private_http(self):
        with pytest.raises(ValueError, match="private/internal"):
            RSSLoader("http://127.0.0.1/feed.xml").load()

    def test_allows_local_path_without_scheme(self, tmp_path):
        # A bare local path (no scheme) must not be rejected by the guard — it
        # reaches feedparser, which returns an empty feed for a nonexistent
        # file. If feedparser isn't installed we only assert the guard passed.
        target = str(tmp_path / "feed.xml")
        try:
            docs = RSSLoader(target).load()
        except ImportError:
            pytest.skip("feedparser not installed")
        assert docs == []
