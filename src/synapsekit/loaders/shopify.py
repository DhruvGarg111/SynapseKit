"""Shopify Admin API loader."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from ._http_utils import http_client, safe_followup_url
from ._record_utils import records_to_documents, response_json
from .base import Document


class ShopifyLoader:
    """Load Shopify Admin API resources, one resource per Document."""

    def __init__(
        self,
        shop_url: str | None = None,
        access_token: str | None = None,
        resource: str = "products",
        api_version: str = "2024-10",
        token: str | None = None,
        host: str | None = None,
        limit: int | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        shop_url = shop_url or host
        access_token = access_token or token
        if not shop_url and not endpoint_url:
            raise ValueError("shop_url or endpoint_url must be provided")
        if not access_token:
            raise ValueError("access_token must be provided")
        if not resource or not re.fullmatch(r"[a-z][a-z0-9_/-]*", resource):
            raise ValueError("resource must be a valid Shopify resource name")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        if shop_url and "://" not in shop_url:
            shop_url = f"https://{shop_url}"
        self._shop_url = shop_url.rstrip("/") if shop_url else None
        self._access_token = access_token
        self._resource = resource.strip("/")
        self._api_version = api_version
        self._limit = limit
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._endpoint_url = endpoint_url.rstrip("/") if endpoint_url else None
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        if self._endpoint_url:
            url = self._endpoint_url
        else:
            url = f"{self._shop_url}/admin/api/{self._api_version}/{self._resource}.json"
        records: list[dict[str, Any]] = []
        with http_client(self._client, self._timeout, "shopify") as client:
            while True:
                params = {"limit": min(self._limit or 250, 250)}
                payload_response = client.get(
                    url,
                    params=params,
                    headers={"X-Shopify-Access-Token": self._access_token},
                )
                payload = response_json(payload_response)
                batch = payload.get(self._response_key(), [])
                if isinstance(batch, list):
                    records.extend(item for item in batch if isinstance(item, dict))
                elif isinstance(batch, Mapping):
                    records.append(dict(batch))
                if self._limit is not None and len(records) >= self._limit:
                    break
                next_url = self._next_url(getattr(payload_response, "headers", {}), url)
                if not next_url:
                    break
                url = next_url

        return records_to_documents(
            records,
            "shopify",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            resource=self._resource,
            shop_url=self._shop_url,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    @staticmethod
    def _next_url(headers: Any, base_url: str) -> str | None:
        link = headers.get("Link", "") if hasattr(headers, "get") else ""
        match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        return safe_followup_url(match.group(1), base_url) if match else None

    def _response_key(self) -> str:
        segments = self._resource.split("/")
        collection = segments[-1]
        is_single_resource = len(segments) > 1 and not collection.endswith("s")
        if is_single_resource:
            collection = segments[-2]
        if not is_single_resource:
            return collection
        irregular = {"categories": "category", "people": "person"}
        if collection in irregular:
            return irregular[collection]
        if collection.endswith("ies"):
            return f"{collection[:-3]}y"
        return collection[:-1] if collection.endswith("s") else collection
