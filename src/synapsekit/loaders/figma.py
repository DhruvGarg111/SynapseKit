"""Figma design-file loader."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from ._http_utils import http_client
from ._record_utils import response_json
from .base import Document


class FigmaLoader:
    """Load a Figma file or selected nodes as design-context Documents."""

    def __init__(
        self,
        access_token: str | None = None,
        file_key: str | None = None,
        file_id: str | None = None,
        token: str | None = None,
        node_ids: list[str] | None = None,
        endpoint_url: str = "https://api.figma.com/v1",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        access_token = access_token or token
        file_key = file_key or file_id
        if not access_token:
            raise ValueError("access_token must be provided")
        if not file_key:
            raise ValueError("file_key must be provided")

        self._access_token = access_token
        self._file_key = file_key
        self._node_ids = node_ids
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        headers = {"X-Figma-Token": self._access_token}
        with http_client(self._client, self._timeout, "figma") as client:
            if self._node_ids:
                response = client.get(
                    f"{self._endpoint_url}/files/{quote(self._file_key, safe='')}/nodes",
                    params={"ids": ",".join(self._node_ids)},
                    headers=headers,
                )
                payload = response_json(response)
                return self._node_documents(payload)

            response = client.get(
                f"{self._endpoint_url}/files/{quote(self._file_key, safe='')}", headers=headers
            )
            payload = response_json(response)

        document = payload.get("document", {})
        text = self._node_text(document)
        if payload.get("name"):
            text = f"{payload['name']}\n{text}" if text else str(payload["name"])
        return [
            Document(
                text=text,
                metadata={
                    "source": "figma",
                    "file_key": self._file_key,
                    "name": payload.get("name"),
                    "last_modified": payload.get("lastModified"),
                    "version": payload.get("version"),
                },
            )
        ]

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _node_documents(self, payload: dict[str, Any]) -> list[Document]:
        documents: list[Document] = []
        nodes = payload.get("nodes", {})
        for node_id, value in nodes.items() if isinstance(nodes, dict) else []:
            node = value.get("document", value) if isinstance(value, dict) else {}
            text = self._node_text(node)
            if not text:
                continue
            node_metadata = {
                key: node[key]
                for key in ("id", "name", "type", "visible", "locked", "opacity")
                if key in node
            }
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": "figma",
                        "file_key": self._file_key,
                        "node_id": node_id,
                        **node_metadata,
                    },
                )
            )
        return documents

    def _node_text(self, node: Any) -> str:
        if not isinstance(node, dict):
            return ""
        parts: list[str] = []
        for key in ("name", "characters", "description"):
            value = node.get(key)
            if value:
                parts.append(str(value))
        for child in node.get("children", []) or []:
            child_text = self._node_text(child)
            if child_text:
                parts.append(child_text)
        return "\n".join(parts)
