"""ClickUp task loader."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from ._http_utils import http_client
from ._record_utils import records_to_documents, response_json
from .base import Document


class ClickUpLoader:
    """Load tasks from a ClickUp list or workspace team."""

    def __init__(
        self,
        api_token: str | None = None,
        list_id: str | None = None,
        team_id: str | None = None,
        token: str | None = None,
        limit: int | None = None,
        include_closed: bool = True,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        endpoint_url: str = "https://api.clickup.com/api/v2",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        api_token = api_token or token
        if not api_token:
            raise ValueError("api_token must be provided")
        if not list_id and not team_id:
            raise ValueError("list_id or team_id must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._api_token = api_token
        self._list_id = list_id
        self._team_id = team_id
        self._limit = limit
        self._include_closed = include_closed
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        if self._list_id:
            url = f"{self._endpoint_url}/list/{quote(self._list_id, safe='')}/task"
        else:
            url = f"{self._endpoint_url}/team/{quote(str(self._team_id), safe='')}/task"
        page = 0
        records: list[dict[str, Any]] = []
        headers = {"Authorization": self._api_token}

        with http_client(self._client, self._timeout, "clickup") as client:
            while True:
                params: dict[str, Any] = {
                    "page": page,
                    "include_closed": str(self._include_closed).lower(),
                }
                response = client.get(url, params=params, headers=headers)
                payload = response_json(response)
                batch = payload.get("tasks", [])
                if isinstance(batch, list):
                    records.extend(item for item in batch if isinstance(item, dict))
                if self._limit is not None and len(records) >= self._limit:
                    break
                if payload.get("last_page", True) or not batch:
                    break
                page += 1

        documents = records_to_documents(
            records,
            "clickup",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            list_id=self._list_id,
            team_id=self._team_id,
        )
        for document in documents:
            document.metadata["task_id"] = document.metadata.get("id")
        return documents

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
