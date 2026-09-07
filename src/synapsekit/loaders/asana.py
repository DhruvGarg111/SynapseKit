"""Asana task loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._http_utils import http_client, safe_followup_url
from ._record_utils import records_to_documents, response_json
from .base import Document


class AsanaLoader:
    """Load Asana tasks, following the API's cursor pagination."""

    def __init__(
        self,
        access_token: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        token: str | None = None,
        limit: int | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        endpoint_url: str = "https://app.asana.com/api/1.0",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        access_token = access_token or token
        if not access_token:
            raise ValueError("access_token must be provided")
        if not project_id and not workspace_id:
            raise ValueError("project_id or workspace_id must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._access_token = access_token
        self._project_id = project_id
        self._workspace_id = workspace_id
        self._limit = limit
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        url = f"{self._endpoint_url}/tasks"
        requested_fields = [*(self._text_fields or []), *(self._metadata_fields or [])]
        opt_fields = list(dict.fromkeys(["gid", "name", "notes", *requested_fields]))
        params: dict[str, Any] = {
            "limit": min(self._limit or 100, 100),
            "opt_fields": ",".join(opt_fields),
        }
        if self._project_id:
            params["project"] = self._project_id
        if self._workspace_id:
            params["workspace"] = self._workspace_id

        records: list[dict[str, Any]] = []
        with http_client(self._client, self._timeout, "asana") as client:
            while True:
                response = client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {self._access_token}"},
                )
                payload = response_json(response)
                page = payload.get("data", [])
                if isinstance(page, list):
                    records.extend(item for item in page if isinstance(item, dict))
                if self._limit is not None and len(records) >= self._limit:
                    break
                next_uri = (
                    payload.get("next_page", {}).get("uri") if payload.get("next_page") else None
                )
                if not next_uri:
                    break
                next_url = safe_followup_url(str(next_uri), url)
                if next_url is None:
                    raise RuntimeError("refusing cross-origin Asana pagination URL")
                url = next_url
                params = {}

        documents = records_to_documents(
            records,
            "asana",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            project_id=self._project_id,
            workspace_id=self._workspace_id,
        )
        for document in documents:
            document.metadata.setdefault("id", document.metadata.get("gid"))
        return documents

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
