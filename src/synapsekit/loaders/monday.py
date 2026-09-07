"""Monday.com board item loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._http_utils import http_client
from ._record_utils import records_to_documents, response_json
from .base import Document


class MondayLoader:
    """Load board records from Monday.com's GraphQL endpoint."""

    def __init__(
        self,
        api_key: str,
        board_id: str | int,
        query: str | None = None,
        variables: dict[str, Any] | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int = 100,
        endpoint_url: str = "https://api.monday.com/v2",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided")
        if not board_id:
            raise ValueError("board_id must be provided")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._api_key = api_key
        self._board_id = board_id
        self._query = query
        self._variables = dict(variables or {})
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._endpoint_url = endpoint_url
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        query = self._query or (
            "query($boardId: [ID!], $limit: Int!, $cursor: String) { "
            "boards(ids: $boardId) { id name items_page(limit: $limit, cursor: $cursor) "
            "{ cursor items { id name column_values { id text value } } } } }"
        )
        variables = {
            "boardId": [self._board_id],
            "limit": min(self._limit, 500),
            "cursor": None,
            **self._variables,
        }
        records: list[dict[str, Any]] = []
        with http_client(self._client, self._timeout, "monday") as client:
            while True:
                response = client.post(
                    self._endpoint_url,
                    json={"query": query, "variables": variables},
                    headers={"Authorization": self._api_key, "Content-Type": "application/json"},
                )
                payload = response_json(response)

                if payload.get("errors"):
                    raise RuntimeError(f"Monday GraphQL request failed: {payload['errors']}")
                page_records = self._extract_records(payload.get("data", {}))
                records.extend(page_records)
                if len(records) >= self._limit or not page_records or self._query:
                    break
                cursor = self._extract_cursor(payload.get("data", {}))
                if not cursor:
                    break
                variables["cursor"] = cursor

        return records_to_documents(
            records,
            "monday",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            board_id=self._board_id,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    @staticmethod
    def _extract_records(data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        boards = data.get("boards", [])
        if isinstance(boards, dict):
            boards = [boards]
        if not isinstance(boards, list):
            return []
        records: list[dict[str, Any]] = []
        for board in boards:
            if not isinstance(board, dict):
                continue
            items_page = board.get("items_page")
            items = (
                items_page.get("items", [])
                if isinstance(items_page, dict)
                else board.get("items", [])
            )
            if items:
                records.extend(item for item in items if isinstance(item, dict))
        return records

    @staticmethod
    def _extract_cursor(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        boards = data.get("boards", [])
        if isinstance(boards, dict):
            boards = [boards]
        if not isinstance(boards, list):
            return None
        for board in boards:
            if not isinstance(board, dict):
                continue
            items_page = board.get("items_page")
            if isinstance(items_page, dict) and items_page.get("cursor"):
                return str(items_page["cursor"])
        return None
