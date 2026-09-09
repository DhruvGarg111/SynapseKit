"""Generic GraphQL endpoint loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._http_utils import http_client
from ._record_utils import records_to_documents, response_json
from .base import Document


class GraphQLLoader:
    """Execute a read-only GraphQL query and convert returned records."""

    def __init__(
        self,
        endpoint_url: str,
        query: str,
        variables: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data_path: str | None = None,
        record_path: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int | None = None,
        page_info_path: str | None = None,
        cursor_variable: str = "after",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not endpoint_url:
            raise ValueError("endpoint_url must be provided")
        if not query:
            raise ValueError("query must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._endpoint_url = endpoint_url
        self._query = query
        self._variables = dict(variables or {})
        self._headers = dict(headers or {})
        self._data_path = data_path
        self._record_path = record_path
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._page_info_path = page_info_path
        self._cursor_variable = cursor_variable
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        headers = {"Content-Type": "application/json", **self._headers}
        variables = dict(self._variables)
        if self._page_info_path:
            variables.setdefault(self._cursor_variable, None)
        records: list[dict[str, Any]] = []
        with http_client(self._client, self._timeout, "graphql") as client:
            while True:
                response = client.post(
                    self._endpoint_url,
                    json={"query": self._query, "variables": variables},
                    headers=headers,
                )
                payload = response_json(response)

                if payload.get("errors"):
                    raise RuntimeError(f"GraphQL request failed: {payload['errors']}")
                records.extend(self._extract_records(payload.get("data", {})))
                if self._limit is not None and len(records) >= self._limit:
                    break
                if not self._page_info_path:
                    break
                page_info = self._at_path(payload.get("data", {}), self._page_info_path)
                if not isinstance(page_info, dict) or not page_info.get("hasNextPage"):
                    break
                cursor = page_info.get("endCursor")
                if cursor is None:
                    break
                variables[self._cursor_variable] = cursor
        return records_to_documents(
            records,
            "graphql",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            endpoint_url=self._endpoint_url,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _extract_records(self, data: Any) -> list[dict[str, Any]]:
        value = data
        for path in (self._data_path, self._record_path):
            if path:
                value = self._at_path(value, path)

        while isinstance(value, dict):
            if isinstance(value.get("nodes"), list):
                value = value["nodes"]
                break
            if isinstance(value.get("edges"), list):
                value = [edge.get("node", edge) for edge in value["edges"]]
                break
            list_values = [item for item in value.values() if isinstance(item, list)]
            dict_values = [item for item in value.values() if isinstance(item, dict)]
            if len(list_values) == 1:
                value = list_values[0]
                break
            if len(value) == 1 and len(dict_values) == 1:
                value = dict_values[0]
                continue
            value = [value]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _at_path(value: Any, path: str) -> Any:
        for segment in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(segment)
        return value
