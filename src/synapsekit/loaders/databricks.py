"""Databricks SQL loader."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from ._record_utils import build_record_text
from .base import Document


class DatabricksLoader:
    """Load rows from a Databricks SQL Warehouse query."""

    def __init__(
        self,
        query: str,
        server_hostname: str | None = None,
        http_path: str | None = None,
        access_token: str | None = None,
        host: str | None = None,
        hostname: str | None = None,
        endpoint_url: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int | None = None,
        client: Any | None = None,
    ) -> None:
        if not query:
            raise ValueError("query must be provided")
        resolved_host = server_hostname or host or hostname or endpoint_url
        if not resolved_host and client is None:
            raise ValueError("server_hostname or host must be provided")
        if not http_path and client is None:
            raise ValueError("http_path must be provided")
        if not access_token and client is None:
            raise ValueError("access_token must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        if resolved_host and "://" in resolved_host:
            parsed = urlparse(resolved_host)
            if parsed.hostname:
                resolved_host = parsed.hostname
                if parsed.port:
                    resolved_host = f"{resolved_host}:{parsed.port}"
        self._query = query
        self._server_hostname = resolved_host
        self._http_path = http_path
        self._access_token = access_token
        self._catalog = catalog
        self._schema = schema
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._client = client

    def load(self) -> list[Document]:
        connection = self._client or self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(self._effective_query())
            rows = cursor.fetchall()
            columns = [self._column_name(column) for column in (cursor.description or [])]
        except Exception as exc:
            raise RuntimeError(f"Databricks query failed: {exc}") from exc
        finally:
            if cursor is not None:
                close = getattr(cursor, "close", None)
                if callable(close):
                    close()
            if self._client is None:
                close = getattr(connection, "close", None)
                if callable(close):
                    close()

        documents: list[Document] = []
        for index, row in enumerate(rows):
            record = (
                dict(row) if isinstance(row, Mapping) else dict(zip(columns, row, strict=False))
            )
            text = build_record_text(record, self._text_fields)
            if not text:
                continue
            metadata: dict[str, Any] = {
                "source": "databricks",
                "row": index,
                "query": self._query,
                "catalog": self._catalog,
                "schema": self._schema,
            }
            if self._metadata_fields is None:
                metadata.update(record)
            else:
                metadata.update(
                    {key: record[key] for key in self._metadata_fields if key in record}
                )
            metadata["source"] = "databricks"
            documents.append(Document(text=text, metadata=metadata))
        return documents

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _connect(self) -> Any:
        try:
            from databricks import sql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector required: pip install synapsekit[databricks]"
            ) from None
        kwargs: dict[str, Any] = {
            "server_hostname": self._server_hostname,
            "http_path": self._http_path,
            "access_token": self._access_token,
        }
        if self._catalog:
            kwargs["catalog"] = self._catalog
        if self._schema:
            kwargs["schema"] = self._schema
        return sql.connect(**kwargs)

    def _effective_query(self) -> str:
        if self._limit is None:
            return self._query
        query = self._query.strip()
        query = re.sub(r";(?=\s*(?:--|/\*))", "", query, count=1)
        query = query.rstrip(";").rstrip()
        return f"SELECT * FROM (\n{query}\n) AS synapsekit_limited_query LIMIT {int(self._limit)}"

    @staticmethod
    def _column_name(column: Any) -> str:
        if isinstance(column, (tuple, list)):
            return str(column[0])
        return str(getattr(column, "name", column))
