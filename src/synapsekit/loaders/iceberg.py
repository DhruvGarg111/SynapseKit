"""Apache Iceberg table loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._record_utils import records_to_documents
from .base import Document


class IcebergLoader:
    """Load rows from an Apache Iceberg table.

    ``catalog`` or ``table`` can be injected for tests and for applications
    that already own a configured catalog session. ``endpoint_url`` is copied
    into the catalog properties for S3-compatible object stores.
    """

    def __init__(
        self,
        identifier: str | None = None,
        table_identifier: str | None = None,
        catalog_name: str = "default",
        catalog_properties: dict[str, Any] | None = None,
        endpoint_url: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int | None = None,
        catalog: Any | None = None,
        table: Any | None = None,
    ) -> None:
        identifier = identifier or table_identifier
        if identifier is None and isinstance(table, str):
            identifier, table = table, None
        if identifier is None and table is None:
            raise ValueError("identifier or table must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._identifier = identifier
        self._catalog_name = catalog_name
        self._catalog_properties = dict(catalog_properties or {})
        if endpoint_url:
            self._catalog_properties.setdefault("s3.endpoint", endpoint_url)
        self._endpoint_url = endpoint_url
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._catalog = catalog
        self._table = table

    def load(self) -> list[Document]:
        table = self._table if self._table is not None else self._load_table()
        scan = table.scan()
        scan_limit = getattr(scan, "limit", None)
        if self._limit is not None and callable(scan_limit):
            scan = scan_limit(self._limit)
        arrow_method = getattr(scan, "to_arrow", None)
        if callable(arrow_method):
            if self._limit is not None and not callable(scan_limit):
                try:
                    arrow_table = arrow_method(limit=self._limit)
                except TypeError:
                    arrow_table = arrow_method()
            else:
                arrow_table = arrow_method()
        else:
            arrow_table = scan
        if hasattr(arrow_table, "to_pylist"):
            rows = arrow_table.to_pylist()
        elif hasattr(arrow_table, "to_pandas"):
            rows = arrow_table.to_pandas().to_dict("records")
        else:
            rows = list(arrow_table)
        if self._limit is not None:
            rows = rows[: self._limit]
        return records_to_documents(
            rows,
            "iceberg",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            identifier=self._identifier,
            endpoint_url=self._endpoint_url,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _load_table(self) -> Any:
        catalog = self._catalog
        if catalog is None:
            try:
                from pyiceberg.catalog import load_catalog
            except ImportError:
                raise ImportError("pyiceberg required: pip install synapsekit[iceberg]") from None
            catalog = load_catalog(self._catalog_name, **self._catalog_properties)
        if self._identifier is None:
            raise ValueError("identifier must be provided when table is not injected")
        return catalog.load_table(self._identifier)
