"""Delta Lake table loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._record_utils import records_to_documents
from .base import Document


class DeltaLakeLoader:
    """Load rows from a Delta Lake table stored locally or in object storage."""

    def __init__(
        self,
        uri: str | None = None,
        path: str | None = None,
        storage_options: dict[str, Any] | None = None,
        endpoint_url: str | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int | None = None,
        table: Any | None = None,
    ) -> None:
        uri = uri or path
        if uri is None and table is None:
            raise ValueError("uri or table must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._uri = uri
        self._storage_options = dict(storage_options or {})
        if endpoint_url:
            self._storage_options.setdefault("AWS_ENDPOINT_URL", endpoint_url)
        self._endpoint_url = endpoint_url
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._table = table

    def load(self) -> list[Document]:
        table = self._table if self._table is not None else self._load_table()
        rows: list[Any]
        dataset_method = getattr(table, "to_pyarrow_dataset", None)
        if self._limit is not None and callable(dataset_method):
            dataset = dataset_method()
            scanner_method = getattr(dataset, "scanner", None)
            scanner = scanner_method() if callable(scanner_method) else dataset
            batches_method = getattr(scanner, "to_batches", None)
            if callable(batches_method):
                rows = []
                for batch in batches_method():
                    batch_rows = batch.to_pylist() if hasattr(batch, "to_pylist") else list(batch)
                    rows.extend(batch_rows)
                    if len(rows) >= self._limit:
                        break
                rows = rows[: self._limit]
            else:
                arrow_table = dataset.to_table()
                rows = arrow_table.to_pylist()[: self._limit]
        else:
            arrow_table = table.to_pyarrow_table()
            rows = arrow_table.to_pylist()
            if self._limit is not None:
                rows = rows[: self._limit]
        return records_to_documents(
            rows,
            "delta_lake",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            uri=self._uri,
            endpoint_url=self._endpoint_url,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _load_table(self) -> Any:
        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError("deltalake required: pip install synapsekit[delta]") from None
        return DeltaTable(self._uri, storage_options=self._storage_options or None)


DeltaLoader = DeltaLakeLoader
