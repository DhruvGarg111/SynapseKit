"""MongoDB-backed graph checkpoints."""

from __future__ import annotations

import asyncio
from typing import Any

from ..._json import dumps as _json_dumps
from ..._json import loads as _json_loads
from ...memory.backends._common import AsyncCheckpointerMixin, encode_document_id
from .base import BaseCheckpointer


class MongoDBCheckpointer(AsyncCheckpointerMixin, BaseCheckpointer):
    """Persist one latest JSON checkpoint per graph in MongoDB."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "synapsekit",
        collection: str = "graph_checkpoints",
        *,
        client: Any | None = None,
        **client_options: Any,
    ) -> None:
        if not uri:
            raise ValueError("uri must be provided")
        if not database or not collection:
            raise ValueError("database and collection must be provided")
        self._uri = uri
        self._database_name = database
        self._collection_name = collection
        self._client = client
        self._client_options = client_options
        self._collection: Any | None = None
        self._init_lock = asyncio.Lock()

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client is None:
            try:
                from pymongo import MongoClient
            except ImportError:
                raise ImportError(
                    "pymongo is required for MongoDBCheckpointer; "
                    "install it with `pip install synapsekit[mongodb]`"
                ) from None
            self._client = MongoClient(self._uri, **self._client_options)
        self._collection = self._client[self._database_name][self._collection_name]
        return self._collection

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        collection = self._ensure_collection()
        collection.replace_one(
            {"_id": encode_document_id(graph_id)},
            {
                "_id": encode_document_id(graph_id),
                "graph_id": graph_id,
                "step": int(step),
                "state": _json_dumps(state),
            },
            upsert=True,
        )

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        document = self._ensure_collection().find_one({"_id": encode_document_id(graph_id)})
        if document is None:
            return None
        state = document["state"]
        if isinstance(state, (str, bytes)):
            state = _json_loads(state)
        return int(document["step"]), dict(state)

    def delete(self, graph_id: str) -> None:
        self._ensure_collection().delete_one({"_id": encode_document_id(graph_id)})

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
