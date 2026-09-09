"""Azure Cosmos DB-backed graph checkpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ..._json import dumps as _json_dumps
from ..._json import loads as _json_loads
from ...memory.backends._common import AsyncCheckpointerMixin, encode_document_id
from .base import BaseCheckpointer


def _is_not_found(exc: BaseException) -> bool:
    try:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
    except ImportError:
        return False
    return isinstance(exc, CosmosResourceNotFoundError)


class CosmosDBCheckpointer(AsyncCheckpointerMixin, BaseCheckpointer):
    """Persist one latest JSON checkpoint per graph in Cosmos DB."""

    def __init__(
        self,
        endpoint: str = "https://localhost:8081",
        key: str | None = None,
        database: str = "synapsekit",
        container: str = "graph_checkpoints",
        *,
        verify_ssl: bool = True,
        client: Any | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http(s) URL")
        if not database or not container:
            raise ValueError("database and container must be provided")
        self._endpoint = endpoint.rstrip("/") + "/"
        self._key = key
        self._database_name = database
        self._container_name = container
        self._verify_ssl = verify_ssl
        self._client = client
        self._container: Any | None = None

    def _ensure_container(self) -> Any:
        if self._container is not None:
            return self._container
        try:
            from azure.cosmos import PartitionKey
        except ImportError:
            raise ImportError(
                "azure-cosmos is required for CosmosDBCheckpointer; "
                "install it with `pip install synapsekit[cosmos]`"
            ) from None
        if self._client is None:
            if not self._key:
                raise ValueError("key must be provided for CosmosDBCheckpointer")
            try:
                from azure.cosmos import CosmosClient
            except ImportError:
                raise ImportError(
                    "azure-cosmos is required for CosmosDBCheckpointer; "
                    "install it with `pip install synapsekit[cosmos]`"
                ) from None
            self._client = CosmosClient(
                self._endpoint,
                credential=self._key,
                connection_verify=self._verify_ssl,
            )
        database = self._client.create_database_if_not_exists(id=self._database_name)
        self._container = database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=PartitionKey(path="/graph_id"),
        )
        return self._container

    @staticmethod
    def _key_for(graph_id: str) -> tuple[str, str]:
        return encode_document_id(graph_id), graph_id

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        item_id, _ = self._key_for(graph_id)
        self._ensure_container().upsert_item(
            {
                "id": item_id,
                "graph_id": graph_id,
                "step": int(step),
                "state": _json_dumps(state),
            }
        )

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        item_id, _ = self._key_for(graph_id)
        try:
            item = self._ensure_container().read_item(item=item_id, partition_key=graph_id)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return int(item["step"]), dict(_json_loads(item["state"]))

    def delete(self, graph_id: str) -> None:
        item_id, _ = self._key_for(graph_id)
        try:
            self._ensure_container().delete_item(item=item_id, partition_key=graph_id)
        except Exception as exc:
            if not _is_not_found(exc):
                raise

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._container = None
