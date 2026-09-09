"""Azure Cosmos DB-backed AgentMemory storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType
from ._common import memory_record_document_id
from ._serialization import memory_record_from_payload, memory_record_to_payload


def _is_not_found(exc: BaseException) -> bool:
    try:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
    except ImportError:
        return False
    return isinstance(exc, CosmosResourceNotFoundError)


class CosmosDBMemoryBackend(BaseMemoryBackend):
    """Persist memory records in an Azure Cosmos DB SQL container."""

    def __init__(
        self,
        endpoint: str = "https://localhost:8081",
        key: str | None = None,
        database: str = "synapsekit",
        container: str = "agent_memory",
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
        self._init_lock = asyncio.Lock()

    async def _ensure_container(self) -> Any:
        if self._container is not None:
            return self._container
        async with self._init_lock:
            if self._container is not None:
                return self._container
            try:
                from azure.cosmos import PartitionKey
            except ImportError:
                raise ImportError(
                    "azure-cosmos is required for CosmosDBMemoryBackend; "
                    "install it with `pip install synapsekit[cosmos]`"
                ) from None
            if self._client is None:
                if not self._key:
                    raise ValueError("key must be provided for CosmosDBMemoryBackend")
                try:
                    from azure.cosmos.aio import CosmosClient
                except ImportError:
                    raise ImportError(
                        "azure-cosmos is required for CosmosDBMemoryBackend; "
                        "install it with `pip install synapsekit[cosmos]`"
                    ) from None
                self._client = CosmosClient(
                    self._endpoint,
                    credential=self._key,
                    connection_verify=self._verify_ssl,
                )
            database = await self._client.create_database_if_not_exists(id=self._database_name)
            self._container = await database.create_container_if_not_exists(
                id=self._container_name,
                partition_key=PartitionKey(path="/agent_id"),
            )
            return self._container

    @staticmethod
    def _to_item(record: MemoryRecord) -> dict[str, Any]:
        payload = memory_record_to_payload(record)
        return {
            "id": memory_record_document_id(record.agent_id, record.id),
            "record_id": record.id,
            "agent_id": record.agent_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "embedding": list(record.embedding),
            "embedding_dimension": payload["embedding_dimension"],
            "created_at": payload["created_at"],
            "accessed_at": payload["accessed_at"],
            "access_count": payload["access_count"],
            "ttl_days": record.ttl_days,
            "metadata": payload["metadata"],
        }

    @staticmethod
    def _from_item(item: dict[str, Any]) -> MemoryRecord:
        payload = dict(item)
        payload["id"] = item["record_id"]
        return memory_record_from_payload(payload)

    async def store(self, record: MemoryRecord) -> None:
        container = await self._ensure_container()
        await container.upsert_item(self._to_item(record))

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        container = await self._ensure_container()
        query = "SELECT * FROM c WHERE c.agent_id = @agent_id"
        parameters = [{"name": "@agent_id", "value": agent_id}]
        records = [
            self._from_item(item)
            async for item in container.query_items(
                query=query, parameters=parameters, partition_key=agent_id
            )
        ]
        records.sort(key=lambda record: record.created_at)
        if memory_type is not None:
            records = [record for record in records if record.memory_type == memory_type]
        if include_expired:
            return records
        now = datetime.now(timezone.utc)
        return [record for record in records if not record.is_expired(now)]

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        container = await self._ensure_container()
        try:
            item = await container.read_item(
                item=memory_record_document_id(agent_id, record_id), partition_key=agent_id
            )
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            return
        timestamp = (accessed_at or datetime.now(timezone.utc)).isoformat()
        item["accessed_at"] = timestamp
        item["access_count"] = int(item.get("access_count", 0)) + 1
        await container.replace_item(item=memory_record_document_id(agent_id, record_id), body=item)

    async def delete(self, agent_id: str, record_id: str) -> bool:
        container = await self._ensure_container()
        try:
            await container.delete_item(
                item=memory_record_document_id(agent_id, record_id), partition_key=agent_id
            )
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        container = await self._ensure_container()
        records = await self.fetch(agent_id, memory_type=memory_type, include_expired=True)
        for record in records:
            await container.delete_item(
                item=memory_record_document_id(record.agent_id, record.id), partition_key=agent_id
            )
        return len(records)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        records = await self.fetch(agent_id, memory_type=memory_type, include_expired=True)
        return len(records)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        container = await self._ensure_container()
        current = now or datetime.now(timezone.utc)
        items = [
            item
            async for item in container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            )
        ]
        expired = [
            self._from_item(item) for item in items if self._from_item(item).is_expired(current)
        ]
        for record in expired:
            await container.delete_item(
                item=memory_record_document_id(record.agent_id, record.id),
                partition_key=record.agent_id,
            )
        return len(expired)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._container = None
