"""MongoDB-backed AgentMemory storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType
from ._common import memory_record_document_id
from ._serialization import memory_record_from_payload, memory_record_to_payload


class MongoDBMemoryBackend(BaseMemoryBackend):
    """Persist episodic and semantic memory records in MongoDB.

    ``pymongo`` is imported only when the first operation initializes the
    client. Install it with ``pip install synapsekit[mongodb]``.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "synapsekit",
        collection: str = "agent_memory",
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

    async def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        async with self._init_lock:
            if self._collection is None:
                await asyncio.to_thread(self._initialize_sync)
        return self._collection

    def _initialize_sync(self) -> None:
        if self._client is None:
            try:
                from pymongo import MongoClient
            except ImportError:
                raise ImportError(
                    "pymongo is required for MongoDBMemoryBackend; "
                    "install it with `pip install synapsekit[mongodb]`"
                ) from None
            self._client = MongoClient(self._uri, **self._client_options)
        self._collection = self._client[self._database_name][self._collection_name]
        self._collection.create_index([("agent_id", 1), ("created_at", 1)])
        self._collection.create_index([("agent_id", 1), ("memory_type", 1)])

    @staticmethod
    def _to_document(record: MemoryRecord) -> dict[str, Any]:
        payload = memory_record_to_payload(record)
        return {
            "_id": memory_record_document_id(record.agent_id, record.id),
            "record_id": record.id,
            "agent_id": record.agent_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "embedding": payload["embedding"],
            "embedding_dimension": payload["embedding_dimension"],
            "created_at": payload["created_at"],
            "accessed_at": payload["accessed_at"],
            "access_count": payload["access_count"],
            "ttl_days": payload["ttl_days"],
            "metadata": payload["metadata"],
        }

    @staticmethod
    def _from_document(document: dict[str, Any]) -> MemoryRecord:
        payload = dict(document)
        payload["id"] = document.get("record_id", document["_id"])
        return memory_record_from_payload(payload)

    async def store(self, record: MemoryRecord) -> None:
        collection = await self._ensure_collection()
        await asyncio.to_thread(
            collection.replace_one,
            {"_id": memory_record_document_id(record.agent_id, record.id)},
            self._to_document(record),
            True,
        )

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        collection = await self._ensure_collection()
        query: dict[str, Any] = {"agent_id": agent_id}
        if memory_type is not None:
            query["memory_type"] = memory_type
        documents = await asyncio.to_thread(
            lambda: list(collection.find(query).sort("created_at", 1))
        )
        records = [self._from_document(document) for document in documents]
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
        collection = await self._ensure_collection()
        timestamp = (accessed_at or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            collection.update_one,
            {"_id": memory_record_document_id(agent_id, record_id), "agent_id": agent_id},
            {"$set": {"accessed_at": timestamp}, "$inc": {"access_count": 1}},
        )

    async def delete(self, agent_id: str, record_id: str) -> bool:
        collection = await self._ensure_collection()
        result = await asyncio.to_thread(
            collection.delete_one,
            {"_id": memory_record_document_id(agent_id, record_id), "agent_id": agent_id},
        )
        return bool(result.deleted_count)

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        collection = await self._ensure_collection()
        query: dict[str, Any] = {"agent_id": agent_id}
        if memory_type is not None:
            query["memory_type"] = memory_type
        result = await asyncio.to_thread(collection.delete_many, query)
        return int(result.deleted_count)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        collection = await self._ensure_collection()
        query: dict[str, Any] = {"agent_id": agent_id}
        if memory_type is not None:
            query["memory_type"] = memory_type
        return int(await asyncio.to_thread(collection.count_documents, query))

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        collection = await self._ensure_collection()
        current = now or datetime.now(timezone.utc)
        documents = await asyncio.to_thread(
            lambda: list(collection.find({"ttl_days": {"$ne": None}}))
        )
        expired_ids = [
            document["_id"]
            for document in documents
            if self._from_document(document).is_expired(current)
        ]
        if not expired_ids:
            return 0
        result = await asyncio.to_thread(collection.delete_many, {"_id": {"$in": expired_ids}})
        return int(result.deleted_count)

    async def aclose(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._collection = None
