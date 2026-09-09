"""DynamoDB-backed AgentMemory storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType
from ._serialization import memory_record_from_payload, memory_record_to_payload


class DynamoDBMemoryBackend(BaseMemoryBackend):
    """Persist memory records in an AWS DynamoDB table or DynamoDB Local.

    The table uses ``pk`` (agent ID) and ``sk`` (``memory#`` + record ID).
    Boto3 is imported lazily and all blocking calls run in worker threads.
    """

    def __init__(
        self,
        table_name: str = "synapsekit_memory",
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
        *,
        auto_create_table: bool = True,
        **client_options: Any,
    ) -> None:
        if not table_name:
            raise ValueError("table_name must be provided")
        self._table_name = table_name
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._auto_create_table = auto_create_table
        self._client_options = client_options
        self._resource: Any | None = None
        self._table: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_table(self) -> Any:
        if self._table is not None:
            return self._table
        async with self._init_lock:
            if self._table is None:
                await asyncio.to_thread(self._initialize_sync)
        return self._table

    def _initialize_sync(self) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for DynamoDBMemoryBackend; "
                "install it with `pip install synapsekit[dynamodb]`"
            ) from None
        options = dict(self._client_options)
        options.update(region_name=self._region_name, endpoint_url=self._endpoint_url)
        self._resource = boto3.resource("dynamodb", **options)
        table = self._resource.Table(self._table_name)
        if self._auto_create_table:
            try:
                table.meta.client.describe_table(TableName=self._table_name)
            except table.meta.client.exceptions.ResourceNotFoundException:
                table = self._resource.create_table(
                    TableName=self._table_name,
                    KeySchema=[
                        {"AttributeName": "pk", "KeyType": "HASH"},
                        {"AttributeName": "sk", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "pk", "AttributeType": "S"},
                        {"AttributeName": "sk", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                table.meta.client.get_waiter("table_exists").wait(TableName=self._table_name)
        self._table = table

    @staticmethod
    def _to_item(record: MemoryRecord) -> dict[str, Any]:
        payload = memory_record_to_payload(record)
        return {
            "pk": record.agent_id,
            "sk": f"memory#{record.id}",
            "record_id": record.id,
            "agent_id": record.agent_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "embedding": payload["embedding"],
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

    def _query_sync(self, agent_id: str) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Key

        table = self._table
        assert table is not None
        items: list[dict[str, Any]] = []
        request: dict[str, Any] = {
            "KeyConditionExpression": Key("pk").eq(agent_id) & Key("sk").begins_with("memory#")
        }
        while True:
            response = table.query(**request)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return items
            request["ExclusiveStartKey"] = last_key

    def _scan_sync(self) -> list[dict[str, Any]]:
        table = self._table
        assert table is not None
        items: list[dict[str, Any]] = []
        request: dict[str, Any] = {}
        while True:
            response = table.scan(**request)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return items
            request["ExclusiveStartKey"] = last_key

    async def store(self, record: MemoryRecord) -> None:
        table = await self._ensure_table()
        await asyncio.to_thread(table.put_item, Item=self._to_item(record))

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        await self._ensure_table()
        items = await asyncio.to_thread(self._query_sync, agent_id)
        records = [self._from_item(item) for item in items]
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
        table = await self._ensure_table()
        key = {"pk": agent_id, "sk": f"memory#{record_id}"}
        existing = await asyncio.to_thread(table.get_item, Key=key)
        if not existing.get("Item"):
            return
        timestamp = (accessed_at or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            table.update_item,
            Key=key,
            UpdateExpression="SET accessed_at = :accessed_at ADD access_count :increment",
            ExpressionAttributeValues={":accessed_at": timestamp, ":increment": 1},
        )

    async def delete(self, agent_id: str, record_id: str) -> bool:
        table = await self._ensure_table()
        response = await asyncio.to_thread(
            table.delete_item,
            Key={"pk": agent_id, "sk": f"memory#{record_id}"},
            ReturnValues="ALL_OLD",
        )
        return bool(response.get("Attributes"))

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        table = await self._ensure_table()
        items = await asyncio.to_thread(self._query_sync, agent_id)
        selected = [
            item for item in items if memory_type is None or item.get("memory_type") == memory_type
        ]
        for item in selected:
            await asyncio.to_thread(table.delete_item, Key={"pk": item["pk"], "sk": item["sk"]})
        return len(selected)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        await self._ensure_table()
        items = await asyncio.to_thread(self._query_sync, agent_id)
        if memory_type is None:
            return len(items)
        return sum(item.get("memory_type") == memory_type for item in items)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        table = await self._ensure_table()
        current = now or datetime.now(timezone.utc)
        items = await asyncio.to_thread(self._scan_sync)
        expired = [
            item
            for item in items
            if item.get("record_id") and self._from_item(item).is_expired(current)
        ]
        for item in expired:
            await asyncio.to_thread(table.delete_item, Key={"pk": item["pk"], "sk": item["sk"]})
        return len(expired)

    async def aclose(self) -> None:
        if self._resource is not None:
            client = getattr(getattr(self._resource, "meta", None), "client", None)
            close = getattr(client, "close", None)
            if close is not None:
                await asyncio.to_thread(close)
        self._table = None
        self._resource = None
