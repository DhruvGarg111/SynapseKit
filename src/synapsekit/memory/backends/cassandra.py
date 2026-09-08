"""Cassandra and ScyllaDB-backed AgentMemory storage."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType
from ._common import validate_identifier
from ._serialization import memory_record_from_payload, memory_record_to_payload


class CassandraMemoryBackend(BaseMemoryBackend):
    """Persist memory records in Cassandra-compatible databases.

    The synchronous ``cassandra-driver`` is isolated in worker threads so all
    AgentMemory operations remain awaitable. The same backend works with
    ScyllaDB's Cassandra-compatible protocol.
    """

    def __init__(
        self,
        hosts: str | Sequence[str] = "localhost",
        port: int = 9042,
        keyspace: str = "synapsekit",
        table: str = "agent_memory",
        *,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        if isinstance(hosts, str):
            hosts = [hosts]
        if not hosts:
            raise ValueError("hosts must contain at least one host")
        if port <= 0 or port > 65535:
            raise ValueError("port must be between 1 and 65535")
        self._hosts = list(hosts)
        self._port = port
        self._keyspace = validate_identifier(keyspace, "keyspace")
        self._table_name = validate_identifier(table, "table")
        self._username = username
        self._password = password
        self._cluster: Any | None = None
        self._session: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        async with self._init_lock:
            if self._session is None:
                await asyncio.to_thread(self._initialize_sync)
        return self._session

    def _initialize_sync(self) -> None:
        try:
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.cluster import Cluster
        except ImportError:
            raise ImportError(
                "cassandra-driver is required for CassandraMemoryBackend; "
                "install it with `pip install synapsekit[cassandra]`"
            ) from None
        auth_provider = None
        if self._username is not None or self._password is not None:
            if not self._username or self._password is None:
                raise ValueError("username and password must be provided together")
            auth_provider = PlainTextAuthProvider(self._username, self._password)
        self._cluster = Cluster(
            contact_points=self._hosts,
            port=self._port,
            auth_provider=auth_provider,
        )
        session = self._cluster.connect()
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {self._keyspace} "
            "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}"
        )
        session.set_keyspace(self._keyspace)
        session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                agent_id text,
                id text,
                content text,
                memory_type text,
                embedding blob,
                embedding_dimension int,
                created_at text,
                accessed_at text,
                access_count int,
                ttl_days int,
                metadata text,
                PRIMARY KEY ((agent_id), id)
            )
            """
        )
        session.execute(
            "CREATE TABLE IF NOT EXISTS synapsekit_memory_agents (agent_id text PRIMARY KEY)"
        )
        self._session = session

    def _rows_sync(self, agent_id: str) -> list[Any]:
        session = self._session
        assert session is not None
        result = session.execute(
            f"SELECT * FROM {self._table_name} WHERE agent_id = %s", (agent_id,)
        )
        return list(result)

    def _row_to_record(self, row: Any) -> MemoryRecord:
        payload = {
            "id": row.id,
            "agent_id": row.agent_id,
            "content": row.content,
            "memory_type": row.memory_type,
            "embedding": row.embedding,
            "embedding_dimension": row.embedding_dimension,
            "created_at": row.created_at,
            "accessed_at": row.accessed_at,
            "access_count": row.access_count,
            "ttl_days": row.ttl_days,
            "metadata": row.metadata,
        }
        return memory_record_from_payload(payload)

    async def store(self, record: MemoryRecord) -> None:
        session = await self._ensure_session()
        payload = memory_record_to_payload(record)
        await asyncio.to_thread(
            session.execute,
            f"""INSERT INTO {self._table_name}
            (agent_id, id, content, memory_type, embedding, embedding_dimension,
             created_at, accessed_at, access_count, ttl_days, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                record.agent_id,
                record.id,
                record.content,
                record.memory_type,
                payload["embedding"],
                payload["embedding_dimension"],
                payload["created_at"],
                payload["accessed_at"],
                payload["access_count"],
                payload["ttl_days"],
                payload["metadata"],
            ),
        )
        await asyncio.to_thread(
            session.execute,
            "INSERT INTO synapsekit_memory_agents (agent_id) VALUES (%s)",
            (record.agent_id,),
        )

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        await self._ensure_session()
        rows = await asyncio.to_thread(self._rows_sync, agent_id)
        records = [self._row_to_record(row) for row in rows]
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
        session = await self._ensure_session()
        rows = await asyncio.to_thread(self._rows_sync, agent_id)
        record = next((self._row_to_record(row) for row in rows if row.id == record_id), None)
        if record is None:
            return
        timestamp = (accessed_at or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            session.execute,
            f"UPDATE {self._table_name} SET accessed_at = %s, access_count = %s "
            "WHERE agent_id = %s AND id = %s",
            (timestamp, record.access_count + 1, agent_id, record_id),
        )

    async def delete(self, agent_id: str, record_id: str) -> bool:
        session = await self._ensure_session()
        rows = await asyncio.to_thread(self._rows_sync, agent_id)
        if not any(row.id == record_id for row in rows):
            return False
        await asyncio.to_thread(
            session.execute,
            f"DELETE FROM {self._table_name} WHERE agent_id = %s AND id = %s",
            (agent_id, record_id),
        )
        return True

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        session = await self._ensure_session()
        rows = await asyncio.to_thread(self._rows_sync, agent_id)
        ids = [row.id for row in rows if memory_type is None or row.memory_type == memory_type]
        for record_id in ids:
            await asyncio.to_thread(
                session.execute,
                f"DELETE FROM {self._table_name} WHERE agent_id = %s AND id = %s",
                (agent_id, record_id),
            )
        return len(ids)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        await self._ensure_session()
        rows = await asyncio.to_thread(self._rows_sync, agent_id)
        if memory_type is None:
            return len(rows)
        return sum(row.memory_type == memory_type for row in rows)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        session = await self._ensure_session()
        current = now or datetime.now(timezone.utc)
        agent_rows = await asyncio.to_thread(
            lambda: list(session.execute("SELECT agent_id FROM synapsekit_memory_agents"))
        )
        removed = 0
        for agent_row in agent_rows:
            rows = await asyncio.to_thread(self._rows_sync, agent_row.agent_id)
            for row in rows:
                if self._row_to_record(row).is_expired(current):
                    await asyncio.to_thread(
                        session.execute,
                        f"DELETE FROM {self._table_name} WHERE agent_id = %s AND id = %s",
                        (row.agent_id, row.id),
                    )
                    removed += 1
        return removed

    async def aclose(self) -> None:
        if self._session is not None:
            await asyncio.to_thread(self._session.shutdown)
        if self._cluster is not None:
            await asyncio.to_thread(self._cluster.shutdown)
        self._session = None
        self._cluster = None


ScyllaMemoryBackend = CassandraMemoryBackend
