from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from synapsekit.graph.checkpointers import (
    CassandraCheckpointer,
    CosmosDBCheckpointer,
    DynamoDBCheckpointer,
    FirestoreCheckpointer,
    MongoDBCheckpointer,
)
from synapsekit.memory import (
    AgentMemory,
    CassandraMemoryBackend,
    CosmosDBMemoryBackend,
    DynamoDBMemoryBackend,
    FirestoreMemoryBackend,
    MemoryRecord,
    MongoDBMemoryBackend,
)
from synapsekit.memory.backends._serialization import (
    memory_record_from_payload,
    memory_record_to_payload,
)


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("mongodb", MongoDBMemoryBackend),
        ("cassandra", CassandraMemoryBackend),
        ("dynamodb", DynamoDBMemoryBackend),
        ("firestore", FirestoreMemoryBackend),
        ("cosmos", CosmosDBMemoryBackend),
    ],
)
def test_agent_memory_selects_new_backend_without_connecting(
    backend_name: str, backend_type: type[object]
) -> None:
    memory = AgentMemory(backend=backend_name, backend_options={})

    assert isinstance(memory._backend, backend_type)


@pytest.mark.parametrize(
    "checkpointer_type",
    [
        MongoDBCheckpointer,
        CassandraCheckpointer,
        DynamoDBCheckpointer,
        FirestoreCheckpointer,
        CosmosDBCheckpointer,
    ],
)
def test_new_checkpointers_expose_async_adapters(checkpointer_type: type[object]) -> None:
    assert callable(checkpointer_type.asave)
    assert callable(checkpointer_type.aload)
    assert callable(checkpointer_type.adelete)


def test_memory_record_serialization_preserves_json_and_embedding_values() -> None:
    created_at = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)
    record = MemoryRecord(
        id="record/one",
        agent_id="agent/one",
        content="remember this",
        memory_type="semantic",
        embedding=[0.25, -1.5, 3.0],
        created_at=created_at,
        accessed_at=created_at,
        access_count=4,
        ttl_days=7,
        metadata={"nested": {"value": True}, "tags": ["a", "b"]},
    )

    restored = memory_record_from_payload(memory_record_to_payload(record))

    assert restored == record


def test_embedding_dimension_mismatch_is_rejected() -> None:
    payload = memory_record_to_payload(
        MemoryRecord(
            id="record",
            agent_id="agent",
            content="content",
            memory_type="episodic",
            embedding=[1.0, 2.0],
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
        )
    )
    payload["embedding_dimension"] = 3

    with pytest.raises(ValueError, match="dimension"):
        memory_record_from_payload(payload)


def test_dynamodb_numeric_wrappers_are_normalized() -> None:
    payload = memory_record_to_payload(
        MemoryRecord(
            id="record",
            agent_id="agent",
            content="content",
            memory_type="episodic",
            embedding=[1.0, 2.0],
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            access_count=2,
            ttl_days=3,
        )
    )
    payload["ttl_days"] = Decimal("3")
    payload["access_count"] = Decimal("2")

    restored = memory_record_from_payload(payload)

    assert restored.ttl_days == 3
    assert restored.access_count == 2


@pytest.mark.parametrize(
    "backend_type",
    [
        MongoDBMemoryBackend,
        DynamoDBMemoryBackend,
        FirestoreMemoryBackend,
        CosmosDBMemoryBackend,
    ],
)
def test_provider_memory_payload_round_trip(backend_type: type[object]) -> None:
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    record = MemoryRecord(
        id="record/with-special-id",
        agent_id="agent",
        content="content",
        memory_type="episodic",
        embedding=[1.0, -2.0],
        created_at=now,
        accessed_at=now,
        access_count=2,
        ttl_days=3,
        metadata={"nested": {"ok": True}},
    )

    if backend_type in (MongoDBMemoryBackend, FirestoreMemoryBackend):
        payload = backend_type._to_document(record)
    else:
        payload = backend_type._to_item(record)
    if backend_type in (MongoDBMemoryBackend, FirestoreMemoryBackend):
        restored = backend_type._from_document(payload)
    else:
        restored = backend_type._from_item(payload)

    assert restored == record


def test_cassandra_payload_round_trip_accepts_driver_row_shape() -> None:
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    record = MemoryRecord(
        id="cassandra-record",
        agent_id="agent",
        content="content",
        memory_type="semantic",
        embedding=[0.5, 0.25],
        created_at=now,
        accessed_at=now,
        metadata={"source": "test"},
    )
    payload = memory_record_to_payload(record)
    backend = CassandraMemoryBackend()
    row = SimpleNamespace(**payload)

    assert backend._row_to_record(row) == record


def test_core_import_does_not_import_optional_backend_sdks() -> None:
    code = """
import sys
import synapsekit
for name in ("pymongo", "cassandra", "boto3", "google.cloud.firestore", "azure.cosmos"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
