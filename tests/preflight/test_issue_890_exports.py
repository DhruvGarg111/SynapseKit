from __future__ import annotations

import inspect

import pytest

MEMORY_BACKENDS = [
    ("MongoDBMemoryBackend", "mongodb"),
    ("CassandraMemoryBackend", "cassandra"),
    ("ScyllaDBMemoryBackend", "scylla"),
    ("DynamoDBMemoryBackend", "dynamodb"),
    ("FirestoreMemoryBackend", "firestore"),
    ("CosmosDBMemoryBackend", "cosmos"),
]

CHECKPOINTERS = [
    "MongoDBCheckpointer",
    "CassandraCheckpointer",
    "ScyllaDBCheckpointer",
    "DynamoDBCheckpointer",
    "FirestoreCheckpointer",
    "CosmosDBCheckpointer",
]


@pytest.mark.parametrize("name,backend_name", MEMORY_BACKENDS)
def test_memory_backend_is_exported_lazily(name: str, backend_name: str) -> None:
    import synapsekit
    import synapsekit.memory as memory

    backend_type = getattr(memory, name)
    assert getattr(synapsekit, name) is backend_type
    assert backend_name in {"mongodb", "cassandra", "scylla", "dynamodb", "firestore", "cosmos"}


@pytest.mark.parametrize("name", CHECKPOINTERS)
def test_checkpointer_is_exported_lazily_with_async_adapters(name: str) -> None:
    import synapsekit
    import synapsekit.graph as graph

    checkpointer_type = getattr(graph, name)
    assert getattr(synapsekit, name) is checkpointer_type
    assert inspect.iscoroutinefunction(checkpointer_type.asave)
    assert inspect.iscoroutinefunction(checkpointer_type.aload)
    assert inspect.iscoroutinefunction(checkpointer_type.adelete)
