from .graph import GraphAgentMemory, GraphMemoryBackend
from .memory import InMemoryMemoryBackend
from .postgres import PostgresMemoryBackend
from .redis import RedisMemoryBackend
from .sqlite import SQLiteMemoryBackend

__all__ = [
    "GraphAgentMemory",
    "GraphMemoryBackend",
    "InMemoryMemoryBackend",
    "SQLiteMemoryBackend",
    "RedisMemoryBackend",
    "PostgresMemoryBackend",
    "MongoDBMemoryBackend",
    "CassandraMemoryBackend",
    "ScyllaMemoryBackend",
    "ScyllaDBMemoryBackend",
    "DynamoDBMemoryBackend",
    "FirestoreMemoryBackend",
    "CosmosDBMemoryBackend",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    _lazy = {
        "MongoDBMemoryBackend": "mongodb",
        "CassandraMemoryBackend": "cassandra",
        "ScyllaMemoryBackend": "cassandra",
        "ScyllaDBMemoryBackend": "scylla",
        "DynamoDBMemoryBackend": "dynamodb",
        "FirestoreMemoryBackend": "firestore",
        "CosmosDBMemoryBackend": "cosmos",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(f".{_lazy[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
