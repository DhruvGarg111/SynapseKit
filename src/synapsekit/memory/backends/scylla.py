"""ScyllaDB compatibility aliases for the Cassandra memory backend."""

from .cassandra import CassandraMemoryBackend

ScyllaMemoryBackend = CassandraMemoryBackend
ScyllaDBMemoryBackend = CassandraMemoryBackend

__all__ = ["ScyllaMemoryBackend", "ScyllaDBMemoryBackend"]
