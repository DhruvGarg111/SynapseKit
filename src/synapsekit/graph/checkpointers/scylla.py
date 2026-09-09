"""ScyllaDB compatibility aliases for the Cassandra checkpointer."""

from .cassandra import CassandraCheckpointer

ScyllaCheckpointer = CassandraCheckpointer
ScyllaDBCheckpointer = CassandraCheckpointer

__all__ = ["ScyllaCheckpointer", "ScyllaDBCheckpointer"]
