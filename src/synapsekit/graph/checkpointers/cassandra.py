"""Cassandra-compatible graph checkpoints."""

from __future__ import annotations

from typing import Any

from ..._json import dumps_bytes as _json_dumps_bytes
from ..._json import loads as _json_loads
from ...memory.backends._common import AsyncCheckpointerMixin, validate_identifier
from .base import BaseCheckpointer


class CassandraCheckpointer(AsyncCheckpointerMixin, BaseCheckpointer):
    """Persist graph checkpoints in Cassandra or ScyllaDB."""

    def __init__(
        self,
        hosts: str | list[str] = "localhost",
        port: int = 9042,
        keyspace: str = "synapsekit",
        table: str = "graph_checkpoints",
        *,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._hosts = [hosts] if isinstance(hosts, str) else list(hosts)
        if not self._hosts:
            raise ValueError("hosts must contain at least one host")
        if port <= 0 or port > 65535:
            raise ValueError("port must be between 1 and 65535")
        self._port = port
        self._keyspace = validate_identifier(keyspace, "keyspace")
        self._table_name = validate_identifier(table, "table")
        self._username = username
        self._password = password
        self._cluster: Any | None = None
        self._session: Any | None = None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.cluster import Cluster
        except ImportError:
            raise ImportError(
                "cassandra-driver is required for CassandraCheckpointer; "
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
                graph_id text PRIMARY KEY,
                step int,
                state blob
            )
            """
        )
        self._session = session
        return session

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        session = self._ensure_session()
        session.execute(
            f"INSERT INTO {self._table_name} (graph_id, step, state) VALUES (%s, %s, %s)",
            (graph_id, int(step), _json_dumps_bytes(state)),
        )

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        session = self._ensure_session()
        row = session.execute(
            f"SELECT step, state FROM {self._table_name} WHERE graph_id = %s",
            (graph_id,),
        ).one()
        if row is None:
            return None
        state = row.state
        if not isinstance(state, (str, bytes)):
            state = bytes(state)
        return int(row.step), dict(_json_loads(state))

    def delete(self, graph_id: str) -> None:
        self._ensure_session().execute(
            f"DELETE FROM {self._table_name} WHERE graph_id = %s", (graph_id,)
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.shutdown()
        if self._cluster is not None:
            self._cluster.shutdown()
        self._session = None
        self._cluster = None


ScyllaCheckpointer = CassandraCheckpointer
