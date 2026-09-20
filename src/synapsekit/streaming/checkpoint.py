"""Durable checkpoints for exactly-once-ish streaming ingestion."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .types import Checkpoint, StreamEvent, _coerce_datetime


class CheckpointStore(Protocol):
    """Async persistence contract for processed events and source positions."""

    async def load(self, stream: str, partition: str = "default") -> Checkpoint | None:
        """Return the latest checkpoint for a stream partition."""
        ...

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        ...

    async def is_processed(self, stream: str, event_id: str) -> bool:
        """Return whether an event was durably applied."""
        ...

    async def commit_event(self, event: StreamEvent) -> None:
        """Atomically record an applied event and advance its position."""
        ...


def _offset_parts(offset: int | str | None) -> tuple[str | None, bool]:
    if offset is None:
        return None, False
    return str(offset), isinstance(offset, int) and not isinstance(offset, bool)


def _offset_after(new: int | str | None, old: int | str | None) -> bool:
    """Compare source positions without ordering unrelated string offsets."""

    if old is None:
        return new is not None
    if new is None:
        return False
    if isinstance(new, int) and isinstance(old, int):
        return new > old
    if isinstance(new, str) and isinstance(old, str) and new.isdigit() and old.isdigit():
        return int(new) > int(old)
    return str(new) != str(old)


def _checkpoint_from_row(row: sqlite3.Row | tuple[Any, ...]) -> Checkpoint:
    if isinstance(row, sqlite3.Row):
        stream = str(row["stream"])
        partition = str(row["partition_key"])
        offset_raw = row["offset"]
        offset_is_int = bool(row["offset_is_int"])
        event_id = str(row["event_id"])
        updated_at_raw = row["updated_at"]
    else:
        stream, partition, offset_raw, offset_is_int, event_id, updated_at_raw = row
        stream = str(stream)
        partition = str(partition)
        event_id = str(event_id)
    offset: int | str | None
    if offset_raw is None:
        offset = None
    elif bool(offset_is_int):
        offset = int(offset_raw)
    else:
        offset = str(offset_raw)
    updated_at = _coerce_datetime(str(updated_at_raw)) or datetime.now(UTC)
    return Checkpoint(
        stream=stream,
        partition=partition,
        offset=offset,
        event_id=event_id,
        updated_at=updated_at,
    )


class InMemoryCheckpointStore:
    """Process-local checkpoint store useful for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], Checkpoint] = {}
        self._processed: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    async def load(self, stream: str, partition: str = "default") -> Checkpoint | None:
        with self._lock:
            return self._checkpoints.get((stream, partition))

    async def get(self, stream: str, partition: str = "default") -> Checkpoint | None:
        """Alias for ``load``."""

        return await self.load(stream, partition)

    async def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            key = (checkpoint.stream, checkpoint.partition)
            current = self._checkpoints.get(key)
            if (
                current is None
                or _offset_after(checkpoint.offset, current.offset)
                or (current.offset is None and checkpoint.offset is None)
            ):
                self._checkpoints[key] = checkpoint

    async def is_processed(self, stream: str, event_id: str) -> bool:
        with self._lock:
            return (stream, str(event_id)) in self._processed

    async def contains(self, stream: str, event_id: str) -> bool:
        """Alias for ``is_processed``."""

        return await self.is_processed(stream, event_id)

    async def commit_event(self, event: StreamEvent) -> None:
        checkpoint = Checkpoint.from_event(event)
        with self._lock:
            self._processed.add((event.stream_key, event.id))
            key = (checkpoint.stream, checkpoint.partition)
            current = self._checkpoints.get(key)
            if (
                current is None
                or _offset_after(checkpoint.offset, current.offset)
                or (current.offset is None and checkpoint.offset is None)
            ):
                self._checkpoints[key] = checkpoint

    async def close(self) -> None:
        """Match the closeable durable-store interface; there is no resource."""


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint and processed-event ledger.

    The write that marks an event processed and advances its partition position
    is one SQLite transaction. The target write is intentionally outside this
    transaction, so a crash between the two sides is safe only when the target
    upsert is idempotent; that is the exactly-once-ish contract of the ingestor.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stream_checkpoints (
                    stream TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    offset TEXT,
                    offset_is_int INTEGER NOT NULL DEFAULT 0,
                    event_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (stream, partition_key)
                );
                CREATE TABLE IF NOT EXISTS stream_processed_events (
                    stream TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    offset TEXT,
                    offset_is_int INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (stream, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_stream_processed_partition
                    ON stream_processed_events(stream, partition_key);
                """
            )

    def _load_sync(self, stream: str, partition: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT stream, partition_key, offset, offset_is_int, event_id, updated_at
                FROM stream_checkpoints
                WHERE stream = ? AND partition_key = ?
                """,
                (stream, partition),
            ).fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    async def load(self, stream: str, partition: str = "default") -> Checkpoint | None:
        return await asyncio.to_thread(self._load_sync, stream, partition)

    async def get(self, stream: str, partition: str = "default") -> Checkpoint | None:
        """Alias for ``load``."""

        return await self.load(stream, partition)

    def _save_sync(self, checkpoint: Checkpoint) -> None:
        offset, offset_is_int = _offset_parts(checkpoint.offset)
        now = checkpoint.updated_at.isoformat()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT offset, offset_is_int FROM stream_checkpoints "
                "WHERE stream = ? AND partition_key = ?",
                (checkpoint.stream, checkpoint.partition),
            ).fetchone()
            old_offset: int | str | None = None
            if row is not None and row["offset"] is not None:
                old_offset = int(row["offset"]) if row["offset_is_int"] else str(row["offset"])
            if (
                row is not None
                and not _offset_after(checkpoint.offset, old_offset)
                and not (checkpoint.offset is None and old_offset is None)
            ):
                return
            self._conn.execute(
                """
                INSERT INTO stream_checkpoints
                    (stream, partition_key, offset, offset_is_int, event_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stream, partition_key) DO UPDATE SET
                    offset = excluded.offset,
                    offset_is_int = excluded.offset_is_int,
                    event_id = excluded.event_id,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.stream,
                    checkpoint.partition,
                    offset,
                    int(offset_is_int),
                    checkpoint.event_id,
                    now,
                ),
            )

    async def save(self, checkpoint: Checkpoint) -> None:
        await asyncio.to_thread(self._save_sync, checkpoint)

    def _is_processed_sync(self, stream: str, event_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM stream_processed_events WHERE stream = ? AND event_id = ?",
                (stream, str(event_id)),
            ).fetchone()
        return row is not None

    async def is_processed(self, stream: str, event_id: str) -> bool:
        return await asyncio.to_thread(self._is_processed_sync, stream, event_id)

    async def contains(self, stream: str, event_id: str) -> bool:
        """Alias for ``is_processed``."""

        return await self.is_processed(stream, event_id)

    def _commit_event_sync(self, event: StreamEvent) -> None:
        offset, offset_is_int = _offset_parts(event.offset)
        now = datetime.now(UTC).isoformat()
        checkpoint = Checkpoint.from_event(event)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO stream_processed_events
                    (stream, event_id, partition_key, offset, offset_is_int, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.stream_key,
                    event.id,
                    event.partition_key,
                    offset,
                    int(offset_is_int),
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT offset, offset_is_int FROM stream_checkpoints "
                "WHERE stream = ? AND partition_key = ?",
                (checkpoint.stream, checkpoint.partition),
            ).fetchone()
            old_offset: int | str | None = None
            if row is not None and row["offset"] is not None:
                old_offset = int(row["offset"]) if row["offset_is_int"] else str(row["offset"])
            if (
                row is not None
                and not _offset_after(checkpoint.offset, old_offset)
                and not (checkpoint.offset is None and old_offset is None)
            ):
                return
            self._conn.execute(
                """
                INSERT INTO stream_checkpoints
                    (stream, partition_key, offset, offset_is_int, event_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stream, partition_key) DO UPDATE SET
                    offset = excluded.offset,
                    offset_is_int = excluded.offset_is_int,
                    event_id = excluded.event_id,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.stream,
                    checkpoint.partition,
                    offset,
                    int(offset_is_int),
                    event.id,
                    now,
                ),
            )

    async def commit_event(self, event: StreamEvent) -> None:
        await asyncio.to_thread(self._commit_event_sync, event)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
