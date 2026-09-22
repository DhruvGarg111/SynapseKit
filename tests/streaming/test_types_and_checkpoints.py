from __future__ import annotations

from datetime import UTC, datetime

import pytest

from synapsekit.streaming import (
    Checkpoint,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
    StreamEvent,
)


def test_stream_event_exposes_stable_identity_and_partition_key() -> None:
    event = StreamEvent(
        payload={"text": "hello"},
        source="kafka",
        stream="updates",
        event_id="updates-0-7",
        partition=0,
        offset=7,
        timestamp="2026-09-16T12:00:00Z",
    )

    assert event.id == "updates-0-7"
    assert event.stable_id == "updates-0-7"
    assert event.data == {"text": "hello"}
    assert event.value == {"text": "hello"}
    assert event.event_time == event.timestamp
    assert event.ingest_time is not None
    assert event.partition_key == "0"
    assert event.stream_key == "kafka:updates"
    assert event.timestamp == datetime(2026, 9, 16, 12, tzinfo=UTC)


def test_stream_event_derives_deterministic_id_when_source_has_no_id() -> None:
    first = StreamEvent(payload="hello", source="test", stream="events", offset=3)
    second = StreamEvent(payload="hello", source="test", stream="events", offset=3)

    assert first.event_id == second.event_id
    assert first.event_id


@pytest.mark.asyncio
async def test_memory_checkpoint_commits_event_and_is_idempotent() -> None:
    store = InMemoryCheckpointStore()
    event = StreamEvent(
        payload="hello",
        source="test",
        stream="events",
        event_id="event-1",
        partition="p0",
        offset=4,
    )

    await store.commit_event(event)
    await store.commit_event(event)

    checkpoint = await store.load(event.stream_key, event.partition_key)
    assert checkpoint == Checkpoint.from_event(event)
    assert await store.is_processed(event.stream_key, event.event_id)


@pytest.mark.asyncio
async def test_sqlite_checkpoint_survives_close_and_reopen(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    event = StreamEvent(
        payload={"id": 42},
        source="kinesis",
        stream="orders",
        event_id="shard-0:42",
        partition="shard-0",
        offset="42",
    )

    first = SQLiteCheckpointStore(path)
    await first.commit_event(event)
    first.close()

    second = SQLiteCheckpointStore(path)
    checkpoint = await second.load(event.stream_key, event.partition_key)
    assert checkpoint is not None
    assert checkpoint.event_id == event.event_id
    assert checkpoint.offset == event.offset
    assert await second.is_processed(event.stream_key, event.event_id)
    await second.commit_event(
        StreamEvent(
            payload={"id": 41},
            source="kinesis",
            stream="orders",
            event_id="shard-0:41",
            partition="shard-0",
            offset="41",
        )
    )
    assert (await second.load(event.stream_key, event.partition_key)).offset == "42"
    second.close()
