from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from synapsekit.streaming import (
    Checkpoint,
    InMemoryCheckpointStore,
    KafkaSource,
    KinesisSource,
    PostgresCDCSource,
    PulsarSource,
    RedpandaSource,
    normalize_debezium_event,
)


class FakeKafkaConsumer:
    def __init__(self, records: list[object]) -> None:
        self.records = records
        self.commits: list[object] = []
        self.closed = False

    def poll(self, timeout_ms: int, max_records: int) -> dict[str, list[object]]:
        if not self.records:
            return {}
        record = self.records.pop(0)
        return {"partition": [record]}

    def commit(self, **kwargs: object) -> None:
        self.commits.append(kwargs)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_kafka_source_normalizes_json_and_commits_after_ack() -> None:
    record = SimpleNamespace(
        topic="updates",
        partition=2,
        offset=7,
        value=b'{"text":"hello"}',
        key=b"order-1",
        headers=[("kind", b"created")],
        timestamp=1_757_000_000_000,
    )
    consumer = FakeKafkaConsumer([record])
    source = KafkaSource(
        topic="updates",
        bootstrap_servers="broker:9092",
        consumer=consumer,
        poll_timeout=0.01,
    )

    await source.start()
    event = await source.__anext__()
    await source.ack(event)
    await source.stop()

    assert event.source == "kafka"
    assert event.stream == "updates"
    assert event.event_id == "updates:2:7"
    assert event.payload == {"text": "hello"}
    assert event.headers == {"kind": "created"}
    assert event.key == "order-1"
    assert len(consumer.commits) == 1
    assert consumer.closed is True


@pytest.mark.asyncio
async def test_redpanda_uses_the_kafka_protocol_with_distinct_source_name() -> None:
    record = SimpleNamespace(
        topic="updates",
        partition=0,
        offset=1,
        value=b"hello",
        key=None,
        headers=[],
        timestamp=None,
    )
    source = RedpandaSource(topic="updates", consumer=FakeKafkaConsumer([record]))

    await source.start()
    event = await source.__anext__()

    assert event.source == "redpanda"
    assert event.payload == "hello"
    await source.stop()


class FakePulsarMessage:
    def data(self) -> bytes:
        return b'{"content":"from pulsar"}'

    def message_id(self) -> str:
        return "ledger:entry"

    def publish_timestamp(self) -> int:
        return 1_757_000_000_000

    def partition_key(self) -> str:
        return "account-1"


class FakePulsarConsumer:
    def __init__(self) -> None:
        self.acks: list[object] = []
        self.closed = False
        self._messages = [FakePulsarMessage()]

    def receive(self, timeout_millis: int) -> object | None:
        return self._messages.pop(0) if self._messages else None

    def acknowledge(self, message: object) -> None:
        self.acks.append(message)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pulsar_source_acknowledges_native_message() -> None:
    consumer = FakePulsarConsumer()
    source = PulsarSource(topic="events", consumer=consumer, receive_timeout=0.01)

    await source.start()
    event = await source.__anext__()
    await source.ack(event)
    await source.stop()

    assert event.payload == {"content": "from pulsar"}
    assert event.event_id == "events:ledger:entry"
    assert event.partition == "account-1"
    assert consumer.acks
    assert consumer.closed is True


class FakeKinesisClient:
    def __init__(self) -> None:
        self.iterator_calls: list[dict[str, object]] = []
        self.records = [
            {
                "Data": b'{"text":"from kinesis"}',
                "SequenceNumber": "10",
                "PartitionKey": "pk",
            }
        ]

    def list_shards(self, **kwargs: object) -> dict[str, object]:
        return {"Shards": [{"ShardId": "shard-0"}]}

    def get_shard_iterator(self, **kwargs: object) -> dict[str, str]:
        self.iterator_calls.append(kwargs)
        return {"ShardIterator": "iterator-0"}

    def get_records(self, **kwargs: object) -> dict[str, object]:
        return {"Records": self.records, "NextShardIterator": "iterator-0"}

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_kinesis_source_resumes_after_checkpoint() -> None:
    client = FakeKinesisClient()
    checkpoints = InMemoryCheckpointStore()
    await checkpoints.save(Checkpoint("kinesis:orders", "shard-0", "9", "event-9"))
    source = KinesisSource(
        stream="orders",
        client=client,
        checkpoint_store=checkpoints,
        poll_interval=0,
    )

    await source.start()
    event = await source.__anext__()
    await source.stop()

    assert event.event_id == "orders:shard-0:10"
    assert event.payload == {"text": "from kinesis"}
    assert client.iterator_calls[0]["ShardIteratorType"] == "AFTER_SEQUENCE_NUMBER"
    assert client.iterator_calls[0]["StartingSequenceNumber"] == "9"


async def _one_event(event: object) -> AsyncIterator[object]:
    yield event


@pytest.mark.asyncio
async def test_debezium_normalization_uses_postgres_lsn_and_after_image() -> None:
    raw = {
        "key": {"id": 42},
        "value": {
            "payload": {
                "op": "u",
                "before": {"id": 42, "name": "old"},
                "after": {"id": 42, "name": "new"},
                "source": {"db": "app", "schema": "public", "table": "users", "lsn": 1234},
                "ts_ms": 1_757_000_000_000,
            }
        },
    }

    event = normalize_debezium_event(raw, stream="db.users")

    assert event.source == "postgres_cdc"
    assert event.stream == "db.users"
    assert event.event_id == "app:1234:users:42"
    assert event.payload["after"]["name"] == "new"
    assert event.metadata["operation"] == "u"
    assert event.metadata["table"] == "users"
    assert event.timestamp == datetime.fromtimestamp(1_757_000_000, UTC)


@pytest.mark.asyncio
async def test_postgres_cdc_source_wraps_async_debezium_messages() -> None:
    raw = {
        "payload": {
            "op": "c",
            "before": None,
            "after": {"id": 1, "body": "created"},
            "source": {"name": "app", "table": "items", "lsn": 99},
        }
    }
    source = PostgresCDCSource(_one_event(raw), stream="items")

    event = await source.__anext__()

    assert event.source == "postgres_cdc"
    assert event.event_id == "app:99:items:1"
    assert event.metadata["operation"] == "c"


class FakeReplicationCursor:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.feedback: list[dict[str, object]] = []
        self.messages = [
            SimpleNamespace(
                data_start=123,
                payload={
                    "op": "u",
                    "after": {"id": 3},
                    "source": {"table": "items"},
                },
            )
        ]

    def start_replication(self, **kwargs: object) -> None:
        self.started.append(kwargs)

    def read_message(self) -> object | None:
        return self.messages.pop(0) if self.messages else None

    def send_feedback(self, **kwargs: object) -> None:
        self.feedback.append(kwargs)


class FakeReplicationConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeReplicationCursor()
        self.closed = False

    def cursor(self) -> FakeReplicationCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_replication_feedback_waits_for_target_ack() -> None:
    connection = FakeReplicationConnection()
    source = PostgresCDCSource(connection=connection, stream="items")

    event = await source.__anext__()

    assert connection.cursor_instance.feedback == []
    await source.ack(event)
    await source.stop()

    assert connection.cursor_instance.feedback == [{"flush_lsn": 123}]
    assert connection.closed is True
