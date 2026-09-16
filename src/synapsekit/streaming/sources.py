"""Async adapters for Kafka-compatible, Pulsar, Kinesis, and CDC streams."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from .checkpoint import CheckpointStore
from .types import StreamEvent


class AsyncEventSource(Protocol):
    """Minimal lifecycle contract consumed by :class:`StreamingIngestor`."""

    def __aiter__(self) -> AsyncIterator[StreamEvent]: ...

    async def start(self) -> None: ...

    async def ack(self, event: StreamEvent) -> None: ...

    async def stop(self) -> None: ...


StreamSource = AsyncEventSource


async def _call_async_or_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a possibly-async function without blocking the event loop."""

    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    result = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _decode(value: Any) -> Any:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return value
    text = bytes(value).decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _decode_key(value: Any) -> Any:
    decoded = _decode(value)
    return decoded


def _headers(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _decode(item) for key, item in value.items()}
    return {str(key): _decode(item) for key, item in value}


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        # Kafka/Pulsar/Kinesis timestamps are milliseconds since the epoch.
        return datetime.fromtimestamp(value / 1000, UTC)
    return None


def _records_from_poll(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        records: list[Any] = []
        for values in result.values():
            if isinstance(values, (list, tuple, deque)):
                records.extend(values)
            elif values is not None:
                records.append(values)
        return records
    if isinstance(result, (list, tuple, deque)):
        return list(result)
    return [result]


class KafkaSource:
    """Consume a Kafka-compatible topic using ``kafka-python`` lazily.

    ``kafka-python`` is synchronous, so polling, commits, and shutdown are
    delegated to worker threads.  This keeps the public source async-first and
    also makes the adapter usable with Redpanda without a second client.
    """

    source_name = "kafka"

    def __init__(
        self,
        topic: str,
        *,
        bootstrap_servers: str | list[str] = "localhost:9092",
        group_id: str | None = None,
        auto_offset_reset: str = "earliest",
        poll_timeout: float = 1.0,
        max_records: int = 100,
        consumer: Any | None = None,
        checkpoint_namespace: str | None = None,
    ) -> None:
        if not topic:
            raise ValueError("topic must not be empty")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be positive")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.topic = topic
        self.stream = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id or f"synapsekit-{topic}"
        self.auto_offset_reset = auto_offset_reset
        self.poll_timeout = poll_timeout
        self.max_records = max_records
        self._consumer = consumer
        self.checkpoint_namespace = checkpoint_namespace
        self._pending: dict[str, Any] = {}
        self._buffer: deque[Any] = deque()
        self._stop_requested = False

    @property
    def _checkpoint_namespace(self) -> str:
        if self.checkpoint_namespace:
            return self.checkpoint_namespace
        servers = (
            ",".join(self.bootstrap_servers)
            if isinstance(self.bootstrap_servers, list)
            else self.bootstrap_servers
        )
        return f"{self.source_name}:{servers}:{self.group_id}:{self.topic}"

    async def start(self) -> None:
        self._stop_requested = False
        if self._consumer is not None:
            return

        def create_consumer() -> Any:
            try:
                from kafka import KafkaConsumer
            except ImportError as exc:
                raise ImportError(
                    "KafkaSource requires the optional 'kafka-python' package; "
                    "install synapsekit with its kafka extra"
                ) from exc
            return KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,
                auto_offset_reset=self.auto_offset_reset,
                consumer_timeout_ms=max(100, int(self.poll_timeout * 1000)),
            )

        self._consumer = await asyncio.to_thread(create_consumer)

    async def __anext__(self) -> StreamEvent:
        while not self._stop_requested:
            if not self._buffer:
                if self._consumer is None:
                    await self.start()
                assert self._consumer is not None
                poll = self._consumer.poll
                result = await _call_async_or_thread(
                    poll,
                    timeout_ms=max(1, int(self.poll_timeout * 1000)),
                    max_records=self.max_records,
                )
                self._buffer.extend(_records_from_poll(result))
            if self._buffer:
                record = self._buffer.popleft()
                topic = str(_value(record, "topic", self.topic))
                partition = _value(record, "partition", 0)
                offset = _value(record, "offset")
                event_id = f"{topic}:{partition}:{offset}"
                event = StreamEvent(
                    payload=_decode(_value(record, "value")),
                    source=self.source_name,
                    stream=topic,
                    event_id=event_id,
                    partition=partition,
                    offset=offset,
                    timestamp=_timestamp(_value(record, "timestamp")),
                    key=_decode_key(_value(record, "key")),
                    headers=_headers(_value(record, "headers")),
                    metadata={
                        "checkpoint_namespace": self._checkpoint_namespace,
                        "consumer_group": self.group_id,
                    },
                )
                self._pending[event.id] = record
                return event
        raise StopAsyncIteration

    def __aiter__(self) -> KafkaSource:
        return self

    async def ack(self, event: StreamEvent) -> None:
        if self._consumer is None:
            return
        commit = getattr(self._consumer, "commit", None)
        if commit is None:
            return
        record = self._pending.pop(event.id, None)
        if record is None or _value(record, "offset") is None:
            await _call_async_or_thread(commit)
            return
        try:
            from kafka import TopicPartition
            from kafka.structs import OffsetAndMetadata
        except ImportError:
            # Dependency-free injected consumers remain useful in unit tests.
            await _call_async_or_thread(commit)
            return
        topic = str(_value(record, "topic", self.topic))
        partition = int(_value(record, "partition", 0))
        offset = int(_value(record, "offset")) + 1
        offsets = {TopicPartition(topic, partition): OffsetAndMetadata(offset, None)}
        await _call_async_or_thread(commit, offsets=offsets)

    async def stop(self) -> None:
        self._stop_requested = True
        consumer, self._consumer = self._consumer, None
        self._buffer.clear()
        self._pending.clear()
        if consumer is not None:
            close = getattr(consumer, "close", None)
            if close is not None:
                await _call_async_or_thread(close)


class RedpandaSource(KafkaSource):
    """Kafka-compatible source with an explicit Redpanda source identity."""

    source_name = "redpanda"


class PulsarSource:
    """Consume a Pulsar topic through the optional ``pulsar-client`` package."""

    source_name = "pulsar"

    def __init__(
        self,
        topic: str,
        *,
        service_url: str = "pulsar://localhost:6650",
        subscription_name: str | None = None,
        receive_timeout: float = 1.0,
        consumer: Any | None = None,
        client: Any | None = None,
        checkpoint_namespace: str | None = None,
    ) -> None:
        if not topic:
            raise ValueError("topic must not be empty")
        if receive_timeout <= 0:
            raise ValueError("receive_timeout must be positive")
        self.topic = topic
        self.stream = topic
        self.service_url = service_url
        self.subscription_name = subscription_name or f"synapsekit-{topic}"
        self.receive_timeout = receive_timeout
        self._consumer = consumer
        self._client = client
        self.checkpoint_namespace = checkpoint_namespace
        self._pending: dict[str, Any] = {}
        self._stop_requested = False

    async def start(self) -> None:
        self._stop_requested = False
        if self._consumer is not None:
            return

        def create_consumer() -> tuple[Any, Any]:
            try:
                import pulsar
            except ImportError as exc:
                raise ImportError(
                    "PulsarSource requires the optional 'pulsar-client' package"
                ) from exc
            client = pulsar.Client(self.service_url)
            consumer = client.subscribe(
                self.topic,
                subscription_name=self.subscription_name,
            )
            return client, consumer

        self._client, self._consumer = await asyncio.to_thread(create_consumer)

    async def __anext__(self) -> StreamEvent:
        while not self._stop_requested:
            if self._consumer is None:
                await self.start()
            assert self._consumer is not None
            receive = self._consumer.receive
            try:
                message = await _call_async_or_thread(
                    receive,
                    timeout_millis=max(1, int(self.receive_timeout * 1000)),
                )
            except Exception as exc:
                if "timeout" in type(exc).__name__.lower():
                    continue
                raise
            if message is None:
                continue
            message_id = _value(message, "message_id")
            if callable(message_id):
                message_id = message_id()
            if message_id is None:
                message_id = "unknown"
            partition = _value(message, "partition_key")
            if callable(partition):
                partition = partition()
            event_id = f"{self.topic}:{message_id}"
            data = _value(message, "data")
            if callable(data):
                data = data()
            published = _value(message, "publish_timestamp")
            if callable(published):
                published = published()
            event = StreamEvent(
                payload=_decode(data),
                source=self.source_name,
                stream=self.topic,
                event_id=event_id,
                partition=partition,
                offset=str(message_id),
                timestamp=_timestamp(published),
                key=partition,
                metadata={
                    "checkpoint_namespace": self.checkpoint_namespace
                    or f"{self.source_name}:{self.service_url}:{self.subscription_name}:{self.topic}",
                    "subscription_name": self.subscription_name,
                },
            )
            self._pending[event.id] = message
            return event
        raise StopAsyncIteration

    def __aiter__(self) -> PulsarSource:
        return self

    async def ack(self, event: StreamEvent) -> None:
        message = self._pending.pop(event.id, None)
        acknowledge = getattr(self._consumer, "acknowledge", None)
        if message is not None and acknowledge is not None:
            await _call_async_or_thread(acknowledge, message)

    async def stop(self) -> None:
        self._stop_requested = True
        consumer, self._consumer = self._consumer, None
        client, self._client = self._client, None
        self._pending.clear()
        if consumer is not None:
            close = getattr(consumer, "close", None)
            if close is not None:
                await _call_async_or_thread(close)
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                await _call_async_or_thread(close)


class KinesisSource:
    """Poll Kinesis shards asynchronously through boto3 worker threads."""

    source_name = "kinesis"

    def __init__(
        self,
        stream: str,
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        shard_id: str | None = None,
        iterator_type: str = "TRIM_HORIZON",
        poll_interval: float = 1.0,
        max_records: int = 100,
        client: Any | None = None,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_namespace: str | None = None,
    ) -> None:
        if not stream:
            raise ValueError("stream must not be empty")
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.stream = stream
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.requested_shard_id = shard_id
        self.iterator_type = iterator_type
        self.poll_interval = poll_interval
        self.max_records = max_records
        self._client = client
        self._checkpoint_store = checkpoint_store
        self.checkpoint_namespace = checkpoint_namespace or f"{self.source_name}:{stream}"
        self._iterators: dict[str, str] = {}
        self._shards: list[str] = []
        self._buffer: deque[tuple[str, dict[str, Any]]] = deque()
        self._next_shard = 0
        self._stop_requested = False

    async def set_checkpoint_store(self, store: CheckpointStore) -> None:
        self._checkpoint_store = store

    async def start(self) -> None:
        self._stop_requested = False
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError("KinesisSource requires the optional 'boto3' package") from exc
            self._client = await asyncio.to_thread(
                boto3.client,
                "kinesis",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        assert self._client is not None
        if self.requested_shard_id is not None:
            shard_ids = [self.requested_shard_id]
        else:
            shard_ids = await self._list_shards()
        self._shards = shard_ids
        self._iterators.clear()
        self._buffer.clear()
        self._next_shard = 0
        for current_shard in shard_ids:
            checkpoint = None
            if self._checkpoint_store is not None:
                checkpoint = await self._checkpoint_store.load(
                    self.checkpoint_namespace,
                    current_shard,
                )
            kwargs: dict[str, Any] = {
                "StreamName": self.stream,
                "ShardId": current_shard,
            }
            if checkpoint is not None and checkpoint.offset is not None:
                kwargs.update(
                    {
                        "ShardIteratorType": "AFTER_SEQUENCE_NUMBER",
                        "StartingSequenceNumber": str(checkpoint.offset),
                    }
                )
            else:
                kwargs["ShardIteratorType"] = self.iterator_type
            response = await self._client_call("get_shard_iterator", **kwargs)
            iterator = response.get("ShardIterator")
            if iterator:
                self._iterators[current_shard] = str(iterator)

    async def _list_shards(self) -> list[str]:
        assert self._client is not None
        shard_ids: list[str] = []
        kwargs: dict[str, Any] = {"StreamName": self.stream}
        while True:
            response = await self._client_call("list_shards", **kwargs)
            shard_ids.extend(
                str(shard["ShardId"]) for shard in response.get("Shards", []) if "ShardId" in shard
            )
            token = response.get("NextToken")
            if not token:
                return shard_ids
            kwargs = {"NextToken": token}

    async def _client_call(self, method: str, **kwargs: Any) -> Any:
        assert self._client is not None
        return await _call_async_or_thread(getattr(self._client, method), **kwargs)

    async def __anext__(self) -> StreamEvent:
        while not self._stop_requested:
            if self._buffer:
                shard, record = self._buffer.popleft()
                sequence = str(record.get("SequenceNumber", ""))
                return StreamEvent(
                    payload=_decode(record.get("Data")),
                    source=self.source_name,
                    stream=self.stream,
                    event_id=f"{self.stream}:{shard}:{sequence}",
                    partition=shard,
                    offset=sequence,
                    timestamp=_timestamp(record.get("ApproximateArrivalTimestamp")),
                    key=_decode_key(record.get("PartitionKey")),
                    metadata={"checkpoint_namespace": self.checkpoint_namespace},
                )
            if not self._iterators:
                raise StopAsyncIteration
            ordered_shards = list(self._iterators)
            for _ in range(len(ordered_shards)):
                shard = ordered_shards[self._next_shard % len(ordered_shards)]
                self._next_shard += 1
                iterator = self._iterators.get(shard)
                if iterator is None:
                    continue
                response = await self._client_call(
                    "get_records",
                    ShardIterator=iterator,
                    Limit=self.max_records,
                )
                next_iterator = response.get("NextShardIterator")
                if next_iterator:
                    self._iterators[shard] = str(next_iterator)
                else:
                    self._iterators.pop(shard, None)
                records = response.get("Records", [])
                if records:
                    self._buffer.extend((shard, record) for record in records)
                    break
            if self.poll_interval:
                await asyncio.sleep(self.poll_interval)
        raise StopAsyncIteration

    def __aiter__(self) -> KinesisSource:
        return self

    async def ack(self, event: StreamEvent) -> None:
        # Kinesis has no consumer acknowledgement.  The checkpoint is the
        # durable acknowledgement and is written by StreamingIngestor.
        return None

    async def stop(self) -> None:
        self._stop_requested = True
        client, self._client = self._client, None
        self._iterators.clear()
        self._shards.clear()
        self._buffer.clear()
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                await _call_async_or_thread(close)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _debezium_parts(raw: Any) -> tuple[Any, Any, dict[str, Any], StreamEvent | None]:
    if isinstance(raw, StreamEvent):
        value = _decode(raw.payload)
        return raw.key, value, dict(raw.metadata), raw
    key = None
    metadata: dict[str, Any] = {}
    value = raw
    if isinstance(raw, Mapping) and "value" in raw:
        key = _decode(raw.get("key"))
        value = _decode(raw.get("value"))
        raw_metadata = raw.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.setdefault("native_partition", raw.get("partition"))
        metadata.setdefault("native_offset", raw.get("offset"))
    else:
        value = _decode(value)
    if isinstance(value, Mapping) and isinstance(value.get("payload"), Mapping):
        value = value["payload"]
    if value is None:
        value = {"after": None, "op": "tombstone"}
    elif not isinstance(value, Mapping):
        value = {"after": value, "op": "r"}
    return key, value, metadata, None


def normalize_debezium_event(
    raw: Any,
    *,
    stream: str = "default",
    source_name: str = "postgres_cdc",
) -> StreamEvent:
    """Convert a Debezium envelope into a stable :class:`StreamEvent`."""

    key, envelope, metadata, original = _debezium_parts(raw)
    source = envelope.get("source")
    source = source if isinstance(source, Mapping) else {}
    after = envelope.get("after")
    if key is None and isinstance(after, Mapping):
        key = {field: after[field] for field in ("id", "key") if field in after} or after
    table = str(source.get("table") or envelope.get("table") or "unknown")
    logical_name = str(source.get("name") or source.get("db") or source_name)
    lsn = source.get("lsn")
    if lsn is None:
        lsn = source.get("commit_lsn")
    transaction = envelope.get("transaction")
    if isinstance(transaction, Mapping):
        transaction_id = transaction.get("id")
    else:
        transaction_id = transaction
    identity = key if key is not None else after
    if isinstance(identity, Mapping) and len(identity) == 1:
        identity_text = str(next(iter(identity.values())))
    else:
        identity_text = _canonical(identity)
    document_id = f"{table}:{identity_text}"
    sequence = source.get("sequence")
    if sequence is None:
        sequence = source.get("total_order")
    position = next(
        (
            candidate
            for candidate in (lsn, sequence, transaction_id, envelope.get("ts_ms"))
            if candidate is not None
        ),
        None,
    )
    event_id = f"{logical_name}:{position}:{table}:{_canonical(identity)}"
    if isinstance(identity, Mapping) and len(identity) == 1:
        event_id = f"{logical_name}:{position}:{table}:{next(iter(identity.values()))}"
    operation = str(envelope.get("op") or "r")
    event_metadata = {
        **metadata,
        "operation": operation,
        "table": table,
        "document_id": document_id,
        "deleted": operation in {"d", "tombstone"},
        "source": f"cdc://{logical_name}/{document_id}",
    }
    if source.get("schema") is not None:
        event_metadata["schema"] = source["schema"]
    if source.get("db") is not None:
        event_metadata["database"] = source["db"]
    if lsn is not None:
        event_metadata["lsn"] = lsn
    if transaction_id is not None:
        event_metadata["transaction"] = transaction_id
    timestamp = _timestamp(envelope.get("ts_ms"))
    if timestamp is None:
        timestamp = _timestamp(source.get("ts_ms"))
    original_stream = original.stream if original is not None else stream
    resolved_stream = original_stream if stream == "default" else stream
    event_metadata.setdefault(
        "checkpoint_namespace",
        f"{source_name}:{resolved_stream}",
    )
    event_metadata.setdefault("path", event_metadata["source"])
    event_metadata.setdefault("chunk_id", document_id)
    original_partition = (
        original.partition if original is not None else metadata.get("native_partition")
    )
    original_offset = (
        original.offset
        if original is not None
        else metadata.get("native_offset")
        if metadata.get("native_offset") is not None
        else lsn
    )
    original_headers = original.headers if original is not None else {}
    return StreamEvent(
        payload=dict(envelope),
        source=source_name,
        stream=resolved_stream,
        event_id=event_id,
        partition=original_partition,
        offset=original_offset,
        timestamp=timestamp,
        key=key,
        headers=original_headers,
        metadata=event_metadata,
    )


_SENTINEL = object()


async def _sync_to_async(values: Iterable[Any]) -> AsyncIterator[Any]:
    iterator = iter(values)
    while True:
        value = await asyncio.to_thread(_next_or_sentinel, iterator)
        if value is _SENTINEL:
            return
        yield value


def _next_or_sentinel(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


class DebeziumSource:
    """Normalize records from Kafka or another async iterable as CDC events."""

    source_name = "postgres_cdc"

    def __init__(
        self,
        source: AsyncIterable[Any] | Iterable[Any],
        *,
        stream: str = "default",
    ) -> None:
        self._source = source
        self.stream = stream
        self._iterator: AsyncIterator[Any] | None = None
        self._pending: dict[str, Any] = {}
        self._stop_requested = False
        self._checkpoint_store: CheckpointStore | None = None

    async def set_checkpoint_store(self, store: CheckpointStore) -> None:
        self._checkpoint_store = store
        configure = getattr(self._source, "set_checkpoint_store", None)
        if configure is not None:
            await _call_async_or_thread(configure, store)

    async def start(self) -> None:
        self._stop_requested = False
        start = getattr(self._source, "start", None)
        if start is not None:
            await _call_async_or_thread(start)
        values = getattr(self._source, "events", None)
        iterator: Any = values() if callable(values) else self._source
        if inspect.isawaitable(iterator):
            iterator = await iterator
        if hasattr(iterator, "__aiter__"):
            self._iterator = iterator.__aiter__()
        else:
            self._iterator = _sync_to_async(iterator)

    async def __anext__(self) -> StreamEvent:
        if self._iterator is None:
            await self.start()
        assert self._iterator is not None
        if self._stop_requested:
            raise StopAsyncIteration
        raw = await self._iterator.__anext__()
        event = normalize_debezium_event(raw, stream=self.stream)
        self._pending[event.id] = raw
        return event

    def __aiter__(self) -> DebeziumSource:
        return self

    async def ack(self, event: StreamEvent) -> None:
        raw = self._pending.pop(event.id, None)
        acknowledge = getattr(self._source, "ack", None)
        if acknowledge is not None:
            await _call_async_or_thread(acknowledge, raw if raw is not None else event)

    async def stop(self) -> None:
        self._stop_requested = True
        stop = getattr(self._source, "stop", None)
        if stop is not None:
            await _call_async_or_thread(stop)
        self._iterator = None
        self._pending.clear()


class _PostgresReplicationSource:
    """Small psycopg replication seam used by :class:`PostgresCDCSource`."""

    def __init__(
        self,
        dsn: str,
        *,
        slot_name: str,
        connection: Any | None = None,
        checkpoint_namespace: str = "postgres_cdc:postgres_cdc",
    ) -> None:
        self.dsn = dsn
        self.slot_name = slot_name
        self.connection = connection
        self.checkpoint_namespace = checkpoint_namespace
        self.cursor: Any | None = None
        self._checkpoint_store: CheckpointStore | None = None
        self._pending: dict[int, Any] = {}
        self._stopped = False

    async def set_checkpoint_store(self, store: CheckpointStore) -> None:
        self._checkpoint_store = store

    async def start(self) -> None:
        self._stopped = False
        connection = self.connection
        if connection is None:
            try:
                import psycopg
            except ImportError as exc:
                raise ImportError(
                    "PostgresCDCSource requires the optional 'psycopg' package"
                ) from exc
            connection = await asyncio.to_thread(
                psycopg.connect,
                self.dsn,
                autocommit=True,
                replication="database",
            )
            self.connection = connection
        assert connection is not None
        self.cursor = connection.cursor()
        start_replication = getattr(self.cursor, "start_replication", None)
        if start_replication is not None:
            kwargs: dict[str, Any] = {
                "slot_name": self.slot_name,
                "decode": True,
            }
            if self._checkpoint_store is not None:
                checkpoint = await self._checkpoint_store.load(
                    self.checkpoint_namespace,
                    "default",
                )
                if checkpoint is not None and checkpoint.offset is not None:
                    kwargs["start_lsn"] = checkpoint.offset
            await _call_async_or_thread(
                start_replication,
                **kwargs,
            )

    async def __anext__(self) -> Any:
        while not self._stopped:
            reader = getattr(self.cursor, "read_message", None)
            if reader is None and self.connection is not None:
                reader = getattr(self.connection, "read_message", None)
            if reader is None:
                raise RuntimeError("psycopg replication cursor has no read_message method")
            message = await _call_async_or_thread(reader)
            if message is None:
                await asyncio.sleep(0)
                continue
            data_start = getattr(message, "data_start", None)
            payload = getattr(message, "payload", message)
            payload = _decode(payload)
            if isinstance(payload, Mapping):
                payload = dict(payload)
                source = payload.get("source")
                source = dict(source) if isinstance(source, Mapping) else {}
                if data_start is not None:
                    source.setdefault("lsn", data_start)
                payload.setdefault("source", source)
            else:
                payload = {
                    "after": payload,
                    "op": "r",
                    "source": {"lsn": data_start},
                }
            self._pending[id(payload)] = message
            return payload
        raise StopAsyncIteration

    async def ack(self, raw: Any) -> None:
        message = self._pending.pop(id(raw), None)
        if message is None:
            return
        feedback = getattr(self.cursor, "send_feedback", None)
        data_start = getattr(message, "data_start", None)
        if feedback is not None and data_start is not None:
            await _call_async_or_thread(feedback, flush_lsn=data_start)

    def __aiter__(self) -> _PostgresReplicationSource:
        return self

    async def stop(self) -> None:
        self._stopped = True
        connection, self.connection = self.connection, None
        self.cursor = None
        self._pending.clear()
        if connection is not None:
            close = getattr(connection, "close", None)
            if close is not None:
                await _call_async_or_thread(close)


class PostgresCDCSource(DebeziumSource):
    """Consume Debezium-style PostgreSQL changes.

    Pass an async iterable (for example a Debezium Kafka source) for the most
    common deployment.  A psycopg logical-replication connection can also be
    supplied for deployments that emit Debezium-compatible envelopes directly.
    """

    def __init__(
        self,
        source: AsyncIterable[Any] | Iterable[Any] | None = None,
        *,
        events: AsyncIterable[Any] | Iterable[Any] | None = None,
        stream: str = "postgres_cdc",
        dsn: str | None = None,
        slot_name: str = "synapsekit_streaming",
        connection: Any | None = None,
    ) -> None:
        if source is not None and events is not None:
            raise ValueError("pass either source or events, not both")
        source = source if source is not None else events
        if source is None:
            if not dsn and connection is None:
                raise ValueError("source/events or a dsn/connection is required")
            source = _PostgresReplicationSource(
                dsn or "",
                slot_name=slot_name,
                connection=connection,
                checkpoint_namespace=f"postgres_cdc:{stream}",
            )
        super().__init__(source, stream=stream)


# Descriptive aliases keep provider-specific and generic import styles usable.
KafkaStreamSource = KafkaSource
RedpandaStreamSource = RedpandaSource
PulsarStreamSource = PulsarSource
KinesisStreamSource = KinesisSource
PostgresLogicalReplicationSource = PostgresCDCSource
CDCSource = DebeziumSource


__all__ = [
    "AsyncEventSource",
    "CDCSource",
    "DebeziumSource",
    "KafkaSource",
    "KafkaStreamSource",
    "KinesisSource",
    "KinesisStreamSource",
    "PostgresLogicalReplicationSource",
    "PostgresCDCSource",
    "PulsarSource",
    "PulsarStreamSource",
    "RedpandaSource",
    "RedpandaStreamSource",
    "StreamSource",
    "normalize_debezium_event",
]
