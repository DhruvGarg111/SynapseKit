"""Async source-to-target streaming ingestion orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterable, Awaitable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

from ..loaders.base import Document
from .checkpoint import CheckpointStore, InMemoryCheckpointStore
from .types import StreamEvent


class EventTransformer(Protocol):
    """Callable contract for converting one stream event into documents."""

    def __call__(
        self, event: StreamEvent
    ) -> (
        Document
        | str
        | Iterable[Document | str]
        | Awaitable[Document | str | Iterable[Document | str]]
        | None
    ):
        """Transform one event."""
        ...


class IngestionTarget(Protocol):
    """Target contract consumed by :class:`StreamingIngestor`."""

    async def upsert(self, documents: list[Document]) -> None:
        """Apply a batch idempotently."""
        ...


@dataclass
class IngestionStats:
    """Counters returned by one ingestor run."""

    received: int = 0
    processed: int = 0
    skipped: int = 0
    documents: int = 0
    batches: int = 0
    failed: int = 0


def _json_text(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, ensure_ascii=False)


def _payload_text(payload: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, Document):
        return payload.text, dict(payload.metadata)
    if isinstance(payload, bytes):
        try:
            payload = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return payload.decode("utf-8", errors="replace"), {}
    if isinstance(payload, str):
        return payload, {}
    if isinstance(payload, Mapping):
        for key in ("text", "content", "body", "message", "value"):
            if key in payload:
                value = payload[key]
                text = value if isinstance(value, str) else _json_text(value)
                metadata = {
                    str(name): value
                    for name, value in payload.items()
                    if str(name) != key and isinstance(value, (str, int, float, bool))
                }
                return text, metadata
        return _json_text(payload), {
            str(name): value
            for name, value in payload.items()
            if isinstance(value, (str, int, float, bool))
        }
    return str(payload), {}


def _event_document(event: StreamEvent) -> Document:
    """Default JSON/scalar transformer used by the ingestor."""

    text, payload_metadata = _payload_text(event.payload)
    metadata = {**payload_metadata, **dict(event.metadata)}
    metadata.update(
        {
            "source_type": event.source,
            "stream": event.stream,
            "event_id": event.id,
            "partition": event.partition,
            "offset": event.offset,
            "ingestion_id": event.id,
        }
    )
    metadata.setdefault("ingest_time", cast(datetime, event.ingest_time).isoformat())
    if event.timestamp is not None:
        timestamp = cast(datetime, event.timestamp).isoformat()
        metadata.setdefault("timestamp", timestamp)
        metadata.setdefault("valid_at", timestamp)
    metadata.setdefault("source", f"stream://{event.source}/{event.stream}/{event.id}")
    metadata.setdefault("path", metadata["source"])
    metadata.setdefault("chunk_id", event.id)
    return Document(text=text, metadata=metadata)


def _as_documents(value: Any) -> list[Document]:
    if value is None:
        return []
    if isinstance(value, Document):
        return [value]
    if isinstance(value, str):
        return [Document(value)]
    if isinstance(value, bytes):
        return [Document(value.decode("utf-8", errors="replace"))]
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return [Document(text, dict(value.get("metadata") or {}))]
        return [Document(_json_text(value))]
    if isinstance(value, Iterable):
        documents: list[Document] = []
        for item in value:
            documents.extend(_as_documents(item))
        return documents
    return [Document(str(value))]


def _decorate_documents(event: StreamEvent, documents: list[Document]) -> list[Document]:
    decorated: list[Document] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        metadata.setdefault("source_type", event.source)
        metadata.setdefault("stream", event.stream)
        metadata["event_id"] = event.id
        metadata.setdefault("partition", event.partition)
        metadata.setdefault("offset", event.offset)
        if event.timestamp is not None:
            timestamp = cast(datetime, event.timestamp).isoformat()
            metadata.setdefault("timestamp", timestamp)
            metadata.setdefault("valid_at", timestamp)
        ingestion_id = event.id if len(documents) == 1 else f"{event.id}:{index}"
        metadata["ingestion_id"] = ingestion_id
        metadata.setdefault("ingest_time", cast(datetime, event.ingest_time).isoformat())
        metadata.setdefault("source", f"stream://{event.source}/{event.stream}/{event.id}")
        metadata.setdefault("path", metadata["source"])
        metadata.setdefault("chunk_id", ingestion_id)
        if document.text.strip():
            decorated.append(Document(document.text, metadata))
    return decorated


def _coerce_target(target: Any) -> Any:
    """Wrap SynapseKit's native stores while leaving custom sinks untouched."""

    if callable(getattr(target, "upsert", None)):
        return target
    # Import lazily: targets imports this module for the default transformer.
    from .targets import KnowledgeMeshSink, VectorStoreSink, WorldModelSink

    if hasattr(target, "rag") and hasattr(target, "store"):
        return KnowledgeMeshSink(target)
    if hasattr(target, "graph_backend") and callable(getattr(target, "ingest", None)):
        return WorldModelSink(target)
    if callable(getattr(target, "add", None)):
        return VectorStoreSink(target)
    return target


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_optional(obj: Any, name: str, *args: Any) -> Any:
    method = getattr(obj, name, None)
    if not callable(method):
        return None
    return await _maybe_await(method(*args))


class StreamingIngestor:
    """Consume events, transform them, and upsert them in ordered batches.

    The source is deliberately pulled only while the current batch has room. No
    provider client is allowed to prefetch an unbounded queue, so awaiting the
    target is the backpressure boundary. Source acknowledgements and checkpoint
    writes happen only after a successful target upsert.
    """

    def __init__(
        self,
        source: AsyncIterable[StreamEvent] | Any,
        target: IngestionTarget | Any,
        *,
        transform: EventTransformer | None = None,
        checkpoint_store: CheckpointStore | None = None,
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.source = source
        self.target = _coerce_target(target)
        self.transform = transform or _event_document
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self.batch_size = batch_size
        self._stop_event = asyncio.Event()
        self._run_task: asyncio.Task[IngestionStats] | None = None
        self._active = False
        # Held for the duration of a batch's upsert+commit+ack sequence, so an
        # external stop() can't tear the source down mid-batch (which would
        # race an in-flight ack against the source's teardown, see stop()).
        self._batch_lock = asyncio.Lock()

    async def run(self, *, max_events: int | None = None) -> IngestionStats:
        """Run until the source is exhausted, stopped, or ``max_events`` is met."""

        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative")
        if self._active:
            raise RuntimeError("StreamingIngestor is already running")

        self._active = True
        self._stop_event.clear()
        stats = IngestionStats()
        batch: list[StreamEvent] = []
        try:
            await _call_optional(self.source, "set_checkpoint_store", self.checkpoint_store)
            await _call_optional(self.source, "start")
            iterator = await self._source_iterator()
            async for raw_event in iterator:
                if self._stop_event.is_set():
                    break
                if max_events is not None and stats.received >= max_events:
                    break
                event = self._coerce_event(raw_event)
                stats.received += 1
                if await self.checkpoint_store.is_processed(event.stream_key, event.id):
                    stats.skipped += 1
                    await _call_optional(self.source, "ack", event)
                else:
                    batch.append(event)
                    if len(batch) >= self.batch_size:
                        documents = await self._process_batch(batch)
                        stats.processed += len(batch)
                        stats.documents += documents
                        stats.batches += 1
                        batch = []
                if max_events is not None and stats.received >= max_events:
                    break

            if batch:
                documents = await self._process_batch(batch)
                stats.processed += len(batch)
                stats.documents += documents
                stats.batches += 1
            return stats
        except Exception:
            stats.failed += 1
            raise
        finally:
            await _call_optional(self.source, "stop")
            self._active = False

    async def start(self, *, max_events: int | None = None) -> asyncio.Task[IngestionStats]:
        """Start ``run`` in the background and return its task."""

        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("StreamingIngestor is already running")
        self._stop_event.clear()
        self._run_task = asyncio.create_task(
            self.run(max_events=max_events),
            name="synapsekit-streaming-ingestor",
        )
        return self._run_task

    async def stop(self) -> None:
        """Request a graceful stop and wait for a background run, if any.

        Calling ``source.stop()`` here (rather than only relying on ``run()``'s
        own ``finally``) is required to interrupt a source blocked inside its
        next-event wait -- that's the only way an in-progress ``run()`` task
        ever notices the stop request. But tearing the source down while a
        batch is mid upsert/commit/ack would race that batch's acks against
        the source's teardown, silently dropping already-persisted
        acknowledgements (e.g. ``KafkaSource.ack`` no-ops once its consumer is
        gone). ``_batch_lock`` closes that window: it's held for the whole
        upsert+commit+ack sequence of a batch, so acquiring it here blocks
        until any in-flight batch has fully acked before the source is
        stopped.
        """

        self._stop_event.set()
        async with self._batch_lock:
            await _call_optional(self.source, "stop")
        task = self._run_task
        if task is not None and task is not asyncio.current_task():
            await task

    async def _process_batch(self, events: list[StreamEvent]) -> int:
        documents: list[Document] = []
        for event in events:
            transformed = await _maybe_await(self.transform(event))
            documents.extend(_decorate_documents(event, _as_documents(transformed)))
        async with self._batch_lock:
            if documents:
                upsert = getattr(self.target, "upsert", None)
                if not callable(upsert):
                    raise TypeError("target must expose an async upsert(documents) method")
                await _maybe_await(upsert(documents))
            for event in events:
                await self.checkpoint_store.commit_event(event)
                await _call_optional(self.source, "ack", event)
        return len(documents)

    async def _source_iterator(self) -> AsyncIterable[Any]:
        events = getattr(self.source, "events", None)
        iterator = events() if callable(events) else self.source
        iterator = await _maybe_await(iterator)
        if not hasattr(iterator, "__aiter__"):
            raise TypeError("source must be an async iterable or expose events()")
        return cast(AsyncIterable[Any], iterator)

    def _coerce_event(self, raw_event: Any) -> StreamEvent:
        if isinstance(raw_event, StreamEvent):
            return raw_event
        source_name = str(getattr(self.source, "source_name", type(self.source).__name__))
        stream_name = str(getattr(self.source, "stream", "default"))
        if isinstance(raw_event, Mapping) and "payload" in raw_event:
            fields = {
                key: raw_event[key]
                for key in (
                    "payload",
                    "source",
                    "stream",
                    "event_id",
                    "partition",
                    "offset",
                    "timestamp",
                    "key",
                    "headers",
                    "metadata",
                )
                if key in raw_event
            }
            fields.setdefault("source", source_name)
            fields.setdefault("stream", stream_name)
            return StreamEvent(**fields)
        return StreamEvent(raw_event, source=source_name, stream=stream_name)
