from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from synapsekit.loaders.base import Document
from synapsekit.streaming import InMemoryCheckpointStore, StreamEvent, StreamingIngestor


class ListSource:
    def __init__(self, events: list[StreamEvent]) -> None:
        self.events = events
        self.index = 0
        self.pulls = 0
        self.acks: list[str] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        self.pulls += 1
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        return event

    async def ack(self, event: StreamEvent) -> None:
        self.acks.append(event.id)

    async def stop(self) -> None:
        self.stopped = True


class RecordingTarget:
    def __init__(self) -> None:
        self.batches: list[list[Document]] = []

    async def upsert(self, documents: list[Document]) -> None:
        self.batches.append(documents)


@pytest.mark.asyncio
async def test_ingestor_transforms_and_submits_bounded_batches() -> None:
    events = [
        StreamEvent("one", source="test", stream="events", event_id="1", offset=1),
        StreamEvent("two", source="test", stream="events", event_id="2", offset=2),
        StreamEvent("three", source="test", stream="events", event_id="3", offset=3),
    ]
    source = ListSource(events)
    target = RecordingTarget()

    async def transform(event: StreamEvent) -> Document:
        await asyncio.sleep(0)
        return Document(event.payload, {"event_id": event.id})

    ingestor = StreamingIngestor(
        source,
        target,
        transform=transform,
        checkpoint_store=InMemoryCheckpointStore(),
        batch_size=2,
    )

    stats = await ingestor.run()

    assert [[doc.text for doc in batch] for batch in target.batches] == [
        ["one", "two"],
        ["three"],
    ]
    assert stats.received == 3
    assert stats.processed == 3
    assert stats.documents == 3
    assert stats.batches == 2
    assert source.started is True
    assert source.stopped is True
    assert source.acks == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_ingestor_skips_durable_redeliveries_but_acknowledges_them() -> None:
    event = StreamEvent("one", source="test", stream="events", event_id="1", offset=1)
    source = ListSource([event, event, StreamEvent("two", "test", "events", "2", offset=2)])
    target = RecordingTarget()
    checkpoints = InMemoryCheckpointStore()
    ingestor = StreamingIngestor(
        source,
        target,
        transform=lambda item: Document(str(item.payload)),
        checkpoint_store=checkpoints,
        batch_size=1,
    )

    stats = await ingestor.run()

    assert stats.processed == 2
    assert stats.skipped == 1
    assert [doc.text for batch in target.batches for doc in batch] == ["one", "two"]
    assert source.acks == ["1", "1", "2"]


@pytest.mark.asyncio
async def test_failed_target_does_not_ack_or_checkpoint() -> None:
    event = StreamEvent("one", source="test", stream="events", event_id="1", offset=1)
    source = ListSource([event])

    class FailingTarget:
        async def upsert(self, documents: list[Document]) -> None:
            raise RuntimeError("target unavailable")

    checkpoints = InMemoryCheckpointStore()
    ingestor = StreamingIngestor(
        source,
        FailingTarget(),
        transform=lambda item: Document(str(item.payload)),
        checkpoint_store=checkpoints,
    )

    with pytest.raises(RuntimeError, match="target unavailable"):
        await ingestor.run()

    assert source.acks == []
    assert await checkpoints.is_processed(event.stream_key, event.id) is False


@pytest.mark.asyncio
async def test_ingestor_resumes_from_a_checkpoint_store() -> None:
    first_event = StreamEvent("one", "test", "events", "1", offset=1)
    second_event = StreamEvent("two", "test", "events", "2", offset=2)
    checkpoints = InMemoryCheckpointStore()
    first_target = RecordingTarget()
    first = StreamingIngestor(
        ListSource([first_event, second_event]),
        first_target,
        transform=lambda item: Document(str(item.payload)),
        checkpoint_store=checkpoints,
        batch_size=1,
    )
    await first.run(max_events=1)

    second_source = ListSource([first_event, second_event])
    second_target = RecordingTarget()
    second = StreamingIngestor(
        second_source,
        second_target,
        transform=lambda item: Document(str(item.payload)),
        checkpoint_store=checkpoints,
        batch_size=1,
    )

    stats = await second.run()

    assert stats.skipped == 1
    assert stats.processed == 1
    assert [doc.text for batch in second_target.batches for doc in batch] == ["two"]


@pytest.mark.asyncio
async def test_ingestor_does_not_pull_next_batch_until_target_finishes() -> None:
    first = StreamEvent("one", "test", "events", "1", offset=1)
    second = StreamEvent("two", "test", "events", "2", offset=2)
    third = StreamEvent("three", "test", "events", "3", offset=3)
    source = ListSource([first, second, third])
    target_started = asyncio.Event()
    release_target = asyncio.Event()

    class BlockingTarget:
        async def upsert(self, documents: list[Document]) -> None:
            target_started.set()
            await release_target.wait()

    ingestor = StreamingIngestor(
        source,
        BlockingTarget(),
        transform=lambda item: Document(str(item.payload)),
        batch_size=2,
    )
    task = asyncio.create_task(ingestor.run())
    await target_started.wait()
    await asyncio.sleep(0)

    assert source.pulls == 2
    release_target.set()
    await task


@pytest.mark.asyncio
async def test_start_and_stop_clean_up_an_infinite_source() -> None:
    class InfiniteSource(ListSource):
        def __init__(self) -> None:
            super().__init__([])
            self.release = asyncio.Event()

        async def __anext__(self) -> StreamEvent:
            self.pulls += 1
            await self.release.wait()
            return StreamEvent("never", "test", "events", str(self.pulls), offset=self.pulls)

        async def stop(self) -> None:
            self.stopped = True
            self.release.set()

    source = InfiniteSource()
    ingestor = StreamingIngestor(
        source,
        RecordingTarget(),
        transform=lambda item: Document(str(item.payload)),
    )

    task = await ingestor.start()
    await asyncio.sleep(0)
    await ingestor.stop()
    stats = await task

    assert stats.received == 0
    assert source.stopped is True


@pytest.mark.asyncio
async def test_default_transform_preserves_event_metadata() -> None:
    event = StreamEvent(
        {"text": "hello", "category": "notice"},
        source="test",
        stream="events",
        event_id="event-1",
        partition=0,
        offset=9,
    )
    target = RecordingTarget()

    await StreamingIngestor(
        ListSource([event]),
        target,
        checkpoint_store=InMemoryCheckpointStore(),
    ).run()

    document = target.batches[0][0]
    assert document.text == "hello"
    assert document.metadata["event_id"] == "event-1"
    assert document.metadata["stream"] == "events"
    assert document.metadata["partition"] == 0
    assert document.metadata["offset"] == 9
