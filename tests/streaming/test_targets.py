from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import numpy as np
import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.loaders.base import Document
from synapsekit.mesh import KnowledgeMesh, MeshConfig
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import (
    EntityMention,
    ExtractionResult,
    InMemoryWorldGraphBackend,
    RelationMention,
    WorldModelRAG,
)
from synapsekit.streaming import InMemoryCheckpointStore, StreamEvent, StreamingIngestor
from synapsekit.streaming.targets import (
    KnowledgeMeshSink,
    VectorStoreSink,
    WorldModelSink,
)


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        for index, char in enumerate(text.encode()):
            vector[index % 4] += char
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="fake", api_key="", provider="fake"))

    async def stream(self, prompt: str, **kwargs: object) -> AsyncGenerator[str]:
        yield "answer"


class StaticExtractor:
    async def extract(self, text: str, policy: object) -> ExtractionResult:
        return ExtractionResult(
            entities=[EntityMention("Alice"), EntityMention("Search API")],
            relations=[RelationMention("Alice", "worked_on", "Search API")],
        )


@pytest.mark.asyncio
async def test_vector_store_sink_replaces_same_ingestion_id() -> None:
    store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    sink = VectorStoreSink(store)
    first = Document("old text", {"ingestion_id": "event-1"})
    second = Document("new text", {"ingestion_id": "event-1"})

    await sink.upsert([first])
    await sink.upsert([second])

    assert len(store) == 1
    assert store._texts == ["new text"]


@pytest.mark.asyncio
async def test_world_model_sink_is_idempotent_and_uses_event_valid_at() -> None:
    graph = InMemoryWorldGraphBackend()
    vector_store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    rag = WorldModelRAG(
        llm=FakeLLM(),
        graph_backend=graph,
        vector_store=vector_store,
        extractor=StaticExtractor(),  # type: ignore[arg-type]
        trace=False,
    )
    sink = WorldModelSink(rag)
    timestamp = datetime(2026, 9, 16, 12, tzinfo=UTC)
    document = Document(
        "Alice worked on Search API",
        {
            "source": "stream://test/events/event-1",
            "ingestion_id": "event-1",
            "valid_at": timestamp.isoformat(),
        },
    )

    await sink.upsert([document])
    await sink.upsert([document])

    assert len(vector_store) == 1
    assert len(graph.edges) == 1
    edge = next(iter(graph.edges.values()))
    assert edge.valid_at == timestamp


@pytest.mark.asyncio
async def test_world_model_sink_appends_distinct_temporal_edges() -> None:
    graph = InMemoryWorldGraphBackend()
    vector_store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    rag = WorldModelRAG(
        llm=FakeLLM(),
        graph_backend=graph,
        vector_store=vector_store,
        extractor=StaticExtractor(),  # type: ignore[arg-type]
        trace=False,
    )
    sink = WorldModelSink(rag)
    first_time = datetime(2026, 9, 16, 12, tzinfo=UTC)
    second_time = datetime(2026, 9, 17, 12, tzinfo=UTC)

    await sink.upsert(
        [
            Document(
                "Alice worked on Search API",
                {
                    "source": "stream://test/events/event-1",
                    "ingestion_id": "event-1",
                    "valid_at": first_time.isoformat(),
                },
            ),
            Document(
                "Alice worked on Search API",
                {
                    "source": "stream://test/events/event-2",
                    "ingestion_id": "event-2",
                    "valid_at": second_time.isoformat(),
                },
            ),
        ]
    )

    assert len(graph.edges) == 2
    assert {edge.valid_at for edge in graph.edges.values()} == {first_time, second_time}


@pytest.mark.asyncio
async def test_mesh_sink_marks_stream_documents_active_and_replaces_them(tmp_path) -> None:
    graph = InMemoryWorldGraphBackend()
    vector_store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    rag = WorldModelRAG(
        llm=FakeLLM(),
        graph_backend=graph,
        vector_store=vector_store,
        extractor=StaticExtractor(),  # type: ignore[arg-type]
        trace=False,
    )
    mesh = KnowledgeMesh(
        MeshConfig(
            roots=[tmp_path],
            state_dir=tmp_path / "mesh-state",
            vector_backend="memory",
            use_git=False,
        ),
        rag=rag,
    )
    sink = KnowledgeMeshSink(mesh)
    path = "stream://test/events/event-1"
    first = Document(
        "old text",
        {
            "source": path,
            "path": path,
            "chunk_id": "event-1",
            "ingestion_id": "event-1",
        },
    )
    second = Document(
        "new text",
        {
            "source": path,
            "path": path,
            "chunk_id": "event-1",
            "ingestion_id": "event-1",
        },
    )

    await sink.upsert([first])
    await sink.upsert([second])

    assert len(vector_store) == 1
    assert vector_store._texts == ["new text"]
    assert mesh.store.active_chunk_ids_for_path(path) == {"event-1"}
    mesh.store.close()


class FiniteSource:
    source_name = "test"
    stream = "events"

    def __init__(self) -> None:
        self.events = [
            StreamEvent("hello", "test", "events", "event-1", offset=1),
            StreamEvent("hello", "test", "events", "event-1", offset=1),
        ]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


@pytest.mark.asyncio
async def test_ingestor_wraps_a_native_vector_store_and_skips_replays() -> None:
    vector_store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    ingestor = StreamingIngestor(
        FiniteSource(),
        vector_store,
        checkpoint_store=InMemoryCheckpointStore(),
        batch_size=1,
    )

    stats = await ingestor.run()

    assert stats.processed == 1
    assert stats.skipped == 1
    assert len(vector_store) == 1


@pytest.mark.asyncio
async def test_vector_sink_removes_cdc_tombstones_by_row_identity() -> None:
    vector_store = InMemoryVectorStore(FakeEmbeddings())  # type: ignore[arg-type]
    sink = VectorStoreSink(vector_store)
    await sink.upsert(
        [
            Document(
                "row body",
                {"ingestion_id": "event-1", "document_id": "items:42"},
            )
        ]
    )

    await sink.upsert(
        [
            Document(
                "tombstone",
                {
                    "ingestion_id": "event-2",
                    "document_id": "items:42",
                    "deleted": True,
                },
            )
        ]
    )

    assert len(vector_store) == 0
