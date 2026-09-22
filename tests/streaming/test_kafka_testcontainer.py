from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synapsekit.mesh import KnowledgeMesh, MeshConfig
from synapsekit.retrieval.world_model import EntityMention, ExtractionResult, RelationMention
from synapsekit.streaming import (
    KafkaSource,
    SQLiteCheckpointStore,
    StreamingIngestor,
)


class StaticExtractor:
    async def extract(self, text: str, policy: object) -> ExtractionResult:
        return ExtractionResult(
            entities=[EntityMention("Alice"), EntityMention("Search API")],
            relations=[RelationMention("Alice", "worked_on", "Search API")],
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kafka_container_replay_is_deduplicated_after_restart(tmp_path: Path) -> None:
    kafka_module = pytest.importorskip("testcontainers.kafka")
    pytest.importorskip("kafka")
    kafka_container = kafka_module.KafkaContainer
    topic = "synapsekit-streaming-test"

    with kafka_container() as container:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=container.get_bootstrap_server(),
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        producer.send(topic, {"text": "first event"})
        producer.send(topic, {"text": "second event"})
        producer.flush()
        producer.close()

        mesh_config = MeshConfig(
            roots=[tmp_path],
            state_dir=tmp_path / "mesh-state",
            vector_backend="memory",
            graph_backend="memory",
            use_git=False,
        )
        mesh = KnowledgeMesh(mesh_config, extractor=StaticExtractor())  # type: ignore[arg-type]
        checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
        first = StreamingIngestor(
            KafkaSource(
                topic,
                bootstrap_servers=container.get_bootstrap_server(),
                group_id="synapsekit-first-run",
                poll_timeout=0.2,
                checkpoint_namespace="kafka:container:replay-test",
            ),
            mesh,
            checkpoint_store=checkpoints,
            batch_size=1,
        )
        first_stats = await asyncio.wait_for(first.run(max_events=2), timeout=30)

        assert first_stats.processed == 2
        assert len(mesh.rag.vector_store) == 2
        assert mesh.status().active_chunks == 2
        assert mesh.rag.graph_backend.edges
        assert all(edge.valid_at is not None for edge in mesh.rag.graph_backend.edges.values())
        mesh.store.close()
        restarted_mesh = KnowledgeMesh(mesh_config, extractor=StaticExtractor())  # type: ignore[arg-type]

        replay = StreamingIngestor(
            KafkaSource(
                topic,
                bootstrap_servers=container.get_bootstrap_server(),
                group_id="synapsekit-replay-run",
                poll_timeout=0.2,
                checkpoint_namespace="kafka:container:replay-test",
            ),
            restarted_mesh,
            checkpoint_store=checkpoints,
            batch_size=1,
        )
        replay_stats = await asyncio.wait_for(replay.run(max_events=2), timeout=30)
        checkpoints.close()

        assert replay_stats.skipped == 2
        assert len(restarted_mesh.rag.vector_store) == 2
        assert restarted_mesh.status().active_chunks == 2
        result = await restarted_mesh.query("first event", top_k=1)
        assert result.hits
        assert result.hits[0].text == "first event"
        restarted_mesh.store.close()
