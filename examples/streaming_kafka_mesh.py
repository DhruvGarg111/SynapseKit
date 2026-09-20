"""Interactive live RAG over a Kafka topic.

Run with a Kafka broker available at KAFKA_BOOTSTRAP_SERVERS and publish JSON
records such as {"text": "Alice deployed the search service"} to the topic.
The checkpoint database lets the process restart without adding the same event
twice.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from synapsekit import (
    ExtractionPolicy,
    KafkaSource,
    KnowledgeMesh,
    SQLiteCheckpointStore,
    StreamingIngestor,
)
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.mesh import MeshConfig
from synapsekit.mesh.embeddings import HashingEmbeddings
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import HeuristicWorldModelExtractor, WorldModelRAG


class DemoLLM(BaseLLM):
    """Offline answerer; replace with a configured provider for production."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="streaming-demo", api_key="", provider="demo"))

    async def stream(self, prompt: str, **kwargs: object) -> AsyncGenerator[str]:
        del prompt, kwargs
        yield "The live mesh contains the latest events retrieved from Kafka."


async def main() -> None:
    topic = os.getenv("SYNAPSEKIT_KAFKA_TOPIC", "application-events")
    brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    state_dir = Path(os.getenv("SYNAPSEKIT_STATE_DIR", ".synapsekit"))

    world_model = WorldModelRAG(
        llm=DemoLLM(),
        extraction=ExtractionPolicy(temporal=True, causal=True),
        graph_backend="in_memory",
        extractor=HeuristicWorldModelExtractor(),
        vector_store=InMemoryVectorStore(HashingEmbeddings()),
    )
    mesh = KnowledgeMesh(
        MeshConfig(roots=[], state_dir=state_dir / "mesh", use_git=False),
        rag=world_model,
    )
    checkpoints = SQLiteCheckpointStore(state_dir / "streaming.sqlite3")
    ingestor = StreamingIngestor(
        KafkaSource(topic, bootstrap_servers=brokers, group_id=f"synapsekit-{topic}"),
        mesh,
        checkpoint_store=checkpoints,
        batch_size=16,
    )

    task = await ingestor.start()
    print(f"Listening to {topic} on {brokers}. Ask a question, or press Ctrl-D to exit.")
    try:
        while True:
            question = await asyncio.to_thread(input, "question> ")
            result = await mesh.query(question)
            print(result.answer)
    except EOFError:
        pass
    finally:
        await ingestor.stop()
        await task
        checkpoints.close()


if __name__ == "__main__":
    asyncio.run(main())
