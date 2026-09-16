# Streaming ingestion

SynapseKit can keep a vector store, `WorldModelRAG`, or `KnowledgeMesh` current as events arrive. The pipeline is:

    source -> async transform -> idempotent target upsert -> checkpoint -> source acknowledgement

The default transform turns JSON, strings, and bytes into a `Document`. It adds `event_id`, `ingestion_id`, source/stream metadata, and `valid_at` when the source supplies a timestamp.

## Installation

The core streaming types have no provider imports. Install only the source integrations you use:

    uv add 'synapsekit[kafka]'
    uv add 'synapsekit[streaming-pulsar]'
    uv add 'synapsekit[streaming-kinesis]'
    uv add 'synapsekit[streaming-cdc]'

The aggregate extra is `synapsekit[streaming]`. Kafka and Redpanda use the Kafka protocol; the adapter runs the synchronous `kafka-python` client in worker threads so polling and commits do not block the event loop. Pulsar, Kinesis, and psycopg are imported only when their adapters are started.

## Kafka to a live mesh

```python
import asyncio
from pathlib import Path

from synapsekit import (
    KafkaSource,
    KnowledgeMesh,
    KnowledgeMeshSink,
    SQLiteCheckpointStore,
    StreamingIngestor,
)


async def main() -> None:
    mesh = KnowledgeMesh(...)  # configure the mesh's RAG dependencies
    checkpoints = SQLiteCheckpointStore(Path(".synapsekit/streaming.db"))
    source = KafkaSource(
        "application-events",
        bootstrap_servers="localhost:9092",
        group_id="synapsekit-application-events",
    )
    ingestor = StreamingIngestor(
        source,
        KnowledgeMeshSink(mesh),
        checkpoint_store=checkpoints,
        batch_size=32,
    )

    task = await ingestor.start()
    answer = await mesh.query("What changed in the latest application events?")
    print(answer)
    await ingestor.stop()
    await task
    checkpoints.close()


asyncio.run(main())
```

`RedpandaSource` has the same constructor and target behavior, with `event.source == "redpanda"`.

## Other sources

`PulsarSource` acknowledges a message only after the target and checkpoint succeed. `KinesisSource` checkpoints each shard sequence number and starts after the stored sequence on restart. `PostgresCDCSource` accepts an async iterable of Debezium envelopes (the normal deployment is a Debezium Kafka source) or a psycopg logical-replication connection. `normalize_debezium_event()` exposes the envelope conversion for custom sources.

All adapters produce `StreamEvent` values with a stable identity. Kafka uses `topic:partition:offset`, Kinesis uses `stream:shard:sequence`, and Debezium uses logical source, LSN, table, and row key. `SQLiteCheckpointStore` atomically records processed identities and partition checkpoints. Because a target write and a checkpoint database cannot share one transaction, recovery is exactly-once-ish: a crash may replay an event, but the built-in targets replace the same `ingestion_id` instead of appending a duplicate.

The ingestor pulls only while the current batch has room and waits for the target before pulling more. This bounded pull pattern is the backpressure boundary. A failed transform or target is not acknowledged or checkpointed; rerunning resumes it.

## Testcontainers demo

The integration test in `tests/streaming/test_kafka_testcontainer.py` starts Kafka in Testcontainers, publishes two JSON events, ingests them, then replays them with a new consumer group using the same SQLite checkpoint database. It asserts both events are visible and the replay adds no duplicate vectors:

    uv sync --group dev --group integration
    uv run pytest tests/streaming/test_kafka_testcontainer.py -m integration -q
