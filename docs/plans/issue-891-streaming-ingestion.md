# Streaming / Event-Driven Ingestion Implementation Plan

**Issue:** #891 — Streaming / event-driven ingestion — keep the knowledge mesh live

**Goal:** Add an async-first, restartable source-to-document-to-target ingestion pipeline that can keep a vector store, `WorldModelRAG`, or `KnowledgeMesh` current without duplicate documents after redelivery.

**Architecture:** `StreamingIngestor` consumes normalized `StreamEvent` objects from an async source, applies an async or synchronous document transformer, and submits a bounded batch to an idempotent target. A durable checkpoint store records processed event IDs and the source position only after the target succeeds; source acknowledgements happen after that record. This gives at-least-once delivery with exactly-once-ish effects when the target upsert is idempotent. Provider SDKs stay behind lazy imports and synchronous clients are isolated with `asyncio.to_thread`.

**Tech Stack:** Python 3.11+, stdlib `asyncio`/`sqlite3`, existing `Document`, `VectorStore`, `WorldModelRAG`, `KnowledgeMesh`, optional Kafka/Redpanda, Pulsar, boto3/Kinesis, and psycopg CDC clients.

---

## Contract checklist from issue #891

| Issue requirement | Planned evidence |
|---|---|
| `StreamingIngestor`: source -> transform -> upsert | Public `StreamingIngestor` with source, transformer, target, `run`, `start`, and `stop`; unit tests cover the complete path. |
| Kafka and Redpanda | `KafkaSource` using Kafka protocol; `RedpandaSource` is the same protocol with a distinct source label and endpoint configuration. |
| Pulsar | `PulsarSource` with lazy `pulsar` import, receive and acknowledge boundaries. |
| AWS Kinesis | `KinesisSource` with lazy boto3 client, shard polling, sequence-number resume, and endpoint override for LocalStack-style tests. |
| PostgreSQL logical replication / Debezium-style CDC | `PostgresCDCSource` for a supplied logical-replication/message iterator plus `DebeziumSource` normalization of `before`/`after`, operation, source LSN, and transaction metadata. |
| Offset checkpointing | `Checkpoint` and `SQLiteCheckpointStore` persist per-stream/per-partition position. |
| Stable-ID deduplication | Processed event ledger plus target metadata `ingestion_id`; vector/mesh targets replace the same ID rather than append a second document. |
| Backpressure and batching | `StreamingIngestor` pulls only while its batch has capacity and submits one bounded batch at a time; `batch_size` is validated and tested. |
| Async-first and graceful resume | Public lifecycle methods are coroutines; sync SDK calls are offloaded; source cleanup runs in `finally`; restart tests reuse a checkpoint database. |
| World Model temporal edges | Event timestamp/`valid_at` metadata is passed through `WorldModelRAG.ingest`, and missing relation timestamps inherit it. |
| Testcontainers Kafka demo | Add a skip-safe Kafka integration test that publishes a record, runs the ingestor into a persisted mesh, queries the mesh, then reopens it and proves no duplicate active chunks. |
| Documentation and example | Add a focused Kafka live-RAG document and executable example using the local mesh plus `SQLiteCheckpointStore`. |

## Implementation tasks

### Task 1: Add normalized event and checkpoint contracts

**Files:**
- Create: `src/synapsekit/streaming/types.py`
- Create: `src/synapsekit/streaming/checkpoint.py`
- Create: `src/synapsekit/streaming/__init__.py`
- Test: `tests/streaming/test_types_and_checkpoints.py`

Write tests first for event identity/partition keys, timestamp normalization, in-memory checkpoint round trips, and SQLite checkpoint persistence. Implement immutable `StreamEvent`, `Checkpoint`, `InMemoryCheckpointStore`, and `SQLiteCheckpointStore`; include an atomic `commit_event` operation that marks an event processed and advances its partition checkpoint together.

### Task 2: Implement the bounded async orchestration core

**Files:**
- Create: `src/synapsekit/streaming/ingestor.py`
- Test: `tests/streaming/test_ingestor.py`

Write failing tests for async transforms, batch submission, backpressure (the source must not be pulled beyond the configured in-flight batch), duplicate redelivery, failure-before-checkpoint, source acknowledgement ordering, clean source shutdown, and `start`/`stop`. Implement `StreamingIngestor` with a single ordered bounded batch worker, durable processed-event checks, target failure propagation, and `IngestionStats`.

### Task 3: Add document transformation and idempotent targets

**Files:**
- Create: `src/synapsekit/streaming/targets.py`
- Modify: `src/synapsekit/retrieval/world_model.py`
- Modify: `src/synapsekit/mesh/core.py`
- Test: `tests/streaming/test_targets.py`
- Test: `tests/retrieval/test_world_model.py`
- Test: `tests/mesh/test_knowledge_mesh.py`

Write tests for mapping scalar/JSON payloads to `Document`, preserving event metadata, stable IDs, vector-store replacement, world-model ingestion, event timestamps becoming `valid_at`, and mesh indexing/query visibility. Implement the default transformer, `VectorStoreSink`, `WorldModelSink`, and `KnowledgeMeshSink`. Keep vector and mesh operations additive and use the existing deletion/upsert seams; do not change unrelated retrieval behavior.

### Task 4: Add provider sources behind lazy optional dependencies

**Files:**
- Create: `src/synapsekit/streaming/sources.py`
- Test: `tests/streaming/test_sources.py`

Write fake-client tests first for Kafka/Redpanda poll and commit positions, Pulsar receive/acknowledge, Kinesis shard/sequence resume, and Debezium/Postgres CDC normalization. Implement all provider imports inside initialization paths. Kafka/Redpanda use the Kafka protocol and offload the synchronous `kafka-python` client; Pulsar and Kinesis follow the same `start`/`__anext__`/`ack`/`stop` contract. CDC accepts injected iterators/clients for deterministic tests and uses source LSN/transaction keys for stable IDs.

### Task 5: Wire public exports and optional dependency metadata

**Files:**
- Modify: `src/synapsekit/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_core_import_without_extras.py`
- Modify: `tests/test_public_api_surface.py`
- Modify: `tests/preflight/test_preflight.py`

Add lazy top-level exports for the streaming API and keep `import synapsekit` working with all provider modules blocked. Add provider-specific optional extras and dependency-module mappings only if needed by the source implementation; refresh `uv.lock` after manifest edits without changing unrelated dependency choices.

### Task 6: Add the Kafka Testcontainers acceptance path and docs

**Files:**
- Create: `tests/streaming/test_kafka_ingestion_integration.py`
- Create: `docs/streaming-kafka.md`
- Create: `examples/live_rag_kafka.py`
- Modify: `.github/workflows/ci.yml`

Use `pytest.importorskip` for provider/Testcontainers modules and a bounded readiness loop. Produce a deterministic event, consume it into a persistent local mesh, query the event text, close/reopen the mesh, and verify the event ID appears once. Add the integration test to the existing integration CI job and document the live-RAG flow, checkpoint path, transformer contract, optional installs, and graceful shutdown.

### Task 7: Recheck the entire issue contract without committing

Run the focused streaming, world-model, mesh, public-import, and preflight suites; run the Kafka integration test where Docker and the optional dependencies are available; run Ruff, formatting, mypy, the async-blocking gate, `git diff --check`, and the full test suite where feasible. Re-read issue #891 and `CONTRIBUTING.md`, map each acceptance row to actual output, inspect reachability/public exports, and confirm the worktree contains no commit or push.

No commit or push is part of this implementation; preserve all pre-existing worktree modifications.
