# Persistent memory backends

`AgentMemory` supports one contract for episodic and semantic records. Every persistent backend stores the record ID, agent ID, content, memory type, embedding, creation/access timestamps, access count, optional TTL, and JSON metadata. `fetch()` excludes expired records by default; `fetch(..., include_expired=True)` includes them. `prune_expired()` removes expired records.

Install only the provider you use:

| Backend | Extra | AgentMemory backend | Graph checkpointer | Local integration target |
|---|---|---|---|---|
| SQLite | core | `sqlite` | `SQLiteCheckpointer` | local file |
| Redis | `redis` | `redis` | `RedisCheckpointer` | Redis container |
| PostgreSQL | `postgres` | `postgres` | `PostgresCheckpointer` | PostgreSQL container |
| MongoDB | `mongodb` | `mongodb` | `MongoDBCheckpointer` | `mongo:7` |
| Cassandra / ScyllaDB | `cassandra` | `cassandra` or `scylla` | `CassandraCheckpointer` or `ScyllaCheckpointer` | `cassandra:4.1` |
| DynamoDB | `dynamodb` | `dynamodb` | `DynamoDBCheckpointer` | DynamoDB Local |
| Firestore | `firestore` | `firestore` | `FirestoreCheckpointer` | Firestore emulator |
| Azure Cosmos DB | `cosmos` | `cosmos` or `cosmosdb` | `CosmosDBCheckpointer` | Linux Cosmos emulator |

Provider SDKs are imported lazily. Importing `synapsekit` does not require any provider extra. Configure provider-specific values through `backend_options`:

```python
from synapsekit import AgentMemory

memory = AgentMemory(
    backend="mongodb",
    backend_options={
        "uri": "mongodb://localhost:27017",
        "database": "my_agent",
        "collection": "memories",
    },
)

memory = AgentMemory(
    backend="dynamodb",
    backend_options={
        "table_name": "agent_memory",
        "region_name": "us-east-1",
        # Set endpoint_url for DynamoDB Local; omit it in AWS.
        "endpoint_url": "http://localhost:8000",
    },
)
```

The Cassandra backend uses the Cassandra protocol and is compatible with ScyllaDB. Cassandra/Scylla identifiers are validated before being used in schema statements; record values are parameterized.

Firestore uses `google.cloud.firestore_v1.AsyncClient` for AgentMemory operations. Set `FIRESTORE_EMULATOR_HOST` when using the emulator, or provide `project_id` and `credentials_path` for Google Cloud.

Cosmos DB creates the configured database and container on first use with an `/agent_id` partition key for memory and `/graph_id` for checkpoints. Use `verify_ssl=False` only with a local emulator after configuring its certificate; keep TLS verification enabled for Azure.

Graph checkpointers retain the existing synchronous `save()`, `load()`, and `delete()` contract. The new provider checkpointers additionally expose `asave()`, `aload()`, and `adelete()`; these run their synchronous SDK operations off the event loop.
