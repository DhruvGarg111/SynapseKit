# Data loaders

Issue #889 adds the following optional integrations. Every loader returns one `Document` per message, record, file, row, or bounded event and provides `aload()` for executor-backed asynchronous use.

| Loader | Import | Extra | Required source | Endpoint/host override |
| --- | --- | --- | --- | --- |
| Gmail | `GmailLoader` | `gmail` | Google credentials plus message query, message ID, or thread ID | Google API service is injectable |
| Linear | `LinearLoader` | `linear` | Linear API key | `endpoint_url` |
| Asana | `AsanaLoader` | `asana` | access token plus project/workspace ID | `endpoint_url` |
| ClickUp | `ClickUpLoader` | `clickup` | API token plus list/team ID | `endpoint_url` |
| Monday.com | `MondayLoader` | `monday` | API key and board ID | `endpoint_url` |
| Box | `BoxLoader` | `box` | access token and folder ID | `endpoint_url` |
| Figma | `FigmaLoader` | `figma` | access token and file key | `endpoint_url` |
| Zoom | `ZoomLoader` | `zoom` | access token and user ID; cloud recording transcript downloads are enabled by default | `endpoint_url` |
| Shopify | `ShopifyLoader` | `shopify` | shop URL and access token | `endpoint_url` or `host` |
| Databricks | `DatabricksLoader` | `databricks` | SQL host, HTTP path, token, and query | `server_hostname`, `host`, or `endpoint_url` |
| Apache Iceberg | `IcebergLoader` | `iceberg` | catalog table identifier | `endpoint_url` in catalog properties |
| Delta Lake | `DeltaLakeLoader` (`DeltaLoader`) | `delta` | local/object-store table URI | `endpoint_url` in storage options |
| GraphQL | `GraphQLLoader` | `graphql` | endpoint URL and query | `endpoint_url` |
| Kafka | `KafkaLoader` | `kafka` | bootstrap servers, topic, and message bound | `bootstrap_servers`, `host`, or `endpoint_url` |

HTTP loaders accept `client=` for a preconfigured blocking HTTP client, and table/queue loaders accept their provider client where applicable. External services are never contacted during import or tests unless a caller invokes `load()`/`aload()` with credentials and a real backend.
