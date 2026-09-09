"""DynamoDB-backed graph checkpoints."""

from __future__ import annotations

from typing import Any

from ..._json import dumps_bytes as _json_dumps_bytes
from ..._json import loads as _json_loads
from ...memory.backends._common import AsyncCheckpointerMixin
from .base import BaseCheckpointer


class DynamoDBCheckpointer(AsyncCheckpointerMixin, BaseCheckpointer):
    """Persist graph checkpoints in DynamoDB or DynamoDB Local."""

    def __init__(
        self,
        table_name: str = "synapsekit_checkpoints",
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
        *,
        auto_create_table: bool = True,
        **client_options: Any,
    ) -> None:
        if not table_name:
            raise ValueError("table_name must be provided")
        self._table_name = table_name
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._auto_create_table = auto_create_table
        self._client_options = client_options
        self._resource: Any | None = None
        self._table: Any | None = None

    def _ensure_table(self) -> Any:
        if self._table is not None:
            return self._table
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for DynamoDBCheckpointer; "
                "install it with `pip install synapsekit[dynamodb]`"
            ) from None
        options = dict(self._client_options)
        options.update(region_name=self._region_name, endpoint_url=self._endpoint_url)
        self._resource = boto3.resource("dynamodb", **options)
        table = self._resource.Table(self._table_name)
        if self._auto_create_table:
            try:
                table.meta.client.describe_table(TableName=self._table_name)
            except table.meta.client.exceptions.ResourceNotFoundException:
                table = self._resource.create_table(
                    TableName=self._table_name,
                    KeySchema=[
                        {"AttributeName": "pk", "KeyType": "HASH"},
                        {"AttributeName": "sk", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "pk", "AttributeType": "S"},
                        {"AttributeName": "sk", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                table.meta.client.get_waiter("table_exists").wait(TableName=self._table_name)
        self._table = table
        return table

    @staticmethod
    def _key(graph_id: str) -> dict[str, str]:
        return {"pk": f"graph#{graph_id}", "sk": "checkpoint"}

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        self._ensure_table().put_item(
            Item={
                **self._key(graph_id),
                "graph_id": graph_id,
                "step": int(step),
                "state": _json_dumps_bytes(state),
            }
        )

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        item = self._ensure_table().get_item(Key=self._key(graph_id)).get("Item")
        if item is None:
            return None
        state = item["state"]
        if not isinstance(state, (str, bytes)):
            state = bytes(state)
        return int(item["step"]), dict(_json_loads(state))

    def delete(self, graph_id: str) -> None:
        self._ensure_table().delete_item(Key=self._key(graph_id))

    def close(self) -> None:
        if self._resource is not None:
            client = getattr(getattr(self._resource, "meta", None), "client", None)
            close = getattr(client, "close", None)
            if close is not None:
                close()
        self._table = None
        self._resource = None
