"""Bounded Kafka topic loader."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from .base import Document


class KafkaLoader:
    """Read at most ``max_messages`` messages from a Kafka topic."""

    def __init__(
        self,
        bootstrap_servers: str | list[str] | None = None,
        topic: str = "",
        max_messages: int = 100,
        timeout_ms: int = 1000,
        group_id: str | None = None,
        auto_offset_reset: str = "earliest",
        host: str | list[str] | None = None,
        endpoint_url: str | list[str] | None = None,
        consumer: Any | None = None,
    ) -> None:
        bootstrap_servers = bootstrap_servers or host or endpoint_url
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers or host must be provided")
        if not topic:
            raise ValueError("topic must be provided")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0")

        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._max_messages = max_messages
        self._timeout_ms = timeout_ms
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
        self._consumer = consumer

    def load(self) -> list[Document]:
        consumer = self._consumer or self._build_consumer()
        documents: list[Document] = []
        try:
            while len(documents) < self._max_messages:
                batch = consumer.poll(
                    timeout_ms=self._timeout_ms,
                    max_records=self._max_messages - len(documents),
                )
                if not batch:
                    break
                messages = (
                    [message for message_list in batch.values() for message in message_list]
                    if isinstance(batch, Mapping)
                    else list(batch)
                )
                if not messages:
                    break
                for message in messages:
                    documents.append(self._message_to_document(message, len(documents)))
                    if len(documents) >= self._max_messages:
                        break
        finally:
            close = getattr(consumer, "close", None)
            if callable(close):
                close()
        return documents

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_consumer(self) -> Any:
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise ImportError("kafka-python required: pip install synapsekit[kafka]") from None
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self._bootstrap_servers,
            "enable_auto_commit": False,
            "auto_offset_reset": self._auto_offset_reset,
            "consumer_timeout_ms": self._timeout_ms,
        }
        if self._group_id is not None:
            kwargs["group_id"] = self._group_id
        return KafkaConsumer(self._topic, **kwargs)

    def _message_to_document(self, message: Any, row: int) -> Document:
        value = getattr(message, "value", message)
        text = self._value_to_text(value)
        headers: dict[str, str | None] = {}
        for key, header_value in getattr(message, "headers", []) or []:
            headers[str(key)] = self._decode_bytes(header_value)
        key = getattr(message, "key", None)
        metadata = {
            "source": "kafka",
            "row": row,
            "topic": getattr(message, "topic", self._topic),
            "partition": getattr(message, "partition", None),
            "offset": getattr(message, "offset", None),
            "timestamp": getattr(message, "timestamp", None),
            "key": self._decode_bytes(key),
            "headers": headers,
        }
        return Document(text=text, metadata=metadata)

    @classmethod
    def _value_to_text(cls, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        if isinstance(value, (Mapping, list, tuple)):
            return json.dumps(value, default=str, sort_keys=True)
        return str(value)

    @staticmethod
    def _decode_bytes(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
