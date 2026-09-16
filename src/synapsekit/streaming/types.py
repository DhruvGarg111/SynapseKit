"""Normalized events used by the streaming ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC)


def _derive_event_id(
    payload: Any,
    source: str,
    stream: str,
    partition: str | int | None,
    offset: int | str | None,
) -> str:
    """Derive a process-stable fallback ID for sources without native IDs."""

    canonical = json.dumps(
        {
            "offset": offset,
            "partition": partition,
            "payload": payload,
            "source": source,
            "stream": stream,
        },
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A source-neutral event with a stable identity and resume position.

    ``payload`` is deliberately untyped because provider clients return JSON,
    bytes, strings, or provider-specific objects. Source adapters normalize their
    native records into this shape while the transformer decides how to produce a
    :class:`~synapsekit.loaders.base.Document`.
    """

    payload: Any
    source: str = "custom"
    stream: str = "default"
    event_id: str | None = None
    partition: str | int | None = None
    offset: int | str | None = None
    timestamp: datetime | str | None = None
    key: Any = None
    headers: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ingest_time: datetime | str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must not be empty")
        if not self.stream:
            raise ValueError("stream must not be empty")
        if self.event_id is None:
            event_id = _derive_event_id(
                self.payload,
                self.source,
                self.stream,
                self.partition,
                self.offset,
            )
        else:
            event_id = str(self.event_id)
            if not event_id:
                raise ValueError("event_id must not be empty")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "timestamp", _coerce_datetime(self.timestamp))
        object.__setattr__(
            self,
            "ingest_time",
            _coerce_datetime(self.ingest_time) or datetime.now(UTC),
        )
        object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def id(self) -> str:
        """Alias for ``event_id`` used by generic source code."""

        assert self.event_id is not None
        return self.event_id

    @property
    def stable_id(self) -> str:
        """Return the ID used for deduplication and target upserts."""

        return self.id

    @property
    def data(self) -> Any:
        """Alias for ``payload`` used by data-oriented callers."""

        return self.payload

    @property
    def value(self) -> Any:
        """Alias for ``payload`` used by broker-oriented callers."""

        return self.payload

    @property
    def event_time(self) -> datetime | None:
        """Alias for the source event timestamp."""

        return cast(datetime | None, self.timestamp)

    @property
    def partition_key(self) -> str:
        """Return a stable string key for a source partition/shard."""

        return "default" if self.partition is None else str(self.partition)

    @property
    def stream_key(self) -> str:
        """Return the checkpoint namespace for this event stream."""

        namespace = self.metadata.get("checkpoint_namespace")
        return str(namespace) if namespace else f"{self.source}:{self.stream}"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The last successfully applied event in one stream partition."""

    stream: str
    partition: str
    offset: int | str | None
    event_id: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC), compare=False)

    @classmethod
    def from_event(cls, event: StreamEvent) -> Checkpoint:
        """Create a checkpoint at ``event``'s source position."""

        return cls(
            stream=event.stream_key,
            partition=event.partition_key,
            offset=event.offset,
            event_id=event.id,
        )

    @property
    def stream_key(self) -> str:
        """Alias for the stream namespace."""

        return self.stream

    @property
    def partition_key(self) -> str:
        """Alias for the partition namespace."""

        return self.partition

    @property
    def position(self) -> int | str | None:
        """Alias for the source cursor stored in ``offset``."""

        return self.offset


# Compatibility aliases for callers that use the longer names from the issue.
StreamCheckpoint = Checkpoint
