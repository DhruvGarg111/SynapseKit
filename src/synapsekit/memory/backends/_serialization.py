from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Any

from ..._json import dumps as _json_dumps
from ..._json import loads as _json_loads
from ..base import MemoryRecord


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def encode_embedding(values: list[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"!{len(values)}d", *(float(value) for value in values))


def decode_embedding(value: Any, dimension: int | None = None) -> list[float]:
    if isinstance(value, memoryview):
        value = value.tobytes()
    elif not isinstance(value, (bytes, bytearray, list)) and hasattr(value, "__bytes__"):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        if not value:
            return []
        if len(value) % 8:
            raise ValueError("embedding binary payload has an invalid length")
        actual_dimension = len(value) // 8
        if dimension is not None and dimension != actual_dimension:
            raise ValueError("embedding dimension does not match its binary payload")
        return list(struct.unpack(f"!{actual_dimension}d", bytes(value)))
    values = [float(item) for item in value]
    if dimension is not None and dimension != len(values):
        raise ValueError("embedding dimension does not match its list payload")
    return values


def memory_record_to_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "agent_id": record.agent_id,
        "content": record.content,
        "memory_type": record.memory_type,
        "embedding": encode_embedding(record.embedding),
        "embedding_dimension": len(record.embedding),
        "created_at": _utc(record.created_at).isoformat(),
        "accessed_at": _utc(record.accessed_at).isoformat(),
        "access_count": int(record.access_count),
        "ttl_days": record.ttl_days,
        "metadata": _json_dumps(record.metadata),
    }


def memory_record_from_payload(payload: dict[str, Any]) -> MemoryRecord:
    created_at = datetime.fromisoformat(str(payload["created_at"]))
    accessed_at = datetime.fromisoformat(str(payload["accessed_at"]))
    metadata_value = payload.get("metadata", "{}")
    if isinstance(metadata_value, dict):
        metadata = metadata_value
    else:
        metadata = _json_loads(metadata_value)
    return MemoryRecord(
        id=str(payload["id"]),
        agent_id=str(payload["agent_id"]),
        content=str(payload.get("content", "")),
        memory_type=payload["memory_type"],  # type: ignore[arg-type]
        embedding=decode_embedding(
            payload.get("embedding", b""), payload.get("embedding_dimension")
        ),
        created_at=_utc(created_at),
        accessed_at=_utc(accessed_at),
        access_count=int(payload.get("access_count", 0)),
        ttl_days=None if payload.get("ttl_days") is None else int(payload["ttl_days"]),
        metadata=dict(metadata),
    )
