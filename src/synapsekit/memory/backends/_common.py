from __future__ import annotations

import base64
import re
from typing import Any, Protocol

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, label: str) -> str:
    """Validate a provider identifier that cannot be parameterized."""
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, and underscores")
    return value


def encode_document_id(value: str) -> str:
    """Encode arbitrary user IDs for document stores with restricted IDs."""
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return f"sk_{encoded.rstrip('=')}"


def decode_document_id(value: str) -> str:
    if not value.startswith("sk_"):
        return value
    encoded = value[3:]
    encoded += "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")


def memory_record_document_id(agent_id: str, record_id: str) -> str:
    """Build a collision-free document key for an agent's record."""
    return encode_document_id(f"{agent_id}\x00{record_id}")


class AsyncCheckpointerMixin:
    """Thread-isolate the synchronous graph checkpoint contract."""

    async def asave(
        self: _SyncCheckpointer, graph_id: str, step: int, state: dict[str, Any]
    ) -> None:
        import asyncio

        await asyncio.to_thread(self.save, graph_id, step, state)

    async def aload(self: _SyncCheckpointer, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        import asyncio

        return await asyncio.to_thread(self.load, graph_id)

    async def adelete(self: _SyncCheckpointer, graph_id: str) -> None:
        import asyncio

        await asyncio.to_thread(self.delete, graph_id)


class _SyncCheckpointer(Protocol):
    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None: ...

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None: ...

    def delete(self, graph_id: str) -> None: ...
