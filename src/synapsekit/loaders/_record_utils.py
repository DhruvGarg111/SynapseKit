"""Private helpers shared by integration-backed document loaders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .base import Document


def as_record(value: Any) -> dict[str, Any] | None:
    """Convert common SDK record shapes to a plain mapping."""
    if isinstance(value, Mapping):
        return dict(value)

    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                return None
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return None


def format_value(value: Any) -> str:
    """Format nested provider values without losing their structure."""
    if isinstance(value, (Mapping, list, tuple, set)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def build_record_text(record: Mapping[str, Any], text_fields: Sequence[str] | None = None) -> str:
    """Build deterministic labelled text from a provider record."""
    fields = text_fields if text_fields is not None else list(record)
    parts: list[str] = []
    for field in fields:
        value = record.get(field)
        if value in (None, ""):
            continue
        parts.append(f"{field}: {format_value(value)}")
    return "\n".join(parts)


def build_record_metadata(
    record: Mapping[str, Any],
    source: str,
    row: int,
    metadata_fields: Sequence[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Lift provider fields into metadata while preserving source context."""
    metadata: dict[str, Any] = {"row": row, **extra}
    if metadata_fields is None:
        metadata.update(record)
    else:
        metadata.update({field: record[field] for field in metadata_fields if field in record})
    metadata["source"] = source
    return metadata


def records_to_documents(
    records: Sequence[Any],
    source: str,
    text_fields: Sequence[str] | None = None,
    metadata_fields: Sequence[str] | None = None,
    limit: int | None = None,
    **extra: Any,
) -> list[Document]:
    """Convert records to one document per non-empty record."""
    documents: list[Document] = []
    for index, value in enumerate(records):
        if limit is not None and len(documents) >= limit:
            break
        record = as_record(value)
        if record is None:
            continue
        text = build_record_text(record, text_fields)
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                metadata=build_record_metadata(
                    record,
                    source,
                    index,
                    metadata_fields,
                    **extra,
                ),
            )
        )
    return documents


def response_json(response: Any) -> Any:
    """Raise provider HTTP failures and return decoded JSON."""
    response.raise_for_status()
    return response.json()
