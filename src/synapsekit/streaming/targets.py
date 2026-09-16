"""Idempotent document targets for streaming ingestion."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ..loaders.base import Document
from .ingestor import _event_document


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _ingestion_id(document: Document) -> str | None:
    for key in ("ingestion_id", "event_id", "document_id", "id"):
        value = document.metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _latest_documents(documents: Iterable[Document]) -> list[Document]:
    """Keep the last document for each stable ID while preserving order."""

    identified: dict[str, Document] = {}
    anonymous: list[Document] = []
    for document in documents:
        stable_id = next(
            (
                str(document.metadata[key])
                for key in ("document_id", "ingestion_id", "event_id", "id")
                if document.metadata.get(key) is not None
            ),
            None,
        )
        if stable_id is None:
            anonymous.append(document)
        else:
            identified[stable_id] = document
    return [*identified.values(), *anonymous]


async def _delete_by_metadata(store: Any, key: str, ids: set[str]) -> int:
    if not ids:
        return 0
    delete = getattr(store, "delete_by_metadata", None)
    if not callable(delete):
        return 0
    try:
        return int(await asyncio.to_thread(delete, key, ids))
    except NotImplementedError:
        return 0


async def _delete_by_ingestion_id(store: Any, ids: set[str]) -> int:
    return await _delete_by_metadata(store, "ingestion_id", ids)


def _document_ids(documents: Iterable[Document]) -> set[str]:
    return {
        str(document.metadata["document_id"])
        for document in documents
        if document.metadata.get("document_id") is not None
    }


class VectorStoreSink:
    """Upsert streaming documents into a SynapseKit vector store.

    SynapseKit's existing vector-store contract exposes ``add`` and optional
    ``delete_by_metadata`` rather than a provider-specific ID API. The sink uses
    the reserved ``ingestion_id`` metadata field to replace redelivered records
    before adding their new embeddings.
    """

    def __init__(self, vector_store: Any) -> None:
        self.vector_store = vector_store

    async def upsert(self, documents: list[Document]) -> None:
        documents = _latest_documents(document for document in documents if document.text.strip())
        if not documents:
            return
        deleted = [document for document in documents if document.metadata.get("deleted")]
        documents = [document for document in documents if not document.metadata.get("deleted")]
        await _delete_by_metadata(
            self.vector_store,
            "document_id",
            _document_ids([*deleted, *documents]),
        )
        if not documents:
            return
        ids = {
            stable_id
            for document in documents
            if (stable_id := _ingestion_id(document)) is not None
        }
        await _delete_by_ingestion_id(self.vector_store, ids)
        add = getattr(self.vector_store, "add", None)
        if not callable(add):
            raise TypeError("vector_store must expose add(texts, metadata)")
        result = add(
            [document.text for document in documents],
            [dict(document.metadata) for document in documents],
        )
        await _maybe_await(result)


class WorldModelSink:
    """Idempotently upsert documents into a :class:`WorldModelRAG` facade."""

    def __init__(self, world_model: Any) -> None:
        self.world_model = world_model

    async def upsert(self, documents: list[Document]) -> None:
        documents = _latest_documents(document for document in documents if document.text.strip())
        if not documents:
            return
        vector_store = getattr(self.world_model, "vector_store", None)
        deleted = [document for document in documents if document.metadata.get("deleted")]
        documents = [document for document in documents if not document.metadata.get("deleted")]
        await _delete_by_metadata(
            vector_store,
            "document_id",
            _document_ids([*deleted, *documents]),
        )
        if not documents:
            return
        ids = {
            stable_id
            for document in documents
            if (stable_id := _ingestion_id(document)) is not None
        }
        await _delete_by_ingestion_id(vector_store, ids)
        ingest = getattr(self.world_model, "ingest", None)
        if not callable(ingest):
            raise TypeError("world_model must expose ingest(documents)")
        await _maybe_await(ingest(documents))


class KnowledgeMeshSink:
    """Upsert streaming documents into a live ``KnowledgeMesh``.

    Stream records are represented as synthetic mesh paths. They are registered
    in the mesh metadata index as active chunks so the normal citation/filtering
    path can retrieve them, while the underlying world model receives the same
    documents and their event timestamps.
    """

    def __init__(self, mesh: Any) -> None:
        self.mesh = mesh

    async def upsert(self, documents: list[Document]) -> None:
        documents = _latest_documents(document for document in documents if document.text.strip())
        if not documents:
            return
        prepared = [self._ensure_mesh_metadata(document) for document in documents]
        deleted = [document for document in prepared if document.metadata.get("deleted")]
        active_documents = [
            document for document in prepared if not document.metadata.get("deleted")
        ]
        grouped: dict[str, list[Document]] = defaultdict(list)
        for document in active_documents:
            grouped[str(document.metadata["path"])].append(document)
        for document in deleted:
            grouped.setdefault(str(document.metadata["path"]), [])

        mesh_store = getattr(self.mesh, "store", None)
        stale_by_path: dict[str, set[str]] = {}
        if mesh_store is not None:
            for path in grouped:
                active = getattr(mesh_store, "active_chunk_ids_for_path", None)
                if callable(active):
                    old_ids = await asyncio.to_thread(active, path)
                    new_ids = {str(document.metadata["chunk_id"]) for document in grouped[path]}
                    stale_by_path[path] = set(old_ids) - new_ids

        rag = getattr(self.mesh, "rag", None)
        vector_store = getattr(rag, "vector_store", None)
        ids = {
            stable_id for document in prepared if (stable_id := _ingestion_id(document)) is not None
        }
        await _delete_by_ingestion_id(vector_store, ids)
        await _delete_by_metadata(
            vector_store,
            "document_id",
            _document_ids([*deleted, *active_documents]),
        )
        ingest = getattr(rag, "ingest", None)
        if not callable(ingest):
            raise TypeError("mesh.rag must expose ingest(documents)")
        if active_documents:
            await _maybe_await(ingest(active_documents))

        if mesh_store is not None:
            mark = getattr(mesh_store, "mark_file_chunks", None)
            if callable(mark):
                for path, path_documents in grouped.items():
                    await asyncio.to_thread(mark, path, path_documents)
            stale_ids = {chunk_id for path_ids in stale_by_path.values() for chunk_id in path_ids}
            if stale_ids:
                delete = getattr(rag, "delete_by_metadata", None)
                if callable(delete):
                    await asyncio.to_thread(delete, "chunk_id", stale_ids)
            save = getattr(self.mesh, "_save_vector_store", None)
            if callable(save):
                await asyncio.to_thread(save)

    @staticmethod
    def _ensure_mesh_metadata(document: Document) -> Document:
        metadata = dict(document.metadata)
        stable_id = _ingestion_id(document) or str(metadata.get("source", "stream-document"))
        path = str(metadata.get("path") or metadata.get("source") or f"stream://{stable_id}")
        metadata.setdefault("source", path)
        metadata["path"] = path
        metadata.setdefault("chunk_id", stable_id)
        metadata.setdefault("ingestion_id", stable_id)
        return Document(document.text, metadata)


# Target aliases make the same objects discoverable under either vocabulary.
VectorStoreTarget = VectorStoreSink
WorldModelTarget = WorldModelSink
KnowledgeMeshTarget = KnowledgeMeshSink
MeshTarget = KnowledgeMeshSink


# Public name for callers that want the default without importing the orchestration module.
def default_document_transform(event: Any) -> Document:
    """Convert a normalized event to a metadata-rich ``Document``."""

    return _event_document(event)


__all__ = [
    "KnowledgeMeshSink",
    "KnowledgeMeshTarget",
    "MeshTarget",
    "VectorStoreSink",
    "VectorStoreTarget",
    "WorldModelSink",
    "WorldModelTarget",
    "default_document_transform",
]
