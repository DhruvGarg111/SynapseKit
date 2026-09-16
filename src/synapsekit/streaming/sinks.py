"""Compatibility exports for streaming ingestion targets."""

from .targets import (
    KnowledgeMeshSink,
    KnowledgeMeshTarget,
    MeshTarget,
    VectorStoreSink,
    VectorStoreTarget,
    WorldModelSink,
    WorldModelTarget,
    default_document_transform,
)

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
