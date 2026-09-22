"""Named facade for RAG applications that include visual documents."""

from __future__ import annotations

from .facade import RAG


class MultimodalRAG(RAG):
    """RAG facade with optional visual-page indexing and vision synthesis.

    ``RAG`` remains the backwards-compatible entry point; this named
    subclass makes multimodal intent explicit without changing defaults.
    """


__all__ = ["MultimodalRAG"]
