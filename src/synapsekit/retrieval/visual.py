"""Visual-document retrieval using late-interaction MaxSim scoring."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..embeddings.multimodal import BaseMultimodalEmbeddings
from ..loaders.base import Document
from ..loaders.visual import DefaultPageRenderer, PageRenderer, VisualPage


def maxsim(query_embeddings: Any, page_embeddings: Any) -> float:
    """Score a query/page pair with ColPali-style late interaction.

    For every query token, the best matching page token is selected and those
    maxima are summed. Inputs must be ``(tokens, dimensions)`` matrices; the
    embedding backend is responsible for token normalization.
    """
    query = np.asarray(query_embeddings, dtype=np.float32)
    page = np.asarray(page_embeddings, dtype=np.float32)
    if query.ndim != 2 or page.ndim != 2:
        raise ValueError("MaxSim inputs must be 2D token matrices")
    if query.shape[1] != page.shape[1]:
        raise ValueError(
            f"MaxSim inputs must have matching dimensions, got {query.shape[1]} and {page.shape[1]}"
        )
    if query.shape[0] == 0 or page.shape[0] == 0:
        return 0.0
    similarities = query @ page.T
    return float(np.max(similarities, axis=1).sum())


class VisualDocumentRetriever:
    """Index rendered document pages and retrieve them with MaxSim.

    The retriever deliberately keeps the image alongside each result. This
    lets a vision-capable answer model inspect the original page instead of
    relying on lossy OCR or extracted text alone.
    """

    def __init__(
        self,
        embedding_backend: BaseMultimodalEmbeddings,
        *,
        renderer: PageRenderer | None = None,
    ) -> None:
        self._embeddings = embedding_backend
        self._renderer = renderer or DefaultPageRenderer()
        self._pages: list[VisualPage] = []
        self._page_embeddings: list[np.ndarray] = []
        self._lock = asyncio.Lock()

    @property
    def embedding_backend(self) -> BaseMultimodalEmbeddings:
        """Return the configured multimodal embedding backend."""
        return self._embeddings

    @property
    def page_count(self) -> int:
        """Number of indexed visual pages."""
        return len(self._pages)

    @property
    def pages(self) -> tuple[VisualPage, ...]:
        """Read-only snapshot of indexed pages in insertion order."""
        return tuple(self._pages)

    async def add_pages(self, pages: Sequence[VisualPage]) -> None:
        """Embed and append visual pages in their input order."""
        values = list(pages)
        if not values:
            return

        async with self._lock:
            vectors = await self._embeddings.embed_images([page.image for page in values])
            if len(vectors) != len(values):
                raise ValueError(
                    "Visual embedding backend must return one embedding per input "
                    f"(expected {len(values)}, got {len(vectors)})"
                )
            start_page = len(self._pages) + 1
            for offset, (page, vector) in enumerate(zip(values, vectors, strict=True)):
                metadata = dict(page.metadata)
                if metadata.get("page") is None:
                    metadata["page"] = start_page + offset
                if metadata.get("bbox") is None:
                    metadata["bbox"] = "full-page"
                if metadata.get("locator") is None:
                    source = Path(str(metadata.get("source", "visual-document"))).name
                    metadata["locator"] = f"{source} page {metadata['page']}"
                self._pages.append(VisualPage(image=page.image, text=page.text, metadata=metadata))
                self._page_embeddings.append(np.asarray(vector, dtype=np.float32))

    async def add_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[VisualPage]:
        """Render, index, and return pages from a local visual document."""
        pages = await self.render_file(path, metadata=metadata)
        await self.add_pages(pages)
        return pages

    async def render_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[VisualPage]:
        """Render a local visual document without changing the index."""
        pages = await self._renderer.render(path)
        if metadata:
            pages = [
                VisualPage(
                    image=page.image,
                    text=page.text,
                    metadata={**page.metadata, **metadata},
                )
                for page in pages
            ]
        return pages

    async def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return visual results with score, image, text, and source metadata."""
        if top_k <= 0:
            return []
        query_embedding = await self._embeddings.embed_text(query)
        async with self._lock:
            pages = list(self._pages)
            page_embeddings = list(self._page_embeddings)

        def _score_all() -> list[tuple[int, float]]:
            return [
                (index, maxsim(query_embedding, embedding))
                for index, embedding in enumerate(page_embeddings)
            ]

        scored = await asyncio.to_thread(_score_all)
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            {
                "text": pages[index].text,
                "score": float(score),
                "metadata": dict(pages[index].metadata),
                "image": pages[index].image,
            }
            for index, score in scored[: min(top_k, len(scored))]
        ]

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return only page text for compatibility with text retrievers."""
        results = await self.retrieve_with_scores(query, top_k=top_k)
        return [str(result["text"]) for result in results]

    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[Document]:
        """Return scored pages as ``Document`` objects."""
        results = await self.retrieve_with_scores(query, top_k=top_k)
        return [
            Document(
                text=str(result["text"]),
                metadata={
                    **dict(result["metadata"]),
                    "score": result["score"],
                    "image": result["image"],
                },
            )
            for result in results
        ]

    async def clear(self) -> None:
        """Remove all indexed pages and embeddings."""
        async with self._lock:
            self._pages.clear()
            self._page_embeddings.clear()


__all__ = ["VisualDocumentRetriever", "maxsim"]
