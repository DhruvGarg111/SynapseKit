"""Contracts for image-and-text multi-vector embedding backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ImageInput = str | Path | bytes | Any


class BaseMultimodalEmbeddings(ABC):
    """Async multi-vector embedding contract for visual retrieval.

    Each input produces a two-dimensional ``(tokens, dimensions)`` array. The
    token axis is intentionally variable because late-interaction models such
    as ColPali represent each page and query with multiple vectors rather than
    one pooled vector.
    """

    dimensions: int | None = None

    @abstractmethod
    async def _embed_images(self, images: list[ImageInput]) -> Sequence[Any]:
        """Return one raw token-vector matrix for every image."""
        raise NotImplementedError

    @abstractmethod
    async def _embed_texts(self, texts: list[str]) -> Sequence[Any]:
        """Return one raw token-vector matrix for every text query."""
        raise NotImplementedError

    async def embed_images(self, images: Sequence[ImageInput]) -> list[np.ndarray]:
        """Embed images and validate/normalize their token vectors."""
        values = list(images)
        if not values:
            return []
        raw = await self._embed_images(values)
        return self._validate_batch(raw, expected=len(values), modality="image")

    async def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed text queries and validate/normalize their token vectors."""
        values = list(texts)
        if not values:
            return []
        raw = await self._embed_texts(values)
        return self._validate_batch(raw, expected=len(values), modality="text")

    async def embed_image(self, image: ImageInput) -> np.ndarray:
        """Embed one image and return its ``(tokens, dimensions)`` matrix."""
        return (await self.embed_images([image]))[0]

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed one text query and return its ``(tokens, dimensions)`` matrix."""
        return (await self.embed_texts([text]))[0]

    def _validate_batch(
        self,
        raw: Sequence[Any],
        *,
        expected: int,
        modality: str,
    ) -> list[np.ndarray]:
        if len(raw) != expected:
            raise ValueError(
                f"{type(self).__name__} must return one embedding per input "
                f"({modality}: expected {expected}, got {len(raw)})"
            )

        validated: list[np.ndarray] = []
        for index, item in enumerate(raw):
            array = np.asarray(item, dtype=np.float32)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(
                    f"{type(self).__name__} returned invalid {modality} embedding "
                    f"at index {index}; expected a non-empty 2D array"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"{type(self).__name__} returned non-finite {modality} embedding "
                    f"at index {index}"
                )

            norms = np.linalg.norm(array, axis=1, keepdims=True)
            zero_rows = int(np.count_nonzero(norms == 0))
            if zero_rows:
                logger.warning(
                    "%s returned %d degenerate zero-norm %s token vector(s) at index %d; "
                    "left unnormalized (zero) and will contribute zero similarity in MaxSim",
                    type(self).__name__,
                    zero_rows,
                    modality,
                    index,
                )
            norms = np.where(norms == 0, 1.0, norms)
            normalized = (array / norms).astype(np.float32)
            if self.dimensions is None:
                self.dimensions = int(normalized.shape[1])
            elif normalized.shape[1] != self.dimensions:
                raise ValueError(
                    f"{type(self).__name__} returned {normalized.shape[1]} dimensions "
                    f"but dimensions={self.dimensions}"
                )
            validated.append(normalized)
        return validated


# Short name for callers that prefer the issue terminology.
MultimodalEmbeddings = BaseMultimodalEmbeddings
ImageTextEmbeddings = BaseMultimodalEmbeddings

__all__ = [
    "BaseMultimodalEmbeddings",
    "ImageInput",
    "ImageTextEmbeddings",
    "MultimodalEmbeddings",
]
