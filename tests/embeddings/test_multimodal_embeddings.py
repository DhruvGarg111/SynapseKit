from __future__ import annotations

import numpy as np
import pytest

from synapsekit.embeddings.multimodal import BaseMultimodalEmbeddings


class FakeMultimodalEmbeddings(BaseMultimodalEmbeddings):
    async def _embed_images(self, images):
        return [np.ones((2, 3), dtype=np.float32) for _ in images]

    async def _embed_texts(self, texts):
        return [np.ones((1, 3), dtype=np.float32) for _ in texts]


@pytest.mark.asyncio
async def test_multimodal_embedding_single_item_helpers_delegate_to_batches():
    backend = FakeMultimodalEmbeddings()

    image = await backend.embed_image(b"png")
    text = await backend.embed_text("what is shown?")

    assert image.shape == (2, 3)
    assert text.shape == (1, 3)


@pytest.mark.asyncio
async def test_multimodal_embedding_empty_batches_are_empty_lists():
    backend = FakeMultimodalEmbeddings()

    assert await backend.embed_images([]) == []
    assert await backend.embed_texts([]) == []


@pytest.mark.asyncio
async def test_multimodal_embedding_validates_batch_output_count():
    class BadBackend(BaseMultimodalEmbeddings):
        async def _embed_images(self, images):
            return []

        async def _embed_texts(self, texts):
            return []

    backend = BadBackend()
    with pytest.raises(ValueError, match="one embedding per input"):
        await backend.embed_images([b"one"])
