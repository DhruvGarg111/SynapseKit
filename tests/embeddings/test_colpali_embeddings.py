from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from synapsekit.embeddings.colpali import ColPaliEmbeddings, ColQwenEmbeddings


class FakeProcessor:
    def __init__(self):
        self.image_inputs = None
        self.query_inputs = None

    def process_images(self, images):
        self.image_inputs = images
        return {"images": images}

    def process_queries(self, texts):
        self.query_inputs = texts
        return {"queries": texts}


class FakeModel:
    def __init__(self, output_name="embeddings"):
        self.output_name = output_name
        self.calls = []
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **batch):
        self.calls.append(batch)
        count = len(next(iter(batch.values())))
        values = np.zeros((count, 2, 2), dtype=np.float32)
        values[:, 0, 0] = 3.0
        values[:, 1, 1] = 4.0
        return SimpleNamespace(**{self.output_name: values})


@pytest.mark.asyncio
async def test_colpali_uses_injected_processor_and_model_for_image_and_text_batches():
    processor = FakeProcessor()
    model = FakeModel()
    backend = ColPaliEmbeddings(processor=processor, model_backend=model)

    image_vectors = await backend.embed_images([b"one", b"two"])
    text_vectors = await backend.embed_texts(["one", "two"])

    assert len(processor.image_inputs) == len(processor.query_inputs) == 2
    assert len(image_vectors) == len(text_vectors) == 2
    assert image_vectors[0].shape == (2, 2)
    assert np.allclose(np.linalg.norm(text_vectors[0], axis=1), 1.0)
    assert model.calls


@pytest.mark.asyncio
async def test_colpali_accepts_last_hidden_state_output():
    backend = ColPaliEmbeddings(
        processor=FakeProcessor(),
        model_backend=FakeModel(output_name="last_hidden_state"),
    )

    result = await backend.embed_text("query")

    assert result.shape == (2, 2)
    assert np.allclose(result[0], [1.0, 0.0])


def test_colqwen_has_colqwen_default_model():
    backend = ColQwenEmbeddings(processor=FakeProcessor(), model_backend=FakeModel())

    assert backend.model == "vidore/colqwen2-v1.0-hf"


def test_multimodal_embedding_exports_are_lazy_and_public():
    import synapsekit
    import synapsekit.embeddings as embeddings

    assert synapsekit.ColPaliEmbeddings is ColPaliEmbeddings
    assert synapsekit.ColQwenEmbeddings is ColQwenEmbeddings
    assert embeddings.MultimodalEmbeddings is not None


def test_colpali_import_does_not_eagerly_import_torch_or_transformers():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import synapsekit.embeddings.colpali; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"
