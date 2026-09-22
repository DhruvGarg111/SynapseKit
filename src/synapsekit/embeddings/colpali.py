"""Lazy ColPali and ColQwen multimodal embedding adapters."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .multimodal import BaseMultimodalEmbeddings, ImageInput


class ColPaliEmbeddings(BaseMultimodalEmbeddings):
    """Embed page images and text queries with a ColPali-family model.

    ``torch`` and ``transformers`` are imported only when the first embedding
    request needs them. Supplying ``processor`` and ``model_backend`` is useful
    for tests and for applications that manage model lifecycles themselves.
    """

    def __init__(
        self,
        model: str = "vidore/colpali-v1.3-hf",
        *,
        processor: Any | None = None,
        model_backend: Any | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self._processor = processor
        self._model_backend = model_backend
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._torch: Any = None

    async def _embed_images(self, images: list[ImageInput]) -> Sequence[Any]:
        prepared = [self._prepare_image(image) for image in images]
        return await self._encode(prepared, modality="image")

    async def _embed_texts(self, texts: list[str]) -> Sequence[Any]:
        return await self._encode(texts, modality="text")

    async def _encode(self, values: list[Any], *, modality: str) -> list[np.ndarray]:
        import asyncio

        processor, model = self._get_backend()
        if modality == "image":
            process = getattr(processor, "process_images", None)
            if callable(process):
                batch = process(values)
            else:
                batch = processor(images=values, return_tensors="pt", padding=True)
        else:
            process = getattr(processor, "process_queries", None)
            if callable(process):
                batch = process(values)
            else:
                batch = processor(text=values, return_tensors="pt", padding=True)

        batch = self._move_batch(batch, model)

        def _run() -> Any:
            no_grad = getattr(self._torch, "no_grad", None)
            if callable(no_grad):
                with no_grad():
                    return model(**batch) if hasattr(batch, "keys") else model(batch)
            return model(**batch) if hasattr(batch, "keys") else model(batch)

        output = await asyncio.to_thread(_run)
        raw = self._extract_embeddings(output)
        return self._split_batch(raw, expected=len(values))

    def _get_backend(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model_backend is not None:
            return self._processor, self._model_backend

        try:
            import torch
        except ImportError:
            raise ImportError(
                "ColPaliEmbeddings requires torch. Install torch separately, "
                "then install synapsekit[transformers]."
            ) from None
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "ColPaliEmbeddings requires transformers. "
                "Install it with: pip install synapsekit[transformers]."
            ) from None

        is_colqwen = "colqwen" in self.model.lower()
        if is_colqwen:
            processor_type = getattr(transformers, "ColQwen2Processor", None)
            model_type = getattr(transformers, "ColQwen2ForRetrieval", None)
        else:
            processor_type = getattr(transformers, "ColPaliProcessor", None)
            model_type = getattr(transformers, "ColPaliForRetrieval", None)

        if processor_type is None or model_type is None:
            processor_type = getattr(transformers, "AutoProcessor", None)
            model_type = getattr(transformers, "AutoModel", None)
        if processor_type is None or model_type is None:
            raise ImportError(
                "The installed transformers package does not provide a ColPali/ColQwen model class. "
                "Upgrade it with: pip install -U transformers"
            )

        kwargs = {"trust_remote_code": self._trust_remote_code}
        self._processor = processor_type.from_pretrained(self.model, **kwargs)
        self._model_backend = model_type.from_pretrained(self.model, **kwargs)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        to_device = getattr(self._model_backend, "to", None)
        if callable(to_device):
            to_device(device)
        eval_model = getattr(self._model_backend, "eval", None)
        if callable(eval_model):
            eval_model()
        self._torch = torch
        return self._processor, self._model_backend

    def _move_batch(self, batch: Any, model: Any) -> Any:
        device = self._device or getattr(model, "device", None)
        if device is None:
            return batch
        if hasattr(batch, "items"):
            for key, value in batch.items():
                move = getattr(value, "to", None)
                if callable(move):
                    batch[key] = move(device)
            return batch
        move = getattr(batch, "to", None)
        return move(device) if callable(move) else batch

    @staticmethod
    def _extract_embeddings(output: Any) -> Any:
        for name in ("embeddings", "last_hidden_state", "pooler_output"):
            value = getattr(output, name, None)
            if value is not None:
                return value
        if isinstance(output, dict):
            for name in ("embeddings", "last_hidden_state", "pooler_output"):
                if output.get(name) is not None:
                    return output[name]
        raise ValueError("ColPali/ColQwen model output did not contain embeddings")

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
        return np.asarray(value, dtype=np.float32)

    def _split_batch(self, raw: Any, *, expected: int) -> list[np.ndarray]:
        if isinstance(raw, (list, tuple)) and raw and not isinstance(raw, np.ndarray):
            arrays = [self._as_numpy(item) for item in raw]
            if len(arrays) == expected and all(item.ndim == 2 for item in arrays):
                return arrays

        array = self._as_numpy(raw)
        if array.ndim == 2:
            if expected == 1:
                return [array]
            if array.shape[0] != expected:
                raise ValueError(
                    f"Model returned {array.shape[0]} embeddings for {expected} inputs"
                )
            return [array[index : index + 1] for index in range(expected)]
        if array.ndim == 3 and array.shape[0] == expected:
            return [array[index] for index in range(expected)]
        raise ValueError(
            f"Model returned an unsupported embedding shape {array.shape}; "
            f"expected {expected} x tokens x dimensions"
        )

    @staticmethod
    def _prepare_image(image: ImageInput) -> Any:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return ColPaliEmbeddings._open_image(path.read_bytes())
            return image
        if isinstance(image, bytes):
            return ColPaliEmbeddings._open_image(image)

        source_type = getattr(image, "source_type", None)
        if source_type == "url":
            return image.url
        if source_type in {"base64", "file"}:
            data = getattr(image, "data", "")
            if not data:
                raise ValueError(
                    f"Image with source_type={source_type!r} has empty data; "
                    "cannot prepare it for embedding"
                )
            return ColPaliEmbeddings._open_image(base64.b64decode(data))
        return image

    @staticmethod
    def _open_image(data: bytes) -> Any:
        try:
            from PIL import Image

            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return data


class ColQwenEmbeddings(ColPaliEmbeddings):
    """ColQwen2 variant of :class:`ColPaliEmbeddings`."""

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0-hf",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)


__all__ = ["ColPaliEmbeddings", "ColQwenEmbeddings"]
