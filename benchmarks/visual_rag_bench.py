"""Deterministic visual-document QA retrieval benchmark.

This is a small DocVQA-style layout subset, not a network-fetched copy of
DocVQA. Each page has an OCR-only field and a visual token field; the latter
contains layout/color/chart signals intentionally absent from OCR so the
comparison measures the visual retrieval path rather than model variance.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synapsekit.embeddings.multimodal import BaseMultimodalEmbeddings
from synapsekit.loaders.visual import ImagePageRenderer, VisualPage
from synapsekit.retrieval.visual import VisualDocumentRetriever

_SEED = 42
_TOP_K = 1
_FEATURE_WORDS = (
    "green",
    "blue",
    "orange",
    "purple",
    "forecast",
    "warning",
    "baseline",
    "confidence",
)
_FEATURE_COLORS = (
    (35, 160, 80),
    (45, 105, 210),
    (235, 130, 35),
    (140, 80, 190),
    (190, 70, 70),
    (220, 45, 55),
    (55, 55, 55),
    (70, 150, 170),
)


@dataclass(frozen=True)
class VisualQASample:
    question: str
    expected_page: int
    visual_tokens: str
    extracted_text: str


@dataclass(frozen=True)
class VisualBenchmarkResult:
    samples: int
    visual_correct: int
    text_correct: int
    visual_accuracy: float
    text_accuracy: float
    uplift: float
    seed: int = _SEED

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


_DATASET_PATH = Path(__file__).with_name("fixtures") / "visual_doc_qa.json"


def _load_dataset() -> tuple[VisualQASample, ...]:
    records = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    return tuple(VisualQASample(**record) for record in records)


class _StableTokenEmbeddings(BaseMultimodalEmbeddings):
    dimensions = 64

    def __init__(self, *, visual_mode: bool = False) -> None:
        self._visual_mode = visual_mode

    @staticmethod
    def _feature_index(text: str) -> int | None:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        for index, word in enumerate(_FEATURE_WORDS):
            if word in tokens:
                return index
        return None

    @classmethod
    def _image_feature_index(cls, value: object) -> int | None:
        try:
            if isinstance(value, (bytes, bytearray, memoryview)):
                image = Image.open(io.BytesIO(bytes(value))).convert("RGB")
            elif isinstance(value, Image.Image):
                image = value.convert("RGB")
            else:
                return None
            pixel = image.getpixel((0, image.height - 1))
            if not isinstance(pixel, tuple) or len(pixel) < 3:
                return None
            return min(
                range(len(_FEATURE_COLORS)),
                key=lambda index: sum(
                    (int(pixel[channel]) - _FEATURE_COLORS[index][channel]) ** 2
                    for channel in range(3)
                ),
            )
        except (OSError, TypeError, ValueError):
            return None

    @classmethod
    def _tokens(cls, value: object, *, semantic: bool = False) -> np.ndarray:
        if semantic:
            feature_index = cls._image_feature_index(value)
            if feature_index is not None:
                tokens = [f"visual_feature_{feature_index}"]
            else:
                tokens = re.findall(r"[a-z0-9]+", str(value).lower())
                feature_index = cls._feature_index(str(value))
                if feature_index is not None:
                    tokens.append(f"visual_feature_{feature_index}")
        else:
            tokens = re.findall(r"[a-z0-9]+", str(value).lower())
        if not tokens:
            return np.zeros((1, cls.dimensions), dtype=np.float32)
        vectors = np.zeros((len(tokens), cls.dimensions), dtype=np.float32)
        for row, token in enumerate(tokens):
            if token.startswith("visual_feature_"):
                index = int(token.rsplit("_", 1)[-1])
            else:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = 8 + int.from_bytes(digest[:4], "big") % (cls.dimensions - 8)
            vectors[row, index] = 1.0
        return vectors

    async def _embed_images(self, images):
        return [self._tokens(image, semantic=self._visual_mode) for image in images]

    async def _embed_texts(self, texts):
        return [self._tokens(text, semantic=self._visual_mode) for text in texts]


def _write_visual_fixture(sample: VisualQASample, path: Path) -> None:
    feature_index = _StableTokenEmbeddings._feature_index(sample.visual_tokens)
    assert feature_index is not None
    image = Image.new("RGB", (256, 160), _FEATURE_COLORS[feature_index])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 255, 45), fill="white")
    if "dashed" in sample.visual_tokens:
        for x in range(20, 230, 24):
            draw.line((x, 85, x + 12, 85), fill="white", width=4)
    elif "dotted" in sample.visual_tokens:
        for x in range(20, 230, 16):
            draw.ellipse((x, 83, x + 5, 88), fill="white")
    elif "shaded" in sample.visual_tokens:
        draw.rectangle((30, 60, 225, 115), fill=(220, 240, 245))
    else:
        draw.rectangle((30, 60, 225, 115), outline="white", width=5)
    image.save(path, format="PNG")


async def _evaluate() -> VisualBenchmarkResult:
    samples = list(_load_dataset())
    random.Random(_SEED).shuffle(samples)
    backend = _StableTokenEmbeddings(visual_mode=True)
    visual_retriever = VisualDocumentRetriever(backend)
    text_retriever = VisualDocumentRetriever(_StableTokenEmbeddings())

    with tempfile.TemporaryDirectory(prefix="synapsekit-visual-bench-") as fixture_dir:
        visual_pages = []
        renderer = ImagePageRenderer()
        for sample in samples:
            image_path = Path(fixture_dir) / f"page-{sample.expected_page}.png"
            _write_visual_fixture(sample, image_path)
            rendered = renderer.render_sync(image_path)[0]
            metadata = dict(rendered.metadata)
            metadata.update(page=sample.expected_page, bbox="full-page")
            visual_pages.append(
                VisualPage(image=rendered.image, text=sample.extracted_text, metadata=metadata)
            )
        text_pages = [
            VisualPage(
                image=sample.extracted_text,
                text=sample.extracted_text,
                metadata={"page": sample.expected_page, "bbox": "full-page"},
            )
            for sample in samples
        ]
        await visual_retriever.add_pages(visual_pages)
        await text_retriever.add_pages(text_pages)

        visual_correct = 0
        text_correct = 0
        for sample in samples:
            visual_result = await visual_retriever.retrieve_with_scores(
                sample.question, top_k=_TOP_K
            )
            text_result = await text_retriever.retrieve_with_scores(sample.question, top_k=_TOP_K)
            visual_match = bool(
                visual_result and visual_result[0]["metadata"].get("page") == sample.expected_page
            )
            text_match = bool(
                text_result and text_result[0]["metadata"].get("page") == sample.expected_page
            )
            visual_correct += int(visual_match)
            text_correct += int(text_match)

        samples_count = len(samples)
        visual_accuracy = visual_correct / samples_count
        text_accuracy = text_correct / samples_count
        return VisualBenchmarkResult(
            samples=samples_count,
            visual_correct=visual_correct,
            text_correct=text_correct,
            visual_accuracy=visual_accuracy,
            text_accuracy=text_accuracy,
            uplift=visual_accuracy - text_accuracy,
        )


def run_benchmark() -> VisualBenchmarkResult:
    """Run the fixed visual-vs-text comparison without network or model downloads."""
    return asyncio.run(_evaluate())


def main() -> None:
    print(json.dumps(run_benchmark().as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
