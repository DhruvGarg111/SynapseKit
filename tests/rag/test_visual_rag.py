from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from synapsekit import RAG
from synapsekit import MultimodalRAG as TopLevelMultimodalRAG
from synapsekit.embeddings.multimodal import BaseMultimodalEmbeddings
from synapsekit.loaders.base import Document
from synapsekit.loaders.visual import VisualPage
from synapsekit.rag.multimodal import MultimodalRAG


class FakeVisualEmbeddings(BaseMultimodalEmbeddings):
    dimensions = 2

    @staticmethod
    def _vector(value: object) -> np.ndarray:
        text = str(value).lower()
        if "answer" in text or "page-a" in text:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0]], dtype=np.float32)

    async def _embed_images(self, images):
        return [self._vector(image) for image in images]

    async def _embed_texts(self, texts):
        return [self._vector(text) for text in texts]


class FakeRenderer:
    async def render(self, path):
        return [
            VisualPage(
                image=b"page-a",
                text="The answer is in the highlighted cell.",
                metadata={"page": 4, "bbox": [12.0, 24.0, 120.0, 240.0]},
            )
        ]


class BrokenRenderer:
    async def render(self, path):
        raise RuntimeError("renderer failed")


class RecordingPacker:
    def __init__(self):
        self.calls = []

    def pack(self, chunks, query=None):
        self.calls.append((chunks, query))
        return [chunks[-1]]


def _make_rag(**kwargs):
    rag = RAG(model="gpt-4o-mini", api_key="[REDACTED]", **kwargs)
    rag._pipeline.add_documents = AsyncMock()
    rag._pipeline.config.retriever.retrieve_with_scores = AsyncMock(return_value=[])
    return rag


@pytest.mark.asyncio
async def test_rag_visual_backend_indexes_visual_pages_and_text_fallback(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"pdf")
    rag = _make_rag(
        visual_embeddings=FakeVisualEmbeddings(),
        visual_renderer=FakeRenderer(),
    )

    with patch(
        "synapsekit.loaders.pdf.PDFLoader.aload",
        new=AsyncMock(return_value=[Document(text="OCR fallback", metadata={"page": 4})]),
    ):
        await rag.add_async(str(pdf_path))

    assert rag.visual_retriever is not None
    assert rag.visual_retriever.page_count == 1
    fallback_docs = rag._pipeline.add_documents.await_args.args[0]
    assert fallback_docs[0].text == "OCR fallback"


@pytest.mark.asyncio
async def test_rag_visual_ask_sends_page_images_and_appends_citations():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    await rag.visual_retriever.add_pages(
        [
            VisualPage(
                image=b"page-a",
                text="The answer is in the highlighted cell.",
                metadata={
                    "source": "report.pdf",
                    "page": 4,
                    "bbox": [12.0, 24.0, 120.0, 240.0],
                },
            )
        ]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "The answer is 42."

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    answer = await rag.ask("What is the answer?")

    content = captured["messages"][-1]["content"]
    assert isinstance(content, list)
    assert any(block.get("type") == "image_url" for block in content)
    prompt_text = next(block["text"] for block in content if block.get("type") == "text")
    assert "page: 4" in prompt_text
    assert "bbox: [12.0, 24.0, 120.0, 240.0]" in prompt_text
    assert "source: report.pdf" in prompt_text
    assert "page 4" in answer
    assert "bbox [12.0, 24.0, 120.0, 240.0]" in answer
    assert "source report.pdf" in answer


@pytest.mark.asyncio
async def test_rag_without_visual_backend_keeps_text_pipeline_path():
    rag = _make_rag()
    rag._pipeline.ask = AsyncMock(return_value="text answer")

    assert await rag.ask("question") == "text answer"
    rag._pipeline.ask.assert_awaited_once_with("question")


def test_multimodal_rag_is_a_named_rag_facade():
    assert issubclass(MultimodalRAG, RAG)
    assert MultimodalRAG.__doc__
    assert TopLevelMultimodalRAG is MultimodalRAG


@pytest.mark.asyncio
async def test_rag_visual_stream_appends_citations_and_uses_anthropic_blocks():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    rag._pipeline.config.llm.config.provider = "anthropic"
    await rag.visual_retriever.add_pages(
        [VisualPage(image=b"page-a", text="answer", metadata={"page": 2, "bbox": [1, 2, 3, 4]})]
    )

    captured = {}

    async def stream_with_messages(messages, **kwargs):
        captured["messages"] = messages
        yield "partial"

    rag._pipeline.config.llm.stream_with_messages = stream_with_messages
    output = "".join([token async for token in rag.stream("answer")])

    content = captured["messages"][-1]["content"]
    assert any(block.get("type") == "image" for block in content)
    assert "partial" in output
    assert "page 2, bbox [1, 2, 3, 4]" in output


@pytest.mark.asyncio
async def test_rag_visual_ingestion_keeps_text_fallback_when_renderer_fails(tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"pdf")
    rag = _make_rag(
        visual_embeddings=FakeVisualEmbeddings(),
        visual_renderer=BrokenRenderer(),
    )

    with patch(
        "synapsekit.loaders.pdf.PDFLoader.aload",
        new=AsyncMock(return_value=[Document(text="OCR fallback", metadata={"page": 1})]),
    ):
        await rag.add_async(str(pdf_path))

    assert rag.visual_retriever is not None
    assert rag.visual_retriever.page_count == 0
    rag._pipeline.add_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_visual_context_packer_preserves_selected_page_image():
    packer = RecordingPacker()
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings(), context_packer=packer)
    await rag.visual_retriever.add_pages(
        [
            VisualPage(
                image=b"\xff\xd8\xffjpeg",
                text="JPEG evidence",
                metadata={"page": 3, "bbox": "full-page"},
            )
        ]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "answer"

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    await rag.ask("Which page?")

    assert packer.calls and packer.calls[0][1] == "Which page?"
    content = captured["messages"][-1]["content"]
    image_block = next(block for block in content if block.get("type") == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    prompt_text = content[0]["text"]
    assert "JPEG evidence" in prompt_text


@pytest.mark.asyncio
async def test_rag_visual_text_only_provider_does_not_stringify_image_blocks():
    rag = _make_rag(visual_embeddings=FakeVisualEmbeddings())
    rag._pipeline.config.llm.supports_multimodal = False
    await rag.visual_retriever.add_pages(
        [VisualPage(image=b"page-a", text="OCR evidence", metadata={"page": 1})]
    )

    captured = {}

    async def generate_with_messages(messages, **kwargs):
        captured["messages"] = messages
        return "text answer"

    rag._pipeline.config.llm.generate_with_messages = generate_with_messages
    await rag.ask("What is shown?")

    content = captured["messages"][-1]["content"]
    assert isinstance(content, str)
    assert "image_url" not in content
    assert "does not accept image messages" in content


def test_rag_rejects_visual_backend_and_retriever_together():
    visual_retriever = MagicMock()
    with pytest.raises(ValueError, match="either visual_retriever"):
        RAG(
            model="gpt-4o-mini",
            api_key="[REDACTED]",
            visual_embeddings=FakeVisualEmbeddings(),
            visual_retriever=visual_retriever,
        )


def test_rag_rejects_visual_model_and_backend_together():
    with pytest.raises(ValueError, match="either visual_embeddings or visual_model"):
        RAG(
            model="gpt-4o-mini",
            api_key="[REDACTED]",
            visual_embeddings=FakeVisualEmbeddings(),
            visual_model="vidore/colpali-v1.3-hf",
        )
