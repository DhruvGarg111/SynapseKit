from __future__ import annotations

import numpy as np
import pytest

from synapsekit.embeddings.multimodal import BaseMultimodalEmbeddings
from synapsekit.loaders.visual import VisualPage
from synapsekit.retrieval.visual import VisualDocumentRetriever, maxsim


class TokenEmbeddings(BaseMultimodalEmbeddings):
    dimensions = 3

    @staticmethod
    def _matrix(value: object) -> np.ndarray:
        token = str(value).lower()
        if "green" in token or "revenue" in token:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if "blue" in token or "cost" in token:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    async def _embed_images(self, images):
        return [self._matrix(image) for image in images]

    async def _embed_texts(self, texts):
        return [self._matrix(text) for text in texts]


@pytest.mark.parametrize(
    "query,page,expected",
    [
        (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            1.5,
        ),
        (
            np.empty((0, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            0.0,
        ),
    ],
)
def test_maxsim_scores_best_page_token_per_query_token(query, page, expected):
    assert maxsim(query, page) == pytest.approx(expected)


def test_maxsim_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions"):
        maxsim(np.ones((1, 2)), np.ones((1, 3)))


def test_visual_retrieval_types_are_publicly_exported():
    import synapsekit
    from synapsekit.retrieval import VisualDocumentRetriever as RetrievalExport
    from synapsekit.retrieval import maxsim as maxsim_export

    assert RetrievalExport is VisualDocumentRetriever
    assert maxsim_export is maxsim
    assert synapsekit.VisualDocumentRetriever is VisualDocumentRetriever
    assert synapsekit.maxsim is maxsim


@pytest.mark.asyncio
async def test_visual_retriever_returns_stable_rich_results_and_text_results():
    retriever = VisualDocumentRetriever(TokenEmbeddings())
    pages = [
        VisualPage(
            image=b"green-image",
            text="Revenue is in the green cell.",
            metadata={"page": 2, "bbox": [10, 20, 30, 40]},
        ),
        VisualPage(image=b"blue-image", text="Blue cost cell.", metadata={"page": 3}),
    ]

    await retriever.add_pages(pages)
    rich = await retriever.retrieve_with_scores("green revenue", top_k=2)

    assert retriever.page_count == 2
    assert rich[0]["text"] == "Revenue is in the green cell."
    assert rich[0]["image"] == b"green-image"
    assert rich[0]["metadata"]["page"] == 2
    assert rich[0]["metadata"]["bbox"] == [10, 20, 30, 40]
    assert await retriever.retrieve("green revenue", top_k=1) == ["Revenue is in the green cell."]


@pytest.mark.asyncio
async def test_visual_retriever_assigns_defaults_for_direct_pages():
    retriever = VisualDocumentRetriever(TokenEmbeddings())

    await retriever.add_pages([VisualPage(image=b"page", text="content")])
    result = await retriever.retrieve_with_scores("content")

    assert result[0]["metadata"]["page"] == 1
    assert result[0]["metadata"]["bbox"] == "full-page"


@pytest.mark.asyncio
async def test_visual_retriever_normalizes_explicit_none_metadata_defaults():
    retriever = VisualDocumentRetriever(TokenEmbeddings())

    await retriever.add_pages(
        [VisualPage(image=b"page", text="content", metadata={"page": None, "bbox": None})]
    )
    result = await retriever.retrieve_with_scores("content")

    assert result[0]["metadata"]["page"] == 1
    assert result[0]["metadata"]["bbox"] == "full-page"


@pytest.mark.asyncio
async def test_visual_retriever_add_file_uses_async_renderer(tmp_path):
    class FakeRenderer:
        async def render(self, path):
            assert path == tmp_path / "report.pdf"
            return [VisualPage(image=b"rendered", text="page")]

    retriever = VisualDocumentRetriever(TokenEmbeddings(), renderer=FakeRenderer())
    path = tmp_path / "report.pdf"
    path.write_bytes(b"pdf")

    pages = await retriever.add_file(path)

    assert len(pages) == 1
    assert retriever.page_count == 1


@pytest.mark.asyncio
async def test_visual_retriever_rejects_backend_count_mismatch():
    class BadEmbeddings(TokenEmbeddings):
        async def _embed_images(self, images):
            return []

    retriever = VisualDocumentRetriever(BadEmbeddings())
    with pytest.raises(ValueError, match="one embedding per input"):
        await retriever.add_pages([VisualPage(image=b"one")])


@pytest.mark.asyncio
async def test_visual_retriever_keeps_insertion_order_for_ties():
    class TieEmbeddings(TokenEmbeddings):
        @staticmethod
        def _matrix(value):
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    retriever = VisualDocumentRetriever(TieEmbeddings())
    await retriever.add_pages(
        [VisualPage(image=b"first", text="first"), VisualPage(image=b"second", text="second")]
    )

    result = await retriever.retrieve_with_scores("query", top_k=2)

    assert [item["text"] for item in result] == ["first", "second"]
