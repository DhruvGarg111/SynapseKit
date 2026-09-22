from __future__ import annotations

import base64
import io
import sys
import types

import pytest

from synapsekit.loaders.pptx import PowerPointLoader
from synapsekit.loaders.visual import (
    DefaultPageRenderer,
    ImagePageRenderer,
    PDFPageRenderer,
    PowerPointPageRenderer,
    VisualPage,
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_visual_page_keeps_image_text_and_source_metadata():
    page = VisualPage(
        image=b"page-bytes",
        text="The total is shown in the green cell.",
        metadata={"page": 4, "bbox": [10, 20, 300, 400]},
    )

    assert page.image == b"page-bytes"
    assert page.text.startswith("The total")
    assert page.metadata == {"page": 4, "bbox": [10, 20, 300, 400]}


def test_visual_page_types_are_publicly_exported():
    import synapsekit
    from synapsekit import PDFPageRenderer as TopLevelPDFPageRenderer
    from synapsekit.loaders import PDFPageRenderer as LoaderPDFPageRenderer
    from synapsekit.loaders import VisualPage as LoaderVisualPage

    assert TopLevelPDFPageRenderer is PDFPageRenderer
    assert LoaderPDFPageRenderer is PDFPageRenderer
    assert LoaderVisualPage is VisualPage
    assert synapsekit.VisualPage is VisualPage


@pytest.mark.asyncio
async def test_image_page_renderer_returns_a_citable_full_page(tmp_path):
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(_TINY_PNG)

    pages = await ImagePageRenderer().render(image_path)

    assert len(pages) == 1
    assert pages[0].image == _TINY_PNG
    assert pages[0].text == ""
    assert pages[0].metadata["source_type"] == "image"
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert pages[0].metadata["locator"] == "scan.png page 1"


@pytest.mark.asyncio
async def test_image_page_renderer_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        await ImagePageRenderer().render(tmp_path / "missing.png")


@pytest.mark.asyncio
async def test_pdf_page_renderer_rasterizes_pages_and_closes_document(tmp_path, monkeypatch):
    class FakePixmap:
        def tobytes(self, format_name):
            assert format_name == "png"
            return b"rendered-pdf-page"

    class FakePDFPage:
        rect = types.SimpleNamespace(width=612, height=792)

        def get_pixmap(self, **kwargs):
            assert kwargs["alpha"] is False
            return FakePixmap()

        def get_text(self, mode):
            assert mode == "text"
            return "Scanned page text"

    class FakePDF:
        def __init__(self):
            self.closed = False
            self.pages = [FakePDFPage(), FakePDFPage()]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            self.closed = True

    document = FakePDF()
    fake_pymupdf = types.SimpleNamespace(
        Matrix=lambda scale_x, scale_y: (scale_x, scale_y),
        open=lambda path: document,
    )
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)

    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"pdf")
    pages = await PDFPageRenderer(scale=1.5, max_pages=1).render(pdf_path)

    assert len(pages) == 1
    assert pages[0].image == b"rendered-pdf-page"
    assert pages[0].text == "Scanned page text"
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["bbox"] == [0.0, 0.0, 612.0, 792.0]
    assert document.closed is True


def test_pdf_page_renderer_has_actionable_optional_dependency_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"pdf")

    with pytest.raises(ImportError, match=r"synapsekit\[visual\]"):
        PDFPageRenderer().render_sync(pdf_path)


@pytest.mark.asyncio
async def test_default_renderer_dispatches_svg_images(tmp_path):
    svg_path = tmp_path / "diagram.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' />", encoding="utf-8")

    pages = await DefaultPageRenderer().render(svg_path)

    assert len(pages) == 1
    assert pages[0].metadata["media_type"] == "image/svg+xml"


def test_powerpoint_page_renderer_creates_visual_page(monkeypatch, tmp_path):
    class FakeShape:
        left = 100
        top = 100
        width = 500
        height = 200
        image = None
        text = "Quarterly revenue"
        has_table = False

    class FakePresentation:
        slide_width = 1000
        slide_height = 500
        slides = [types.SimpleNamespace(shapes=[FakeShape()])]

    monkeypatch.setitem(
        sys.modules,
        "pptx",
        types.SimpleNamespace(Presentation=lambda path: FakePresentation()),
    )
    slide_path = tmp_path / "deck.pptx"
    slide_path.write_bytes(b"pptx")

    pages = PowerPointPageRenderer(width=1000).render_sync(slide_path)

    assert len(pages) == 1
    assert pages[0].image.startswith(b"\x89PNG")
    assert pages[0].text == "Quarterly revenue"
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["slide"] == 0
    assert pages[0].metadata["slide_number"] == 1


def test_powerpoint_page_renderer_draws_chart_only_slides(monkeypatch, tmp_path):
    class FakeSeries:
        values = [2, 4, 1]

    class FakePlot:
        series = [FakeSeries()]

    class FakeChart:
        plots = [FakePlot()]

    class FakeShape:
        left = 100
        top = 100
        width = 700
        height = 300
        image = None
        text = ""
        has_table = False
        has_chart = True
        chart = FakeChart()

    class FakePresentation:
        slide_width = 1000
        slide_height = 500
        slides = [types.SimpleNamespace(shapes=[FakeShape()])]

    monkeypatch.setitem(
        sys.modules,
        "pptx",
        types.SimpleNamespace(Presentation=lambda path: FakePresentation()),
    )
    slide_path = tmp_path / "chart.pptx"
    slide_path.write_bytes(b"pptx")

    pages = PowerPointPageRenderer(width=1000, prefer_native=False).render_sync(slide_path)

    from PIL import Image

    image = Image.open(io.BytesIO(pages[0].image)).convert("RGB")
    extrema = image.getextrema()
    assert any(channel_min < 255 for channel_min, _channel_max in extrema)


@pytest.mark.asyncio
async def test_powerpoint_loader_exposes_async_text_fallback(monkeypatch, tmp_path):
    slide_path = tmp_path / "deck.pptx"
    slide_path.write_bytes(b"pptx")
    loader = PowerPointLoader(str(slide_path))
    expected = ["fallback"]
    monkeypatch.setattr(loader, "load", lambda: expected)

    assert await loader.aload() == expected


def test_powerpoint_loader_keeps_legacy_slide_index_and_page_number(monkeypatch, tmp_path):
    class FakeParagraph:
        text = "Slide text"

    class FakeTextFrame:
        paragraphs = [FakeParagraph()]

    class FakeShape:
        has_text_frame = True
        text_frame = FakeTextFrame()

    class FakePresentation:
        slides = [types.SimpleNamespace(shapes=[FakeShape()])]

    monkeypatch.setitem(
        sys.modules,
        "pptx",
        types.SimpleNamespace(Presentation=lambda path: FakePresentation()),
    )
    slide_path = tmp_path / "deck.pptx"
    slide_path.write_bytes(b"pptx")

    document = PowerPointLoader(str(slide_path)).load()[0]

    assert document.metadata["page"] == 1
    assert document.metadata["slide"] == 0
    assert document.metadata["slide_number"] == 1


def test_powerpoint_loader_extracts_table_only_slides(monkeypatch, tmp_path):
    class FakeCell:
        def __init__(self, text):
            self.text = text

    class FakeRow:
        cells = [FakeCell("Region"), FakeCell("Revenue")]

    class FakeTable:
        rows = [FakeRow()]

    class FakeShape:
        has_table = True
        table = FakeTable()
        has_text_frame = False

    class FakePresentation:
        slides = [types.SimpleNamespace(shapes=[FakeShape()])]

    monkeypatch.setitem(
        sys.modules,
        "pptx",
        types.SimpleNamespace(Presentation=lambda path: FakePresentation()),
    )
    slide_path = tmp_path / "table.pptx"
    slide_path.write_bytes(b"pptx")

    document = PowerPointLoader(str(slide_path)).load()[0]

    assert document.text == "Region | Revenue"
