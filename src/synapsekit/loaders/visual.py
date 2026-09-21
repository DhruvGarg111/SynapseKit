"""Render document pages into image-backed :class:`VisualPage` objects.

The data contract is dependency-free. Concrete PDF and slide rendering
backends import their optional libraries only when ``render()`` is called.
"""

from __future__ import annotations

import asyncio
import io
import mimetypes
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class VisualPage:
    """One image-backed page with text fallback and source metadata.

    ``image`` is passed to the visual embedder and answer provider as-is. It
    may be a full-page render or a caller-supplied crop. ``bbox`` metadata
    identifies that image in the original page coordinate space; renderers
    use a full-page bbox and never apply an implicit second crop.
    """

    image: Any
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PageRenderer(Protocol):
    """Async protocol implemented by document page renderers."""

    async def render(self, path: str | Path) -> list[VisualPage]:
        """Render a document path into ordered visual pages."""
        ...


def _page_metadata(
    path: Path,
    *,
    source_type: str,
    loader: str,
    page: int,
    bbox: list[float],
    media_type: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "source": str(path),
        "file": str(path),
        "source_type": source_type,
        "chunk_type": "visual_page",
        "media_type": media_type,
        "loader": loader,
        "page": page,
        "bbox": bbox,
        "locator": f"{path.name} page {page}",
        **extra,
    }


def _image_size(data: bytes) -> tuple[float, float]:
    """Read image dimensions when Pillow is available; otherwise use a unit box."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        return float(width), float(height)
    except Exception:
        return 1.0, 1.0


class ImagePageRenderer:
    """Render one image file as one visual page."""

    async def render(self, path: str | Path) -> list[VisualPage]:
        return await asyncio.to_thread(self.render_sync, path)

    def render_sync(self, path: str | Path) -> list[VisualPage]:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        data = image_path.read_bytes()
        width, height = _image_size(data)
        media_type, _ = mimetypes.guess_type(str(image_path))
        return [
            VisualPage(
                image=data,
                metadata=_page_metadata(
                    image_path,
                    source_type="image",
                    loader=type(self).__name__,
                    page=1,
                    bbox=[0.0, 0.0, width, height],
                    media_type=media_type or "image/png",
                    image_media_type=media_type or "image/png",
                ),
            )
        ]


class PDFPageRenderer:
    """Rasterize PDF pages with optional PyMuPDF."""

    def __init__(self, *, scale: float = 2.0, max_pages: int | None = None) -> None:
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be greater than 0")
        self._scale = scale
        self._max_pages = max_pages

    async def render(self, path: str | Path) -> list[VisualPage]:
        return await asyncio.to_thread(self.render_sync, path)

    def render_sync(self, path: str | Path) -> list[VisualPage]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        fitz = self._load_pymupdf()
        document = fitz.open(str(pdf_path))
        pages: list[VisualPage] = []
        try:
            page_limit = len(document)
            if self._max_pages is not None:
                page_limit = min(page_limit, self._max_pages)
            for index in range(page_limit):
                pdf_page = document[index]
                rect = pdf_page.rect
                pixmap = pdf_page.get_pixmap(
                    matrix=fitz.Matrix(self._scale, self._scale),
                    alpha=False,
                )
                pages.append(
                    VisualPage(
                        image=pixmap.tobytes("png"),
                        text=str(pdf_page.get_text("text") or "").strip(),
                        metadata=_page_metadata(
                            pdf_path,
                            source_type="pdf",
                            loader=type(self).__name__,
                            page=index + 1,
                            bbox=[0.0, 0.0, float(rect.width), float(rect.height)],
                            media_type="application/pdf",
                            image_media_type="image/png",
                        ),
                    )
                )
        finally:
            document.close()
        return pages

    @staticmethod
    def _load_pymupdf() -> Any:
        try:
            import pymupdf

            return pymupdf
        except ImportError:
            try:
                import fitz

                return fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF is required for PDF visual rendering. "
                    "Install it with: pip install synapsekit[visual]"
                ) from None


class PowerPointPageRenderer:
    """Render PPTX slides to PNG canvases.

    LibreOffice is used when available for native-fidelity conversion. The
    dependency-free fallback uses python-pptx and Pillow and preserves text,
    pictures, fills, lines, tables, and chart data for environments without a
    slide office suite.
    """

    def __init__(
        self,
        *,
        width: int = 1600,
        max_pages: int | None = None,
        prefer_native: bool = True,
        native_timeout: float = 120.0,
    ) -> None:
        if width <= 0:
            raise ValueError("width must be greater than 0")
        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be greater than 0")
        if native_timeout <= 0:
            raise ValueError("native_timeout must be greater than 0")
        self._width = width
        self._max_pages = max_pages
        self._prefer_native = prefer_native
        self._native_timeout = native_timeout

    async def render(self, path: str | Path) -> list[VisualPage]:
        return await asyncio.to_thread(self.render_sync, path)

    def render_sync(self, path: str | Path) -> list[VisualPage]:
        slide_path = Path(path)
        if not slide_path.exists():
            raise FileNotFoundError(f"PowerPoint file not found: {slide_path}")
        if self._prefer_native:
            native_pages = self._render_with_libreoffice(slide_path)
            if native_pages is not None:
                return native_pages
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError(
                "python-pptx is required for PowerPoint visual rendering. "
                "Install it with: pip install synapsekit[visual]"
            ) from None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise ImportError(
                "Pillow is required for PowerPoint visual rendering. "
                "Install it with: pip install pillow"
            ) from None

        presentation = Presentation(str(slide_path))
        slide_width = float(presentation.slide_width)
        slide_height = float(presentation.slide_height)
        canvas_height = max(1, round(self._width * slide_height / slide_width))
        pages: list[VisualPage] = []
        page_limit = len(presentation.slides)
        if self._max_pages is not None:
            page_limit = min(page_limit, self._max_pages)

        for index, slide in enumerate(presentation.slides):
            if index >= page_limit:
                break
            canvas = Image.new("RGB", (self._width, canvas_height), self._background_color(slide))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            text_parts: list[str] = []
            for shape in slide.shapes:
                left = self._coordinate(getattr(shape, "left", 0), slide_width, self._width)
                top = self._coordinate(getattr(shape, "top", 0), slide_height, canvas_height)
                width = max(
                    1, self._coordinate(getattr(shape, "width", 0), slide_width, self._width)
                )
                height = max(
                    1, self._coordinate(getattr(shape, "height", 0), slide_height, canvas_height)
                )
                self._draw_shape_fill(shape, draw, left, top, width, height)
                self._draw_picture(shape, canvas, left, top, width, height, Image)
                self._draw_chart(shape, draw, left, top, width, height)
                self._draw_shape_line(shape, draw, left, top, width, height)

                text = str(getattr(shape, "text", "") or "").strip()
                if text:
                    text_parts.append(text)
                    wrapped = "\n".join(textwrap.wrap(text, width=max(12, width // 8)))
                    draw.multiline_text((left + 4, top + 4), wrapped, fill="black", font=font)

                if bool(getattr(shape, "has_table", False)):
                    table = getattr(shape, "table", None)
                    if table is not None:
                        rows: list[str] = []
                        for row in table.rows:
                            cells = [str(cell.text or "").strip() for cell in row.cells]
                            rows.append(" | ".join(cells))
                        table_text = "\n".join(row for row in rows if row.strip())
                        if table_text:
                            text_parts.append(table_text)
                            draw.multiline_text(
                                (left + 4, top + 4),
                                table_text,
                                fill="black",
                                font=font,
                            )

            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG")
            page_number = index + 1
            pages.append(
                VisualPage(
                    image=buffer.getvalue(),
                    text="\n".join(text_parts),
                    metadata=_page_metadata(
                        slide_path,
                        source_type="pptx",
                        loader=type(self).__name__,
                        page=page_number,
                        bbox=[0.0, 0.0, float(self._width), float(canvas_height)],
                        media_type=(
                            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        ),
                        slide=page_number - 1,
                        slide_number=page_number,
                        image_media_type="image/png",
                    ),
                )
            )
        return pages

    def _render_with_libreoffice(self, slide_path: Path) -> list[VisualPage] | None:
        """Use LibreOffice when available for faithful native slide rendering."""
        executable = shutil.which("soffice") or shutil.which("libreoffice")
        if executable is None:
            return None

        try:
            with tempfile.TemporaryDirectory(prefix="synapsekit-pptx-") as directory:
                output_dir = Path(directory)
                subprocess.run(
                    [
                        executable,
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(output_dir),
                        str(slide_path),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self._native_timeout,
                )
                pdf_path = output_dir / f"{slide_path.stem}.pdf"
                if not pdf_path.exists():
                    return None
                rendered = PDFPageRenderer(
                    scale=self._width / 960.0,
                    max_pages=self._max_pages,
                ).render_sync(pdf_path)
        except (ImportError, OSError, subprocess.SubprocessError, ValueError):
            return None

        pages: list[VisualPage] = []
        for page in rendered:
            metadata = dict(page.metadata)
            page_number = int(metadata.get("page", len(pages) + 1))
            metadata.update(
                {
                    "source": str(slide_path),
                    "file": str(slide_path),
                    "source_type": "pptx",
                    "chunk_type": "visual_page",
                    "loader": type(self).__name__,
                    "page": page_number,
                    "slide": page_number - 1,
                    "slide_number": page_number,
                    "image_media_type": "image/png",
                    "media_type": (
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    ),
                    "locator": f"{slide_path.name} page {page_number}",
                }
            )
            pages.append(VisualPage(image=page.image, text=page.text, metadata=metadata))
        return pages

    @staticmethod
    def _background_color(slide: Any) -> Any:
        try:
            fill = slide.background.fill
            if getattr(fill, "type", None) is not None:
                return PowerPointPageRenderer._rgb_color(getattr(fill, "fore_color", None), "white")
        except (AttributeError, ValueError, TypeError):
            pass
        return "white"

    @staticmethod
    def _rgb_color(color: Any, default: Any) -> Any:
        try:
            value = str(color.rgb)
        except (AttributeError, TypeError, ValueError):
            return default
        return f"#{value}" if len(value) == 6 else default

    @staticmethod
    def _draw_shape_fill(
        shape: Any, draw: Any, left: int, top: int, width: int, height: int
    ) -> None:
        try:
            fill = shape.fill
            if getattr(fill, "type", None) is None:
                return
            color = PowerPointPageRenderer._rgb_color(getattr(fill, "fore_color", None), "white")
            shape_kind = str(getattr(shape, "auto_shape_type", "")).lower()
            box = (left, top, left + width, top + height)
            if "ellipse" in shape_kind or "oval" in shape_kind:
                draw.ellipse(box, fill=color)
            elif "round" in shape_kind:
                draw.rounded_rectangle(box, radius=max(1, min(width, height) // 8), fill=color)
            else:
                draw.rectangle(box, fill=color)
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _draw_shape_line(
        shape: Any, draw: Any, left: int, top: int, width: int, height: int
    ) -> None:
        try:
            line = shape.line
            fill = getattr(line, "fill", None)
            if fill is None or getattr(fill, "type", None) is None:
                return
            color = PowerPointPageRenderer._rgb_color(getattr(fill, "fore_color", None), "black")
            shape_kind = str(getattr(shape, "shape_type", "")).lower()
            if "line" in shape_kind or hasattr(shape, "begin_x"):
                draw.line((left, top, left + width, top + height), fill=color, width=2)
            else:
                draw.rectangle(
                    (left, top, left + width, top + height),
                    outline=color,
                    width=2,
                )
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _draw_chart(shape: Any, draw: Any, left: int, top: int, width: int, height: int) -> None:
        if not bool(getattr(shape, "has_chart", False)):
            return
        box = (left, top, left + width, top + height)
        draw.rectangle(box, fill="#f8fafc", outline="#475569", width=2)
        try:
            values: list[float] = []
            for plot in shape.chart.plots:
                for series in plot.series:
                    values.extend(float(value) for value in series.values if value is not None)
        except (AttributeError, TypeError, ValueError):
            values = []
        if not values:
            draw.text((left + 8, top + 8), "chart", fill="#334155")
            return

        padding = max(8, min(width, height) // 10)
        x0, y0 = left + padding, top + height - padding
        x1, y1 = left + width - padding, top + padding
        draw.line((x0, y0, x1, y0), fill="#334155", width=2)
        draw.line((x0, y0, x0, y1), fill="#334155", width=2)
        max_value = max(abs(value) for value in values) or 1.0
        bar_width = max(1, (x1 - x0) // max(1, len(values) * 2))
        for index, value in enumerate(values):
            x_left = x0 + index * (bar_width * 2) + 1
            y_value = y0 - round((value / max_value) * max(1, y0 - y1))
            draw.rectangle(
                (x_left, min(y0, y_value), x_left + bar_width, max(y0, y_value)),
                fill="#2563eb",
            )

    @staticmethod
    def _coordinate(value: Any, source: float, target: int) -> int:
        try:
            return max(0, round(float(value) / source * target))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0

    @staticmethod
    def _draw_picture(
        shape: Any, canvas: Any, left: int, top: int, width: int, height: int, image_module: Any
    ) -> None:
        image = getattr(shape, "image", None)
        if image is None:
            return
        try:
            picture = image_module.open(io.BytesIO(image.blob)).convert("RGB")
            picture.thumbnail((width, height))
            canvas.paste(picture, (left, top))
        except Exception:
            # Keep a failed picture visible in the page image instead of
            # silently turning an image-only region into a blank canvas.
            try:
                from PIL import ImageDraw

                marker = ImageDraw.Draw(canvas)
                marker.rectangle(
                    (left, top, left + width, top + height),
                    outline="#dc2626",
                    width=2,
                )
                marker.line((left, top, left + width, top + height), fill="#dc2626", width=2)
                marker.line((left + width, top, left, top + height), fill="#dc2626", width=2)
            except (AttributeError, TypeError, ValueError):
                return


class DefaultPageRenderer:
    """Dispatch common local document paths to their visual renderer."""

    def __init__(self, *, pdf_scale: float = 2.0, max_pages: int | None = None) -> None:
        self._pdf = PDFPageRenderer(scale=pdf_scale, max_pages=max_pages)
        self._pptx = PowerPointPageRenderer(max_pages=max_pages)
        self._image = ImagePageRenderer()

    async def render(self, path: str | Path) -> list[VisualPage]:
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            return await self._pdf.render(path)
        if suffix in {".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"}:
            return await self._pptx.render(path)
        if suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
            ".heic",
            ".heif",
            ".svg",
        }:
            return await self._image.render(path)
        raise ValueError(f"Unsupported visual document format: {suffix or '<none>'}")


VisualPageRenderer = DefaultPageRenderer

__all__ = [
    "DefaultPageRenderer",
    "ImagePageRenderer",
    "PDFPageRenderer",
    "PageRenderer",
    "PowerPointPageRenderer",
    "VisualPage",
    "VisualPageRenderer",
]
