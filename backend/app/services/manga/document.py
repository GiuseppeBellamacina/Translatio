"""Manga chapters (a CBZ/ZIP of page images) as documents: unit = page, block = text region.

The pages are read once, before the translation (``prepare``: detection, panels, regions in
reading order, OCR), and cached as ``<file>.pages/page-N.json`` next to the archive, so the
blocks stay the same across runs and resumes.
"""

import asyncio
import html
import json
import re
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pymupdf

from app.agents.pipeline import TranslationProgress
from app.services.chunker import SmartChunker, TextBlock
from app.services.documents import Document, Translations, Unit
from app.services.manga.detect import detect
from app.services.manga.erase import erase
from app.services.manga.ocr import is_sfx
from app.services.manga.panels import panels
from app.services.manga.regions import text_regions
from app.services.manga.typeset import fits, letter

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
Reader = Callable[[bytes, list[list[float]]], Awaitable[list[str]]]  # page, boxes → texts


def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _write(path: Path, data: bytes) -> None:
    """Whole or absent: an interrupted write leaves the page to do again."""
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def cache_dir(path: Path) -> Path:
    return path.with_name(path.name + ".pages")


def _typeset(
    image: bytes, regions: list[dict], translations: list[str | None], erased: bool
) -> bytes:
    """The page with each translated region lettered (typeset.py); a page whose lettering
    could not be erased gets a white box under each text."""
    pix = pymupdf.Pixmap(image)
    doc = pymupdf.open()
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, stream=image)
    for region, text in zip(regions, translations, strict=True):
        if text is None:
            continue
        if not erased:
            page.draw_rect(pymupdf.Rect(region["box"]) + (-3, -3, 3, 3), color=None, fill=(1, 1, 1))
        letter(page, region, text)
    return page.get_pixmap(dpi=72).tobytes("png")


class MangaDocument(Document):
    format = "cbz"
    suffix = ".cbz"
    media_type = "application/vnd.comicbook+zip"
    right_to_left = True  # manga keep the Japanese layout in every language

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            names = [
                n
                for n in archive.namelist()
                if Path(n).suffix.lower() in IMAGES and not n.startswith("__MACOSX/")
            ]
        if not names:
            raise ValueError("The archive has no page images")
        self.pages = sorted(names, key=_natural)
        self.cache = cache_dir(self.path)

    def image(self, index: int) -> bytes:
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(self.pages[index])

    def _cached(self, index: int) -> Path:
        return self.cache / f"page-{index + 1}.json"

    def _clean(self, index: int) -> Path:
        """The page with its lettering erased."""
        return self.cache / f"page-{index + 1}.png"

    def unread(self) -> int:
        """Pages not read yet: their text (and its cost) is known only after the OCR."""
        return sum(not self._cached(i).exists() for i in range(len(self.pages)))

    def regions(self, index: int) -> list[dict]:
        """The page's text regions in reading order (empty until the page is read)."""
        cached = self._cached(index)
        return json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else []

    @property
    def units(self) -> list[Unit]:
        return [
            Unit(item_id=f"page-{i + 1}", title=f"Page {i + 1}") for i in range(len(self.pages))
        ]

    def blocks(self, index: int, chunker: SmartChunker) -> list[TextBlock]:
        return [chunker.make_block("p", r["text"]) for r in self.regions(index)]

    def text(self, index: int) -> str:
        return "\n".join(r["text"] for r in self.regions(index))

    def render(self, index: int, translations: list[str | None] | None) -> str | None:
        regions = self.regions(index)
        translations = translations or [None] * len(regions)
        paragraphs = [
            html.escape(t if t is not None else r["text"]) for r, t in zip(regions, translations)
        ]
        return "<html><body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"

    async def prepare(
        self, reader: Reader | None, cancelled: asyncio.Event
    ) -> AsyncIterator[TranslationProgress]:
        """Read the pages not read yet, then erase their lettering (a page read before is only
        erased); stops early when the run is cancelled."""
        if reader is None:
            raise ValueError("A manga needs an OCR reader")
        self.cache.mkdir(exist_ok=True)
        total = len(self.pages)
        for index in range(total):
            if cancelled.is_set():
                return
            if self._clean(index).exists():
                continue
            image = self.image(index)
            if not self._cached(index).exists():
                detected = await detect(image)
                order = panels(image, [d["box"] for d in detected], self.right_to_left)
                found = text_regions(detected, self.right_to_left, order)
                texts = await reader(image, [r["box"] for r in found])
                regions = [
                    {"box": r["box"], "kind": r["kind"], "bubble": r["bubble"], "text": t}
                    for r, t in zip(found, texts, strict=True)
                    if t.strip() and not is_sfx(t)  # sound effects stay as drawn
                ]
                _write(self._cached(index), json.dumps(regions, ensure_ascii=False).encode())
            regions = self.regions(index)
            clean = await erase(image, [r["box"] for r in regions]) if regions else image
            _write(self._clean(index), clean)
            yield TranslationProgress(
                event_type="page_read",
                chapter_index=index,
                total_chapters=total,
                message=f"Page {index + 1}/{total} read: {len(regions)} text regions",
            )

    def fit_doubts(self, index: int, translations: list[str | None] | None) -> list[dict]:
        regions = self.regions(index)
        return [
            {
                "block": block,
                "quote": region["text"][:80],
                "reason": "Too long for its speech bubble, even at the smallest readable size: "
                "shorten it, keeping the meaning and the tone.",
            }
            for block, (region, text) in enumerate(zip(regions, translations or [], strict=False))
            if text and not fits(region, text)
        ]

    def page(self, index: int, translations: list[str | None] | None) -> bytes | None:
        """The page lettered on its erased image (PNG), or None when nothing is translated."""
        if not translations or all(t is None for t in translations):
            return None
        clean = self._clean(index)
        return _typeset(
            clean.read_bytes() if clean.exists() else self.image(index),
            self.regions(index),
            translations,
            erased=clean.exists(),
        )

    def save(self, path: Path, translations: Translations, reflow: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as out:
            for index, name in enumerate(self.pages):
                lettered = self.page(index, translations.get(index))
                if lettered:
                    out.writestr(str(Path(name).with_suffix(".png")), lettered)
                else:
                    out.writestr(name, self.image(index))
        return path

    def info(self) -> dict:
        return {"title": self.path.stem, "language": "japanese"}


class MangaPdf(MangaDocument):
    """A PDF of page images (a scan without a text layer) read like a manga, written back as
    a PDF with the same page sizes."""

    format = "pdf"
    suffix = ".pdf"
    media_type = "application/pdf"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with pymupdf.open(self.path) as pdf:
            self.pages = [f"page-{i + 1}.png" for i in range(len(pdf))]
        self.cache = cache_dir(self.path)

    def image(self, index: int) -> bytes:
        """The page at the scan's own resolution (its largest image), as PNG."""
        with pymupdf.open(self.path) as pdf:
            page = pdf[index]
            width = max((info[2] for info in page.get_images()), default=0)
            zoom = width / page.rect.width if width else 2
            return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")

    def save(self, path: Path, translations: Translations, reflow: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = pymupdf.open()
        with pymupdf.open(self.path) as pdf:
            for index, source in enumerate(pdf):
                lettered = self.page(index, translations.get(index))
                if lettered:
                    page = out.new_page(width=source.rect.width, height=source.rect.height)
                    page.insert_image(page.rect, stream=lettered)
                else:
                    out.insert_pdf(pdf, from_page=index, to_page=index)
        out.save(path, garbage=4, deflate=True)
        return path
