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
from app.services.manga.panels import panels
from app.services.manga.regions import text_regions

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
Reader = Callable[[bytes, list[list[float]]], Awaitable[list[str]]]  # page, boxes → texts


def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def cache_dir(path: Path) -> Path:
    return path.with_name(path.name + ".pages")


def _typeset(image: bytes, regions: list[dict], translations: list[str | None]) -> bytes:
    """The page with each translated region's text replaced.

    shortcut: a white box over the original text box and the translation fitted in it; the
    real erasing (bubble fill, inpainting over art) and typesetting are epic items 6-7.
    """
    pix = pymupdf.Pixmap(image)
    page = pymupdf.open().new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, stream=image)
    for region, text in zip(regions, translations, strict=True):
        if text is None:
            continue
        box = pymupdf.Rect(region["box"]) + (-3, -3, 3, 3)
        page.draw_rect(box, color=None, fill=(1, 1, 1))
        page.insert_htmlbox(
            box,
            f'<p style="text-align:center;font-family:sans-serif;font-weight:bold;'
            f'margin:0">{html.escape(text)}</p>',
            scale_low=0,
        )
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
        """Read the pages not read yet; stops early when the run is cancelled."""
        if reader is None:
            raise ValueError("A manga needs an OCR reader")
        self.cache.mkdir(exist_ok=True)
        total = len(self.pages)
        for index in range(total):
            if cancelled.is_set():
                return
            if self._cached(index).exists():
                continue
            image = self.image(index)
            detected = await detect(image)
            order = panels(image, [d["box"] for d in detected], self.right_to_left)
            found = text_regions(detected, self.right_to_left, order)
            texts = await reader(image, [r["box"] for r in found])
            regions = [
                {"box": r["box"], "kind": r["kind"], "bubble": r["bubble"], "text": t}
                for r, t in zip(found, texts, strict=True)
                if t.strip()
            ]
            temporary = self._cached(index).with_suffix(".tmp")
            temporary.write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self._cached(index))  # whole or absent: a resume reads it again
            yield TranslationProgress(
                event_type="page_read",
                chapter_index=index,
                total_chapters=total,
                message=f"Page {index + 1}/{total} read: {len(regions)} text regions",
            )

    def save(self, path: Path, translations: Translations, reflow: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as out:
            for index, name in enumerate(self.pages):
                page_translations = translations.get(index)
                if page_translations and any(t is not None for t in page_translations):
                    image = _typeset(self.image(index), self.regions(index), page_translations)
                    out.writestr(str(Path(name).with_suffix(".png")), image)
                else:
                    out.writestr(name, self.image(index))
        return path

    def info(self) -> dict:
        return {"title": self.path.stem, "language": "japanese"}
