"""Manga chapters as documents: pages read once (cached), translated through the pipeline."""

import asyncio
import zipfile

import pymupdf

from app.agents.pipeline import TranslationPipeline
from app.models.schemas import TranslationResult
from app.services.documents import open_document
from app.services.manga import document
from scripts.make_sample_manga import sample_page
from tests.conftest import ScriptedLLM, translate
from tests.test_manga_regions import SAMPLE

TEXTS = ["IT'S NOT HERE.", "WHERE DID YOU PUT THE LETTER?", "Meanwhile...", "BAM!"]


def _chapter(tmp_path, monkeypatch, pages: int = 2):
    """A CBZ of sample pages, with the detector and the OCR faked; returns (path, reads)."""
    png, _ = sample_page()
    path = tmp_path / "chapter.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        for n in range(pages, 0, -1):  # stored out of order: pages sort naturally
            archive.writestr(f"p{n}.png", png)
        archive.writestr("notes.txt", "not a page")

    async def detected(image: bytes) -> list[dict]:
        return SAMPLE

    monkeypatch.setattr(document, "detect", detected)
    reads = []

    async def reader(image: bytes, boxes: list) -> list[str]:
        reads.append(len(boxes))
        return TEXTS[: len(boxes) - 1] + [""]  # the last region reads empty: no block

    return path, reads, reader


async def test_pages_are_read_once(tmp_path, monkeypatch):
    path, reads, reader = _chapter(tmp_path, monkeypatch, pages=10)
    doc = open_document(path)
    assert doc.pages[:3] == ["p1.png", "p2.png", "p3.png"] and len(doc.units) == 10
    assert doc.text(0) == ""  # not read yet

    events = [e async for e in doc.prepare(reader, asyncio.Event())]
    assert [e.event_type for e in events] == ["page_read"] * 10 and reads == [4] * 10
    regions = doc.regions(0)
    assert [r["text"] for r in regions] == TEXTS[:3]  # reading order, empty text dropped
    assert regions[0]["bubble"] is not None and regions[2]["bubble"] is None

    again = open_document(path)  # a resume: the cache answers
    assert [e async for e in again.prepare(reader, asyncio.Event())] == []
    assert reads == [4] * 10 and again.text(0) == "\n".join(TEXTS[:3])


async def test_a_chapter_is_translated_through_the_pipeline(tmp_path, monkeypatch):
    path, _, reader = _chapter(tmp_path, monkeypatch)
    llm = ScriptedLLM()
    events = await translate(
        path, TranslationPipeline(llm=llm, enable_review=False), tmp_path / "out.cbz", reader=reader
    )
    done = events[-1]
    assert done.event_type == "job_complete" and done.data["failed_units"] == []
    assert len(llm.calls_for(TranslationResult)) == 1  # the two short pages in one call

    with zipfile.ZipFile(done.data["output_path"]) as out:
        assert out.namelist() == ["p1.png", "p2.png"]
        page = pymupdf.open(stream=out.read("p1.png"), filetype="png")
    assert page[0].rect == pymupdf.Rect(0, 0, 800, 1200)  # same page size, text replaced
