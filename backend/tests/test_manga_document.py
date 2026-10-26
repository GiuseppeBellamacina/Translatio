"""Manga chapters as documents: pages read once (cached), translated through the pipeline."""

import asyncio
import base64
import json
import sqlite3
import zipfile

import httpx
import pymupdf

from app.agents.pipeline import TranslationPipeline
from app.core.config import settings
from app.models.schemas import DoubtResolution, TranslationResult
from app.services.documents import open_document
from app.services.manga import document
from app.services.manga.erase import erase
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

    async def erased(image: bytes, boxes: list) -> bytes:
        return image

    monkeypatch.setattr(document, "detect", detected)
    monkeypatch.setattr(document, "erase", erased)
    reads = []

    async def reader(image: bytes, boxes: list) -> list[str]:
        reads.append(len(boxes))
        return TEXTS[: len(boxes) - 1] + ["[BAM]"]  # the OCR marks the sound effect: no block

    return path, reads, reader


async def test_pages_are_read_once(tmp_path, monkeypatch):
    path, reads, reader = _chapter(tmp_path, monkeypatch, pages=10)
    doc = open_document(path)
    assert doc.pages[:3] == ["p1.png", "p2.png", "p3.png"] and len(doc.units) == 10
    assert doc.text(0) == ""  # not read yet

    events = [e async for e in doc.prepare(reader, asyncio.Event())]
    assert [e.event_type for e in events] == ["page_read"] * 10 and reads == [4] * 10
    regions = doc.regions(0)
    assert [r["text"] for r in regions] == TEXTS[:3]  # reading order, the sound effect left out
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


async def test_pages_read_before_are_only_erased(tmp_path, monkeypatch):
    path, reads, reader = _chapter(tmp_path, monkeypatch, pages=2)
    doc = open_document(path)
    [e async for e in doc.prepare(reader, asyncio.Event())]
    (doc.cache / "page-2.png").unlink()  # read by an older version: no clean page yet
    events = [e async for e in doc.prepare(reader, asyncio.Event())]
    assert [e.chapter_index for e in events] == [1] and reads == [4, 4]  # erased, not read
    assert (doc.cache / "page-2.png").exists()


async def test_erase_client(monkeypatch):
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/erase" and body["boxes"] == [[1, 2, 3, 4]]
        return httpx.Response(200, json={"image": base64.b64encode(b"clean").decode()})

    monkeypatch.setattr(settings, "manga_ml_url", "http://ml:8100")
    assert await erase(b"png", [[1, 2, 3, 4]], httpx.MockTransport(reply)) == b"clean"


async def test_a_pdf_of_page_images_is_read_like_a_manga(tmp_path, monkeypatch):
    _, _, reader = _chapter(tmp_path, monkeypatch)
    png, _ = sample_page()
    path = tmp_path / "scan.pdf"
    scan = pymupdf.open()
    for _ in range(2):  # 800×1200 px images on 400×600 pt pages
        scan.new_page(width=400, height=600).insert_image(pymupdf.Rect(0, 0, 400, 600), stream=png)
    scan.save(path)

    doc = open_document(path)
    assert type(doc).__name__ == "MangaPdf" and len(doc.units) == 2
    assert pymupdf.Pixmap(doc.image(0)).width == 800  # the scan's own resolution
    llm = ScriptedLLM()
    events = await translate(
        path, TranslationPipeline(llm=llm, enable_review=False), tmp_path / "out.pdf", reader=reader
    )
    out = pymupdf.open(events[-1].data["output_path"])
    assert [p.rect for p in out] == [pymupdf.Rect(0, 0, 400, 600)] * 2


def test_manga_upload_preview_and_page_images(tmp_path, monkeypatch, db):
    from fastapi.testclient import TestClient

    from app.main import app

    path, _, _ = _chapter(tmp_path, monkeypatch)
    with TestClient(app) as client, open(path, "rb") as f:
        book = client.post("/api/upload", files={"file": ("ch1.cbz", f)}).json()
        assert book["manga"] is True and book["total_chapters"] == 2
        assert client.get(f"/api/books/{book['id']}/preview").json()["images"] is True
        page = client.get(f"/api/books/{book['id']}/pages/0.png?translated=true")
        assert page.status_code == 200 and page.headers["content-type"] == "image/png"
        assert pymupdf.Pixmap(page.content).width == 800  # nothing translated: the original
        assert client.get(f"/api/books/{book['id']}/pages/9.png").status_code == 400


async def test_estimate_waits_for_the_pages_to_be_read(tmp_path, monkeypatch, db):
    from fastapi.testclient import TestClient

    from app.main import app

    path, _, reader = _chapter(tmp_path, monkeypatch)
    with TestClient(app) as client:
        with open(path, "rb") as f:
            book = client.post("/api/upload", files={"file": ("ch1.cbz", f)}).json()
        before = client.get(f"/api/books/{book['id']}/estimate").json()
        assert before["unread_pages"] == 2 and before["profile"]["source_tokens"] == 0
        with sqlite3.connect(settings.db_path) as conn:
            sql = "select original_path from books where id = ?"
            (upload,) = conn.execute(sql, (book["id"],)).fetchone()
        doc = open_document(upload)
        [e async for e in doc.prepare(reader, asyncio.Event())]
        after = client.get(f"/api/books/{book['id']}/estimate").json()
        assert after["unread_pages"] == 0 and after["profile"]["source_tokens"] > 0


def test_lettering_case_follows_most_letters():
    from app.services.manga.typeset import letter

    page = pymupdf.open().new_page(width=600, height=600)
    region = {"box": [100, 100, 300, 200], "bubble": None, "text": "Che IL MiNISTRO"}
    letter(page, region, "the minister")
    assert "THE" in page.get_text()


def test_a_folder_of_pages_is_uploaded_as_a_cbz(tmp_path, db):
    from fastapi.testclient import TestClient

    from app.main import app

    png, _ = sample_page()
    files = [
        ("files", ("Ch1/p2.png", png, "image/png")),
        ("files", ("Ch1/p1.png", png, "image/png")),
        ("files", ("Ch1/../../escape.png", png, "image/png")),  # stays inside the archive
        ("files", ("Ch1/notes.txt", b"not a page", "text/plain")),
    ]
    with TestClient(app) as client:
        book = client.post("/api/upload-folder", files=files, data={"name": "Ch1"}).json()
        assert book["filename"] == "Ch1.cbz" and book["manga"] is True
        assert book["total_chapters"] == 3
        empty = client.post("/api/upload-folder", files=[files[3]], data={"name": "Ch1"})
        assert empty.status_code == 400
    with sqlite3.connect(settings.db_path) as conn:
        (upload,) = conn.execute(
            "select original_path from books where id = ?", (book["id"],)
        ).fetchone()
    assert open_document(upload).pages == ["escape.png", "p1.png", "p2.png"]


async def test_text_too_long_for_its_bubble_goes_to_the_doubt_review(tmp_path, monkeypatch):
    path, _, reader = _chapter(tmp_path, monkeypatch, pages=1)
    monkeypatch.setattr(document, "fits", lambda region, text: region["text"] != "Meanwhile...")
    llm = ScriptedLLM()
    llm.handlers[DoubtResolution] = lambda text, call: DoubtResolution(translation="Later...")
    events = await translate(
        path, TranslationPipeline(llm=llm, enable_review=False), tmp_path / "out.cbz", reader=reader
    )
    flagged = [e for e in events if e.event_type == "doubt_flagged"]
    assert [d["block"] for e in flagged for d in e.data["doubts"]] == [2]
    assert "speech bubble" in flagged[0].data["doubts"][0]["reason"]
    assert len(llm.calls_for(DoubtResolution)) == 1  # shortened by the doubt review
    saved = [e for e in events if e.event_type == "chapter_saved"][-1]
    assert saved.data["translations"][2] == "Later..."
