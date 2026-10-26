"""Books: upload, list, delete, detail, preview, progress, estimate, download."""

import asyncio
import mimetypes
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routers.queue import queue_items
from app.api.schemas import (
    BookDetailResponse,
    BookResponse,
    ChapterResponse,
)
from app.core.bootstrap import load_config
from app.core.config import settings
from app.core.encryption import decrypt_api_key
from app.db.database import get_db
from app.db.models import (
    Book,
    Bookmark,
    BookStatus,
    Chapter,
    ChapterStatus,
    CompendiumRecord,
    GlossaryEntry,
    ProviderConfig,
    QueueItem,
    TranslationJob,
)
from app.jobs import jobs
from app.services.chunker import make_chunker
from app.services.cost_estimator import DEFAULT_MODELS, TokenProfile, estimate, profile_document
from app.services.documents import open_document
from app.services.html_processor import extract_visible_text
from app.services.manga.document import IMAGES, MangaDocument, cache_dir
from app.services.model_catalog import list_models, price_for
from app.services.pricing import ModelPrice, PriceTable, get_prices

router = APIRouter()


router = APIRouter()


# === BOOKS ===


SUPPORTED_SUFFIXES = {".epub", ".pdf", ".cbz", ".zip"}  # .cbz/.zip: manga pages


@router.post("/upload", response_model=BookResponse)
async def upload_book(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, "Only .epub, .pdf and .cbz/.zip (manga) files are supported")

    # Unique name on disk: two uploads with the same filename must not overwrite each other
    upload_path = settings.uploads_dir / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return await _register(upload_path, Path(file.filename).name, db)


@router.post("/upload-folder", response_model=BookResponse)
async def upload_folder(
    files: list[UploadFile] = File(...), name: str = Form(...), db: AsyncSession = Depends(get_db)
):
    """A folder of page images (a manga chapter): packed into a CBZ, then a book like one."""
    pages = [f for f in files if Path(f.filename or "").suffix.lower() in IMAGES]
    if not pages:
        raise HTTPException(400, "The folder has no page images (.png, .jpg, .webp)")
    filename = f"{Path(name).name or 'pages'}.cbz"
    upload_path = settings.uploads_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
    with zipfile.ZipFile(upload_path, "w") as archive:
        for page in pages:
            # the path inside the folder (subfolders kept), never outside the archive
            parts = [p for p in Path(page.filename).parts[1:] if p not in ("", ".", "..")]
            archive.writestr("/".join(parts) or Path(page.filename).name, page.file.read())
    return await _register(upload_path, filename, db)


async def _register(upload_path: Path, filename: str, db: AsyncSession) -> Book:
    """A book (and its chapters) for a file just written to the uploads folder."""
    suffix = upload_path.suffix.lower()
    try:
        doc = open_document(upload_path)
        info, units = doc.info(), doc.units
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid {suffix[1:].upper()} file: {e}")

    book = Book(
        filename=filename,
        original_path=str(upload_path),
        source_language=info.get("language") or "english",
        target_language="italian",
        status=BookStatus.uploaded,
        total_chapters=len(units),
        manga=isinstance(doc, MangaDocument),
    )
    db.add(book)
    await db.commit()
    await db.refresh(book)

    for i, unit in enumerate(units):
        db.add(
            Chapter(
                book_id=book.id,
                item_id=unit.item_id,
                title=unit.title,
                order_index=i,
                status=ChapterStatus.pending,
            )
        )
    await db.commit()

    return book


@router.get("/books", response_model=list[BookResponse])
async def list_books(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Book).order_by(Book.created_at.desc()))
    return result.scalars().all()


@router.delete("/books/{book_id}")
async def delete_book(book_id: int, db: AsyncSession = Depends(get_db)):
    """Remove a book with its progress, glossary, jobs, queue entries and files."""
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    if jobs.is_running(book_id):
        raise HTTPException(409, "Pause the translation before deleting the book")
    for model in (Chapter, GlossaryEntry, CompendiumRecord, Bookmark, TranslationJob, QueueItem):
        await db.execute(delete(model).where(model.book_id == book_id))
    files = [Path(p) for p in (book.original_path, book.translated_path) if p]
    await db.execute(delete(Book).where(Book.id == book_id))
    await db.commit()
    for path in files:
        if path.resolve().is_relative_to(settings.data_dir.resolve()):  # only our own files
            path.unlink(missing_ok=True)
            shutil.rmtree(cache_dir(path), ignore_errors=True)  # manga: the pages read
    jobs.events.recent.pop(book_id, None)
    return {"status": "deleted"}


@router.get("/books/{book_id}", response_model=BookDetailResponse)
async def get_book(book_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(404, "Book not found")

    # Get chapters
    chapters_result = await db.execute(
        select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.order_index)
    )
    chapters = chapters_result.scalars().all()

    return BookDetailResponse(
        **{c.name: getattr(book, c.name) for c in Book.__table__.columns},
        chapters=[
            ChapterResponse(
                id=ch.id,
                item_id=ch.item_id,
                title=ch.title,
                order_index=ch.order_index,
                status=ch.status.value if ch.status else "pending",
                has_original=ch.original_html is not None,
                has_translation=ch.translated_html is not None,
            )
            for ch in chapters
        ],
    )


@router.get("/books/{book_id}/preview")
async def get_preview(book_id: int, chapter_index: int = 0, db: AsyncSession = Depends(get_db)):
    """Get side-by-side preview of original vs translated for a chapter (or PDF page)."""
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")

    doc = open_document(book.original_path)
    if not 0 <= chapter_index < len(doc.units):
        raise HTTPException(400, "Chapter index out of range")

    result = await db.execute(
        select(Chapter).where(Chapter.book_id == book_id, Chapter.order_index == chapter_index)
    )
    chapter = result.scalar_one_or_none()
    translated_text = ""
    if chapter and chapter.translated_html:
        translated_text = extract_visible_text(chapter.translated_html)

    return {
        "chapter_index": chapter_index,
        "total_chapters": len(doc.units),
        "original_text": doc.text(chapter_index),
        "translated_text": translated_text,
        "original_html": doc.render(chapter_index, None) or "",
        "images": isinstance(doc, MangaDocument),  # a manga page: see page_image
    }


@router.get("/books/{book_id}/pages/{index}.png")
async def page_image(
    book_id: int, index: int, translated: bool = False, db: AsyncSession = Depends(get_db)
):
    """A manga page as an image: the original, or lettered with the translations so far."""
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    doc = open_document(book.original_path)
    if not isinstance(doc, MangaDocument):
        raise HTTPException(400, "Only manga books have page images")
    if not 0 <= index < len(doc.units):
        raise HTTPException(400, "Page index out of range")
    image = None
    if translated:
        chapter = (
            await db.execute(
                select(Chapter).where(Chapter.book_id == book_id, Chapter.order_index == index)
            )
        ).scalar_one_or_none()
        if chapter and chapter.block_translations:
            image = await asyncio.to_thread(doc.page, index, chapter.block_translations)
    if image:
        return Response(image, media_type="image/png")
    kind = mimetypes.guess_type(doc.pages[index])[0] or "image/png"
    return Response(doc.image(index), media_type=kind)


@router.get("/books/{book_id}/bookmarks")
async def get_bookmarks(book_id: int, db: AsyncSession = Depends(get_db)):
    """The passages translated with doubts, and what the final review decided."""
    rows = await db.execute(
        select(Bookmark).where(Bookmark.book_id == book_id).order_by(Bookmark.unit, Bookmark.block)
    )
    return [
        {
            "id": b.id,
            "unit": b.unit,
            "block": b.block,
            "quote": b.quote,
            "reason": b.reason,
            "source": b.source,
            "translation": b.translation,
            "status": b.status,
            "resolution": b.resolution,
            "note": b.note,
        }
        for b in rows.scalars()
    ]


@router.get("/books/{book_id}/progress")
async def get_progress(book_id: int, db: AsyncSession = Depends(get_db)):
    """Run state of a book's translation, also usable after a page reload."""
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    rows = (
        await db.execute(
            select(Chapter.order_index, Chapter.status).where(Chapter.book_id == book_id)
        )
    ).all()
    completed = sum(status == ChapterStatus.completed for _, status in rows)
    running = jobs.is_running(book_id)
    job = (
        await db.execute(
            select(TranslationJob)
            .where(TranslationJob.book_id == book_id)
            .order_by(TranslationJob.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    eta = None
    if running and book_id in jobs.runner.started:
        start, done_at_start = jobs.runner.started[book_id]
        done_now = completed - done_at_start
        if done_now > 0:
            eta = (time.time() - start) / done_now * (book.total_chapters - completed)

    return {
        "status": book.status.value if book.status else "uploaded",
        "running": running,
        "total_units": book.total_chapters,
        "completed_units": completed,
        "failed_units": sorted(i for i, s in rows if s == ChapterStatus.failed),
        "in_flight": (
            sorted(i for i, s in rows if s == ChapterStatus.translating) if running else []
        ),
        "started_at": job.started_at.isoformat() if job and job.started_at else None,
        "eta_seconds": eta,
        "recent_events": jobs.events.log(book_id),
        # settings of the last run (older jobs did not keep them), to show and resume with
        "config": job.params if job and "provider_id" in (job.params or {}) else None,
        "queue_status": await db.scalar(
            select(QueueItem.status).where(
                QueueItem.book_id == book_id, QueueItem.status.in_(("pending", "paused"))
            )
        ),
        "queue": await _book_queue(db, book_id),
    }


async def _book_queue(db: AsyncSession, book_id: int) -> dict | None:
    """The queue (batch) the book was last added to, when it has other books too."""
    batch_id = await db.scalar(
        select(QueueItem.batch_id)
        .where(QueueItem.book_id == book_id, QueueItem.batch_id.is_not(None))
        .order_by(QueueItem.id.desc())
        .limit(1)
    )
    if batch_id is None:
        return None
    items = await queue_items(db, QueueItem.batch_id == batch_id)
    return {"batch_id": batch_id, "items": items} if len(items) > 1 else None


@router.get("/books/{book_id}/estimate")
async def estimate_cost(
    book_id: int,
    target_language: str | None = None,
    refresh_prices: bool = False,
    provider_id: int | None = None,
    model: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Estimated translation cost per model, with current prices.

    With ``provider_id`` and ``model``, only that model (empty list if its price is unknown).
    """
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")

    provider = await db.get(ProviderConfig, provider_id) if provider_id is not None else None
    if provider_id is not None and not provider:
        raise HTTPException(404, "Provider not found")
    chunk_tokens = ((provider and provider.params) or {}).get("chunk_tokens", settings.chunk_tokens)

    # the profile depends on the chunk size: cached for the last size asked
    cached = book.token_profile or {}
    doc = open_document(book.original_path)
    unread = doc.unread() if isinstance(doc, MangaDocument) else 0  # no text before the OCR
    if cached.get("chunk_tokens") != chunk_tokens or unread:
        chunker = make_chunker(chunk_tokens)
        profile = await asyncio.to_thread(profile_document, doc, chunker)
        cached = {**asdict(profile), "chunk_tokens": chunk_tokens}
        if not unread:  # a manga's profile is final only once every page is read
            book.token_profile = cached
            await db.commit()
    profile = TokenProfile(**{k: v for k, v in cached.items() if k != "chunk_tokens"})
    language = target_language or book.target_language
    prices = await get_prices(force_refresh=refresh_prices)

    if provider is not None and model:
        price = price_for(provider.provider_type, model, prices)
        if price is None and provider.provider_type in ("custom", "openrouter"):
            # OpenAI-compatible routers (OpenRouter) publish prices in their model list
            api_key = (
                decrypt_api_key(provider.api_key_encrypted) if provider.api_key_encrypted else None
            )
            try:
                listed = await list_models(
                    provider.provider_type, api_key, provider.base_url, prices
                )
            except Exception:
                listed = []
            option = next((m for m in listed if m["id"] == model), None)
            if option and option["input_per_million"] is not None:
                price = ModelPrice(
                    "openai",
                    option["input_per_million"] / 1e6,
                    option["output_per_million"] / 1e6,
                    reasoning=False,
                )
        if price is None:
            return {**estimate(profile, prices, language, {}), "unread_pages": unread}
        single = PriceTable({model: price}, prices.fetched_at, prices.source)
        return {
            **estimate(profile, single, language, {price.provider: [model]}),
            "unread_pages": unread,
        }

    # config.yaml pricing.models (it replaces the built-in list, so it can drop a vendor),
    # plus the default models of configured providers
    configured = (load_config().get("pricing") or {}).get("models")
    models = {p: list(names) for p, names in (configured or DEFAULT_MODELS).items()}
    providers = await db.execute(select(ProviderConfig).where(ProviderConfig.is_active.is_(True)))
    for provider in providers.scalars():
        if provider.provider_type in models and provider.default_model:
            models[provider.provider_type].append(provider.default_model)
    return {**estimate(profile, prices, language, models), "unread_pages": unread}


@router.get("/books/{book_id}/download")
async def download_book(book_id: int, db: AsyncSession = Depends(get_db)):
    from fastapi.responses import FileResponse

    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book or not book.translated_path:
        raise HTTPException(404, "Translated book not found")

    path = Path(book.translated_path)
    if not path.exists():
        raise HTTPException(404, "Translated file not found on disk")

    return FileResponse(
        path=str(path),
        filename=f"translated_{book.filename}",
        media_type={".pdf": "application/pdf", ".cbz": MangaDocument.media_type}.get(
            path.suffix, "application/epub+zip"
        ),
    )
