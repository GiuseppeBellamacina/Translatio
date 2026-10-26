"""OCR by a vision LLM: a page's text crops, enlarged and numbered on sheets, one call per sheet."""

import base64

import pymupdf
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.core.llm_factory import structured
from app.models.schemas import Transcription
from app.services.manga.ocr import clean

SCALE = 2  # crops are enlarged: lettering on web-size pages is small
SHEET_HEIGHT = 1600  # px; the APIs shrink taller images
GAP, LABEL = 16, 56  # px between crops, and left of them for the number


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an OCR engine for comics. Transcribe the lettering of each numbered box "
            "exactly as written: same language, words and punctuation; never translate or "
            "correct. Join the lines of a box into one line and rejoin a word hyphenated at a "
            "line end. A box without text gives an empty string. Write a sound effect "
            "(onomatopoeia such as BAM, SLAM, TOSS, DING DONG) inside square brackets: [BAM]. "
            "The red numbers are not text.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Boxes: {count}. Return exactly {count} texts."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{sheet}"}},
            ],
        ),
    ]
)


def sheets(image: bytes, boxes: list[list[float]]) -> list[tuple[bytes, int]]:
    """PNG sheets of the boxes' crops, numbered from 1 on each sheet, with their box counts."""
    pix = pymupdf.Pixmap(image)
    src = pymupdf.open()
    page = src.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, stream=image)
    groups: list[list[pymupdf.Rect]] = []
    height = SHEET_HEIGHT
    for box in boxes:
        rect = pymupdf.Rect(box) & page.rect
        if height + rect.height * SCALE + GAP > SHEET_HEIGHT:
            groups.append([])
            height = GAP
        groups[-1].append(rect)
        height += rect.height * SCALE + GAP
    found = []
    for group in groups:
        width = LABEL + SCALE * max(r.width for r in group) + GAP
        sheet = pymupdf.open().new_page(
            width=width, height=GAP + sum(r.height * SCALE + GAP for r in group)
        )
        y = GAP
        for n, rect in enumerate(group, 1):
            sheet.insert_text((8, y + 24), str(n), fontsize=22, fontname="hebo", color=(0.8, 0, 0))
            target = pymupdf.Rect(LABEL, y, LABEL + rect.width * SCALE, y + rect.height * SCALE)
            sheet.show_pdf_page(target, src, 0, clip=rect)
            y = target.y1 + GAP
            sheet.draw_line((0, y - GAP / 2), (width, y - GAP / 2), color=(0.6, 0.6, 0.6))
        found.append((sheet.get_pixmap(dpi=72).tobytes("png"), len(group)))
    return found


async def _sheet(png: bytes, count: int, llm: BaseChatModel) -> list[str]:
    def check(result: Transcription) -> list[str]:
        if count == 1 and result.texts:  # one box read as several lines
            return [" ".join(result.texts)]
        if len(result.texts) != count:  # retried: a lost or merged box shifts every text
            raise ValueError(f"{len(result.texts)} texts for {count} boxes")
        return result.texts

    chain = (PROMPT | structured(llm, Transcription) | RunnableLambda(check)).with_retry(
        stop_after_attempt=max(1, settings.chain_retries), wait_exponential_jitter=True
    )
    return await chain.ainvoke({"count": count, "sheet": base64.b64encode(png).decode()})


async def read(image: bytes, boxes: list[list[float]], llm: BaseChatModel) -> list[str]:
    """The text of each box, read by ``llm`` (a vision model)."""
    texts: list[str] = []
    for png, count in sheets(image, boxes):
        try:
            texts += await _sheet(png, count, llm)
        except Exception:
            if count == 1:
                raise
            # a sheet the model keeps miscounting (or looping on): its boxes one at a time
            for box in boxes[len(texts) : len(texts) + count]:
                ((single, _),) = sheets(image, [box])
                texts += await _sheet(single, 1, llm)
    return [clean(t) for t in texts]
