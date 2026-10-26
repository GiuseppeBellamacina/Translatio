"""OCR of manga text regions through the GPU service (ml/app.py), and the cleanup of its lines."""

import base64
import re
import unicodedata

import httpx

from app.services.manga.detect import service

ENGINES = ("manga-ocr", "easyocr", "doctr")
# a word hyphenated at a line end: "PROMES-\nSO", or "RIPOR - TARLA" once the OCR joined lines
_HYPHEN = re.compile(r"(\w)(?:-\n\s*| ?- +)(\w)")


def clean(text: str) -> str:
    """One line: hyphenated words joined, fullwidth forms folded ("ＢＡＭ！" → "BAM!")."""
    return " ".join(_HYPHEN.sub(r"\1\2", unicodedata.normalize("NFKC", text)).split())


async def read(
    image: bytes,
    boxes: list[list[float]],
    engine: str,
    language: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[str]:
    """The text of each box of the page; ``language`` is a code such as en, ja, it."""
    payload = {"image": base64.b64encode(image).decode(), "boxes": boxes}
    async with service(transport) as client:
        response = await client.post(
            "/ocr", json=payload, params={"engine": engine, "lang": language}
        )
        response.raise_for_status()
        return [clean(t) for t in response.json()["texts"]]
