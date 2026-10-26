"""OCR of manga text regions through the GPU service (ml/app.py), and the cleanup of its lines."""

import base64
import re
import unicodedata

import httpx

from app.services.manga.detect import service

ENGINES = ("manga-ocr", "easyocr", "doctr")
LANGUAGE_CODES = {"english": "en", "japanese": "ja", "italian": "it", "french": "fr"}
LANGUAGE_CODES |= {"german": "de", "spanish": "es", "portuguese": "pt", "korean": "ko"}
# a word hyphenated at a line end: "PROMES-\nSO", or "RIPOR - TARLA" once the OCR joined lines
_HYPHEN = re.compile(r"(\w)(?:-\n\s*| ?- +)(\w)")


def is_sfx(text: str) -> bool:
    """A sound effect, as the vision OCR marks them: the whole text in square brackets."""
    text = text.strip()
    return len(text) > 2 and text.startswith("[") and text.endswith("]")


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
