"""Client of the manga GPU service (ml/app.py): text and bubble boxes of a page image."""

import httpx

from app.core.config import settings

TEXT_LABELS = {"text_bubble", "text_free"}


async def detect(
    image: bytes, threshold: float = 0.3, transport: httpx.AsyncBaseTransport | None = None
) -> list[dict]:
    """``{"box": [x0, y0, x1, y1], "label": bubble|text_bubble|text_free, "score"}`` in pixels."""
    if not settings.manga_ml_url:
        raise RuntimeError("Manga mode needs the GPU service: set TRANSLATIO_MANGA_ML_URL")
    async with httpx.AsyncClient(
        base_url=settings.manga_ml_url, timeout=120, transport=transport
    ) as client:
        response = await client.post("/detect", content=image, params={"threshold": threshold})
        response.raise_for_status()
        return response.json()["regions"]
