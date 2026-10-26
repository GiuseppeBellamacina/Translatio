"""Client of the GPU service's erasing (ml/app.py): a page without the text of its regions."""

import base64

import httpx

from app.services.manga.detect import service


async def erase(
    image: bytes, boxes: list[list[float]], transport: httpx.AsyncBaseTransport | None = None
) -> bytes:
    """The page as PNG, the lettering of ``boxes`` filled (plain background) or inpainted."""
    payload = {"image": base64.b64encode(image).decode(), "boxes": boxes}
    async with service(transport) as client:
        response = await client.post("/erase", json=payload)
        response.raise_for_status()
        return base64.b64decode(response.json()["image"])
