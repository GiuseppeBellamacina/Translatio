"""Manga ML service: the GPU models of manga mode behind a small HTTP API.

POST /detect takes an image as the raw request body and returns the text and bubble boxes
found by comic-text-and-bubble-detector (RT-DETR-v2, Apache-2.0).
POST /ocr reads the given boxes of a page with one of the OCR engines (all Apache-2.0),
loaded on first use.
"""

import base64
import io
from collections.abc import Callable
from functools import cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

DETECTOR = "ogkalu/comic-text-and-bubble-detector"
EASYOCR_LANGS = {"ja": ["ja", "en"], "it": ["it", "en"], "en": ["en"]}
PAD = 4  # px around each box, so glyphs touching its edge are whole
UPSCALE = 2  # lettering on web-size pages is small; enlarged crops read better
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = RTDetrImageProcessor.from_pretrained(DETECTOR)
model = RTDetrV2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

app = FastAPI(title="Translatio manga ML")


def _image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(400, "not an image") from e


@app.get("/health")
def health() -> dict:
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    return {"status": "ok", "device": device, "gpu": gpu, "detector": DETECTOR}


@app.post("/detect")
async def detect(request: Request, threshold: float = 0.3) -> dict:
    """Boxes in image pixels: label bubble, text_bubble (text in a bubble) or text_free."""
    image = _image(await request.body())
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    found = processor.post_process_object_detection(
        outputs, target_sizes=[(image.height, image.width)], threshold=threshold
    )[0]
    regions = [
        {
            "box": [round(v, 1) for v in box.tolist()],
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
        }
        for box, label, score in zip(found["boxes"], found["labels"], found["scores"], strict=True)
    ]
    return {"width": image.width, "height": image.height, "regions": regions}


@cache
def _engine(name: str, lang: str) -> Callable[[Image.Image], list[str]]:
    """The engine's reader: a crop in, its lines of text out."""
    if name == "manga-ocr":
        from manga_ocr import MangaOcr

        reader = MangaOcr()
        return lambda crop: [reader(crop)]
    if name == "easyocr":
        import easyocr

        reader = easyocr.Reader(EASYOCR_LANGS.get(lang, [lang]), gpu=device == "cuda")
        return lambda crop: reader.readtext(np.array(crop), detail=0, paragraph=True)
    if name == "doctr":
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(pretrained=True).to(device)

        def read(crop: Image.Image) -> list[str]:
            page = predictor([np.array(crop)]).pages[0]
            return [" ".join(w.value for w in line.words) for b in page.blocks for line in b.lines]

        return read
    raise HTTPException(400, f"unknown OCR engine {name!r}")


class OcrRequest(BaseModel):
    image: str  # base64
    boxes: list[list[float]]


@app.post("/ocr")
def ocr(request: OcrRequest, engine: str, lang: str = "en") -> dict:
    """The text of each box, its lines separated by newlines."""
    read = _engine(engine, lang)
    image = _image(base64.b64decode(request.image))
    texts = []
    for x0, y0, x1, y1 in request.boxes:
        area = (max(x0 - PAD, 0), max(y0 - PAD, 0), min(x1 + PAD, image.width), min(y1 + PAD, image.height))
        with torch.inference_mode():
            crop = image.crop(area)
            if engine != "manga-ocr":  # it resizes to its own input size
                crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)
            texts.append("\n".join(read(crop)))
    return {"texts": texts}
