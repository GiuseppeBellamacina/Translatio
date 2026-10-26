"""Manga ML service: the GPU models of manga mode behind a small HTTP API.

POST /detect takes an image as the raw request body and returns the text and bubble boxes
found by comic-text-and-bubble-detector (RT-DETR-v2, Apache-2.0).
POST /ocr reads the given boxes of a page with one of the OCR engines (all Apache-2.0),
loaded on first use. POST /erase removes the text of the given boxes (fill or LaMa).
"""

import base64
import io
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path

import cv2
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
INK = 128  # grey level between lettering and its background
SPECK = 12  # px: ink blobs smaller than this do not tell the letter height
PLAIN = 12  # grey spread under which a box's background is plain colour, filled not inpainted
# LaMa big model as TorchScript (Apache-2.0), as packaged by simple-lama-inpainting
LAMA_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)
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


@cache
def _lama() -> torch.jit.ScriptModule:
    path = Path(os.environ.get("HF_HOME", "/models")) / "lama" / "big-lama.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(LAMA_URL, str(path))
    return torch.jit.load(str(path), map_location=device).eval()


def _inpaint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa on the masked pixels (the input padded to a multiple of 8); the rest is kept."""
    h, w = mask.shape
    pad = ((0, -h % 8), (0, -w % 8))
    image = np.pad(rgb, (*pad, (0, 0)), mode="symmetric")
    holes = np.pad(mask, pad, mode="symmetric") > 0
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float().div(255).to(device)
    holes_tensor = torch.from_numpy(holes.astype(np.float32))[None, None].to(device)
    with torch.inference_mode():
        out = _lama()(tensor, holes_tensor)[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]
    return np.where(mask[..., None] > 0, out, rgb)


def _glyphs(ink: np.ndarray) -> np.ndarray:
    """The ink blobs sized like the box's letters: a dark hair or shadow inside it stays."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    if not count - 1:
        return np.zeros(ink.shape, np.uint8)
    sized = heights[stats[1:, cv2.CC_STAT_AREA] >= SPECK]  # dots and screentone left out
    letter = np.median(sized) if sized.size else heights.max()
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    keep = np.flatnonzero((heights <= 2 * letter) & (widths <= 4 * letter)) + 1
    return np.isin(labels, keep).astype(np.uint8)


class EraseRequest(BaseModel):
    image: str  # base64
    boxes: list[list[float]]


@app.post("/erase")
def erase(request: EraseRequest) -> dict:
    """The page without the text of the boxes: a plain background is filled with its own
    colour, text over the art is inpainted by LaMa (all of a page's art in one pass)."""
    rgb = np.array(_image(base64.b64decode(request.image)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    art = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in request.boxes:
        x0, y0 = max(int(x0) - PAD, 0), max(int(y0) - PAD, 0)
        x1, y1 = min(int(x1) + PAD, gray.shape[1]), min(int(y1) + PAD, gray.shape[0])
        patch = gray[y0:y1, x0:x1]
        ink = patch < INK if np.median(patch) >= INK else patch > INK  # dark or light lettering
        mask = cv2.dilate(_glyphs(ink), np.ones((5, 5), np.uint8))  # glyph edges
        rest = patch[mask == 0]
        if rest.size and rest.std() < PLAIN:
            area = rgb[y0:y1, x0:x1]
            area[mask > 0] = np.median(area[mask == 0], axis=0)
        else:
            art[y0:y1, x0:x1] |= mask
    if art.any():
        rgb = _inpaint(rgb, art)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="PNG")
    return {"image": base64.b64encode(out.getvalue()).decode()}
