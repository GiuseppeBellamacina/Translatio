"""Manga ML service: the GPU models of manga mode behind a small HTTP API.

POST /detect takes an image as the raw request body and returns the text and bubble boxes
found by comic-text-and-bubble-detector (RT-DETR-v2, Apache-2.0).
"""

import io

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

DETECTOR = "ogkalu/comic-text-and-bubble-detector"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = RTDetrImageProcessor.from_pretrained(DETECTOR)
model = RTDetrV2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

app = FastAPI(title="Translatio manga ML")


@app.get("/health")
def health() -> dict:
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    return {"status": "ok", "device": device, "gpu": gpu, "detector": DETECTOR}


@app.post("/detect")
async def detect(request: Request, threshold: float = 0.3) -> dict:
    """Boxes in image pixels: label bubble, text_bubble (text in a bubble) or text_free."""
    try:
        image = Image.open(io.BytesIO(await request.body())).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(400, "not an image") from e
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
