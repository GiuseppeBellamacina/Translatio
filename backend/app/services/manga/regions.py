"""Text regions of a manga page from the detector's boxes (app/services/manga/detect.py).

A region is ``{"box", "kind": dialogue|caption|sfx, "bubble": box | None, "score"}``:
text in a bubble is dialogue (its bubble is kept for typesetting), free text is a caption,
unless it is much bigger than the page's bubble text: then it is a sound effect, left alone.
"""

from statistics import median

from app.services.manga.metrics import SFX

# shortcut: compares box short sides, so a tall multi-line caption can pass for a sound effect;
# switch to glyph size (box area / OCR characters) once OCR is in.
SFX_SCALE = 1.2


def _inside(inner: list[float], outer: list[float]) -> float:
    """Share of ``inner``'s area that lies in ``outer``."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def _short(box: list[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def regions(detected: list[dict]) -> list[dict]:
    bubbles = [d["box"] for d in detected if d["label"] == "bubble"]
    in_bubbles = [_short(d["box"]) for d in detected if d["label"] == "text_bubble"]
    usual = median(in_bubbles) if in_bubbles else None  # no bubble text: nothing to compare
    found = []
    for d in detected:
        if d["label"] == "bubble":
            continue
        holder = max(bubbles, key=lambda b: _inside(d["box"], b), default=None)
        if holder is not None and _inside(d["box"], holder) < 0.5:
            holder = None
        if d["label"] == "text_bubble":
            kind = "dialogue"
        elif usual and _short(d["box"]) > SFX_SCALE * usual:
            kind = SFX
        else:
            kind = "caption"
        found.append({"box": d["box"], "kind": kind, "bubble": holder, "score": d["score"]})
    return found


def text_regions(detected: list[dict]) -> list[dict]:
    """The regions to translate: sound effects are left alone."""
    return [r for r in regions(detected) if r["kind"] != SFX]
