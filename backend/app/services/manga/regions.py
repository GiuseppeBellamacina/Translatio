"""Text regions of a manga page from the detector's boxes (app/services/manga/detect.py).

A region is ``{"box", "kind": dialogue|caption|sfx, "bubble": box | None, "score"}``:
text in a bubble is dialogue (its bubble is kept for typesetting), free text is a caption.
"""

from app.services.manga.metrics import SFX


def _inside(inner: list[float], outer: list[float]) -> float:
    """Share of ``inner``'s area that lies in ``outer``."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def regions(detected: list[dict]) -> list[dict]:
    bubbles = [d["box"] for d in detected if d["label"] == "bubble"]
    found = []
    for d in detected:
        if d["label"] == "bubble":
            continue
        holder = max(bubbles, key=lambda b: _inside(d["box"], b), default=None)
        if holder is not None and _inside(d["box"], holder) < 0.5:
            holder = None
        kind = "dialogue" if d["label"] == "text_bubble" else "caption"
        found.append({"box": d["box"], "kind": kind, "bubble": holder, "score": d["score"]})
    return found


def text_regions(detected: list[dict]) -> list[dict]:
    """The regions to translate: sound effects are left alone."""
    return [r for r in regions(detected) if r["kind"] != SFX]
