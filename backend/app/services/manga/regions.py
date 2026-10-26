"""Text regions of a manga page from the detector's boxes (app/services/manga/detect.py).

A region is ``{"box", "kind": dialogue|caption|sfx, "bubble": box | None, "score"}``:
text in a bubble is dialogue (its bubble is kept for typesetting), free text is a caption.
The detector does not pick up hand-drawn sound effects (checked on a real chapter); lettered
ones ("DING DONG") look like captions and need OCR to be told apart.
"""

from app.services.manga.metrics import SFX


def _inside(inner: list[float], outer: list[float]) -> float:
    """Share of ``inner``'s area that lies in ``outer``."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def reading_order(found: list[dict], right_to_left: bool) -> list[dict]:
    """Rows of regions that overlap vertically, top to bottom, each row read right to left
    (manga, in any language: translated editions keep the layout) or left to right (comics).

    shortcut: no panel detection, so a tall bubble can join two rows of panels; use Magi-like
    panel order if the eval's order score shows it.
    """
    area = lambda r: r.get("bubble") or r["box"]  # noqa: E731  a bubble places its text
    rows: list[tuple[float, list[dict]]] = []  # (row bottom, regions)
    for r in sorted(found, key=lambda r: area(r)[1]):
        top, bottom = area(r)[1], area(r)[3]
        if rows and rows[-1][0] - top >= (bottom - top) / 2:
            rows[-1] = (max(rows[-1][0], bottom), rows[-1][1] + [r])
        else:
            rows.append((bottom, [r]))
    return [
        r
        for _, row in rows
        for r in sorted(row, key=lambda r: area(r)[0] + area(r)[2], reverse=right_to_left)
    ]


def regions(detected: list[dict], right_to_left: bool = True) -> list[dict]:
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
    return reading_order(found, right_to_left)


def text_regions(detected: list[dict], right_to_left: bool = True) -> list[dict]:
    """The regions to translate, in reading order: sound effects are left alone."""
    return [r for r in regions(detected, right_to_left) if r["kind"] != SFX]
