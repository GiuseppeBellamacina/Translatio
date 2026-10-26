"""Scores of manga text detection and OCR against hand-checked ground truth.

A region is a dict ``{"box": [x0, y0, x1, y1], "kind": "dialogue|caption|sfx", "text": ...}``
in page pixels, listed in reading order. Sound effects in the ground truth must *not* be
detected; the rest must, in that order.
"""

import unicodedata
from collections import Counter
from collections.abc import Sequence

SFX = "sfx"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(
    pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]], threshold: float = 0.5
) -> list[tuple[int, int]]:
    """(pred index, truth index) pairs, best IoU first, each box used at most once."""
    scores = sorted(
        ((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)), reverse=True
    )
    used_pred, used_truth, pairs = set(), set(), []
    for score, i, j in scores:
        if score < threshold:
            break
        if i not in used_pred and j not in used_truth:
            used_pred.add(i)
            used_truth.add(j)
            pairs.append((i, j))
    return pairs


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def _clean(text: str) -> str:
    """Case and apostrophe style do not matter for translation."""
    return " ".join(unicodedata.normalize("NFKC", text).replace("’", "'").casefold().split())


def score_page(pred: list[dict], truth: list[dict]) -> Counter:
    """Counts for one page; add the counters of several pages and pass them to `rates`."""
    wanted = [t for t in truth if t["kind"] != SFX]
    sfx = [t for t in truth if t["kind"] == SFX]
    boxes = [p["box"] for p in pred]
    pairs = match(boxes, [t["box"] for t in wanted])
    counts = Counter(
        pred=len(pred),
        truth=len(wanted),
        hits=len(pairs),
        sfx=len(sfx),
        sfx_hits=len(match(boxes, [t["box"] for t in sfx])),
    )
    found_at = {j: i for i, j in pairs}  # ground truth is listed in reading order
    for j in range(len(wanted) - 1):
        if j in found_at and j + 1 in found_at:
            counts["order_pairs"] += 1
            counts["order_ok"] += found_at[j] < found_at[j + 1]
    for i, j in pairs:
        if pred[i].get("text") is not None and wanted[j].get("text"):
            text, expected = _clean(pred[i]["text"]), _clean(wanted[j]["text"])
            counts["chars"] += len(expected)
            counts["errors"] += edit_distance(text, expected)
    return counts


def rates(counts: Counter) -> dict[str, float | None]:
    """Precision, recall, share of SFX detected by mistake, character error rate (None: n/a)."""

    def ratio(a: str, b: str) -> float | None:
        return counts[a] / counts[b] if counts[b] else None

    return {
        "precision": ratio("hits", "pred"),
        "recall": ratio("hits", "truth"),
        "sfx_included": ratio("sfx_hits", "sfx"),
        "cer": ratio("errors", "chars"),
        "order": ratio("order_ok", "order_pairs"),
    }
