"""Score a manga text detector (and its OCR) on pages with hand-checked ground truth.

    docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm backend \
        python scripts/manga_eval.py /data/manga-test/en/spy-x-family-1

Each page image (png, jpg, webp) has a ``<page>.json`` next to it:
``{"boxes": [{"box": [x0, y0, x1, y1], "kind": "dialogue|caption|sfx", "text": "..."}]}``
in image pixels (app/services/manga/metrics.py). ``--bootstrap`` writes the detector's output
as that file for the pages that have none, to correct by hand; those pages are not scored.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.manga.detect import TEXT_LABELS  # noqa: E402
from app.services.manga.detect import detect as ml_detect  # noqa: E402
from app.services.manga.metrics import SFX, rates, score_page  # noqa: E402

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
Detector = Callable[[Path], Awaitable[list[dict]]]


async def _truth(image: Path) -> list[dict]:
    """The ground truth without SFX: checks that the files load (it scores 100%)."""
    boxes = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))["boxes"]
    return [b for b in boxes if b["kind"] != SFX]


async def _ml(image: Path) -> list[dict]:
    """The GPU detector's text boxes (in bubbles and free), SFX included."""
    return [r for r in await ml_detect(image.read_bytes()) if r["label"] in TEXT_LABELS]


DETECTORS: dict[str, Detector] = {"truth": _truth, "ml": _ml}


def _natural(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _pages(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGES), key=_natural)


async def evaluate(
    folder: Path, detect: Detector, bootstrap: bool = False
) -> tuple[list[tuple[str, Counter]], Counter]:
    """Counts per scored page (with the detector's milliseconds) and their total."""
    rows, total = [], Counter()
    for image in _pages(folder):
        truth = image.with_suffix(".json")
        if not truth.exists() and not bootstrap:
            continue
        start = time.perf_counter()
        pred = await detect(image)
        ms = round((time.perf_counter() - start) * 1000)
        if not truth.exists():
            draft = [
                {"box": p["box"], "kind": p.get("kind", "dialogue"), "text": p.get("text", "")}
                for p in pred
            ]
            truth.write_text(json.dumps({"boxes": draft}, indent=2, ensure_ascii=False), "utf-8")
            continue
        counts = score_page(pred, json.loads(truth.read_text(encoding="utf-8"))["boxes"])
        counts["ms"] = ms
        rows.append((image.name, counts))
        total += counts
    return rows, total


def _line(name: str, counts: Counter, pages: int = 1) -> str:
    r = rates(counts)
    cells = [f"{r[k]:.2f}" if r[k] is not None else "-" for k in r]
    ms = counts["ms"] / pages
    return (
        f"{name:<28} {counts['pred']:>5} {counts['truth']:>5} {counts['hits']:>5} "
        + " ".join(f"{c:>6}" for c in cells)
        + f" {ms:>7.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a manga text detector on pages with ground truth."
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--detector", choices=DETECTORS, default="truth")
    parser.add_argument("--bootstrap", action="store_true", help="write missing ground truth")
    args = parser.parse_args()
    rows, total = asyncio.run(evaluate(args.folder, DETECTORS[args.detector], args.bootstrap))
    print(
        f"{'page':<28} {'pred':>5} {'truth':>5} {'hits':>5} {'prec':>6} {'recall':>6} "
        f"{'sfx':>6} {'cer':>6} {'order':>6} {'ms/page':>7}"
    )
    for name, counts in rows:
        print(_line(name, counts))
    if rows:
        print(_line(f"TOTAL ({len(rows)} pages)", total, len(rows)))


if __name__ == "__main__":
    main()
