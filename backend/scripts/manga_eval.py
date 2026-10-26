"""Score a manga text detector (and its OCR) on pages with hand-checked ground truth.

    docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.manga.yml \
        run --rm backend python scripts/manga_eval.py /data/manga-test/en/spy-x-family-1

Each page image (png, jpg, webp) has a ``<page>.json`` next to it:
``{"boxes": [{"box": [x0, y0, x1, y1], "kind": "dialogue|caption|sfx", "text": "..."}]}``
in image pixels and reading order (app/services/manga/metrics.py). ``--bootstrap`` drafts that
file for the pages that have none (every region, SFX labelled) and draws it in
``check/<page>.png`` (numbered; blue dialogue, green captions, red SFX), to correct by hand;
those pages are not scored.
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

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.manga.detect import TEXT_LABELS  # noqa: E402
from app.services.manga.detect import detect as ml_detect  # noqa: E402
from app.services.manga.metrics import SFX, rates, score_page  # noqa: E402
from app.services.manga.ocr import ENGINES, read  # noqa: E402
from app.services.manga.panels import panels  # noqa: E402
from app.services.manga.regions import regions, text_regions  # noqa: E402

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
Detector = Callable[[Path], Awaitable[list[dict]]]
right_to_left = True  # manga in any language; --ltr for western comics
COLOURS = {"dialogue": (0, 0.4, 1), "caption": (0, 0.6, 0), SFX: (0.9, 0, 0)}


async def _truth(image: Path) -> list[dict]:
    """The ground truth without SFX: checks that the files load (it scores 100%)."""
    boxes = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))["boxes"]
    return [b for b in boxes if b["kind"] != SFX]


async def _ml(image: Path) -> list[dict]:
    """The GPU detector's text boxes (in bubbles and free), SFX included."""
    return [r for r in await ml_detect(image.read_bytes()) if r["label"] in TEXT_LABELS]


async def _regions(image: Path) -> list[dict]:
    """Text regions in reading order without SFX."""
    data = image.read_bytes()
    detected = await ml_detect(data)
    found = panels(data, [d["box"] for d in detected], right_to_left)
    return text_regions(detected, right_to_left, found)


async def _all_regions(image: Path) -> list[dict]:
    """Every region in reading order, SFX labelled: the drafts of `--bootstrap`."""
    data = image.read_bytes()
    detected = await ml_detect(data)
    return regions(
        detected, right_to_left, panels(data, [d["box"] for d in detected], right_to_left)
    )


DETECTORS: dict[str, Detector] = {"truth": _truth, "ml": _ml, "regions": _regions}


def check_image(image: Path, boxes: list[dict]) -> Path:
    """The page with its boxes numbered in reading order and coloured by kind, in ``check/``."""
    size = pymupdf.Pixmap(str(image))
    doc = pymupdf.open()
    page = doc.new_page(width=size.width, height=size.height)
    page.insert_image(page.rect, filename=str(image))
    for n, b in enumerate(boxes, 1):
        colour = COLOURS.get(b["kind"], (0.5, 0, 0.5))
        page.draw_rect(b["box"], color=colour, width=3)
        corner = (b["box"][0], b["box"][1] - 5 if b["box"][1] > 30 else b["box"][3] + 25)
        page.insert_text(corner, str(n), fontsize=26, fontname="hebo", color=colour)
    out = image.parent / "check" / f"{image.stem}.png"
    out.parent.mkdir(exist_ok=True)
    page.get_pixmap(dpi=72).save(out)
    return out


def _natural(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _pages(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGES), key=_natural)


async def evaluate(
    folder: Path, detect: Detector, draft: Detector | None = None
) -> tuple[list[tuple[str, Counter]], Counter]:
    """Counts per scored page (with the detector's milliseconds) and their total.

    Pages without ground truth are not scored; with ``draft`` they get one to correct by hand.
    """
    rows, total = [], Counter()
    for image in _pages(folder):
        truth = image.with_suffix(".json")
        if not truth.exists():
            if draft:
                boxes = [
                    {"box": r["box"], "kind": r["kind"], "text": r.get("text", "")}
                    for r in await draft(image)
                ]
                truth.write_text(
                    json.dumps({"boxes": boxes}, indent=2, ensure_ascii=False), "utf-8"
                )
                check_image(image, boxes)
            continue
        start = time.perf_counter()
        pred = await detect(image)
        ms = round((time.perf_counter() - start) * 1000)
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
    parser.add_argument("--detector", choices=DETECTORS, default="regions")
    parser.add_argument(
        "--bootstrap", action="store_true", help="draft missing ground truth, SFX included"
    )
    parser.add_argument("--ltr", action="store_true", help="read left to right (western comics)")
    parser.add_argument("--ocr", choices=ENGINES, help="read the regions, to score the CER")
    parser.add_argument("--lang", default="en", help="source language code for the OCR")
    args = parser.parse_args()
    global right_to_left
    right_to_left = not args.ltr
    detect = DETECTORS[args.detector]

    async def with_text(image: Path) -> list[dict]:
        found = await detect(image)
        texts = await read(image.read_bytes(), [r["box"] for r in found], args.ocr, args.lang)
        return [{**r, "text": t} for r, t in zip(found, texts, strict=True)]

    rows, total = asyncio.run(
        evaluate(
            args.folder, with_text if args.ocr else detect, _all_regions if args.bootstrap else None
        )
    )
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
