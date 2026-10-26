"""Generate a synthetic manga page with its ground truth, for tests and smoke checks.

    docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm backend \
        python scripts/make_sample_manga.py /data/manga-test/sample

writes ``page-1.png`` and ``page-1.json`` (the format read by ``scripts/manga_eval.py``):
two speech bubbles, a caption drawn over hatched art and a large sound effect.
"""

import json
import sys
from pathlib import Path

import pymupdf

WIDTH, HEIGHT = 800, 1200
# (kind, area, text, font size, bubble)
REGIONS = [
    ("dialogue", (60, 80, 360, 280), "WHERE DID\nYOU PUT\nTHE LETTER?", 22, True),
    ("dialogue", (440, 140, 740, 320), "IT'S NOT\nHERE.", 24, True),
    ("caption", (80, 700, 420, 760), "Meanwhile, at the station...", 18, False),
    ("sfx", (440, 800, 770, 980), "BAM!", 90, False),
]


def sample_page() -> tuple[bytes, list[dict]]:
    """PNG bytes (1 px per point) and the regions' ground truth boxes from the drawn words."""
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    page.draw_rect(pymupdf.Rect(30, 30, WIDTH - 30, 560), width=3)
    page.draw_rect(pymupdf.Rect(30, 600, WIDTH - 30, HEIGHT - 30), width=3)
    for x in range(40, WIDTH - 160, 14):  # hatched art under the caption and the SFX
        page.draw_line((x, 610), (x + 120, HEIGHT - 40), color=(0.55, 0.55, 0.55), width=2)
    truth = []
    for kind, area, text, size, bubble in REGIONS:
        rect = pymupdf.Rect(area)
        if bubble:
            page.draw_oval(rect, color=(0, 0, 0), fill=(1, 1, 1), width=3)
            rect = rect + (40, 30, -40, -30)
        page.insert_textbox(rect, text, fontsize=size, fontname="hebo", align=1)
        words = [pymupdf.Rect(w[:4]) for w in page.get_text("words", clip=rect)]
        box = words[0]
        for w in words[1:]:
            box |= w
        truth.append({"box": [round(v) for v in box], "kind": kind, "text": " ".join(text.split())})
    return page.get_pixmap(dpi=72).tobytes("png"), truth


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_books/manga")
    target.mkdir(parents=True, exist_ok=True)
    png, truth = sample_page()
    (target / "page-1.png").write_bytes(png)
    (target / "page-1.json").write_text(json.dumps({"boxes": truth}, indent=2))
    print(f"wrote {target / 'page-1.png'} with {len(truth)} regions")


if __name__ == "__main__":
    main()
