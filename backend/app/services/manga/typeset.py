"""Lettering of translated manga regions: each text fitted, centred, in its bubble's room.

The font is Comic Neue Bold (SIL OFL 1.1, app/data/fonts). The size starts from the original
letters (box area / characters) and shrinks until the text fits; text over the art gets a
white outline so it reads on any background.
"""

import html
from pathlib import Path

import pymupdf

FONTS = Path(__file__).resolve().parents[2] / "data" / "fonts"
INSET = 0.72  # share of a bubble's box inside its oval: a little under 1/sqrt(2), for margins
SIZES = (9.0, 36.0)  # px: the font size range
OUTLINE = 1.5  # px of white around text over the art
CSS = (
    "@font-face {font-family: comic; src: url(ComicNeue-Bold.ttf);}"
    "p {font-family: comic; text-align: center; margin: 0; line-height: 1.1;}"
)
_ARCHIVE = pymupdf.Archive(str(FONTS))


def room(region: dict) -> pymupdf.Rect:
    """Where the translation goes: in a bubble, its inscribed width over the height of the
    inscribed area and the original text box; otherwise the text box, with a small margin."""
    box = pymupdf.Rect(region["box"])
    if not region.get("bubble"):
        return box + (-4, -4, 4, 4)
    bubble = pymupdf.Rect(region["bubble"])
    dx, dy = bubble.width * (1 - INSET) / 2, bubble.height * (1 - INSET) / 2
    inner = bubble + (dx, dy, -dx, -dy)
    # an oval narrows towards its top and bottom: the width stays the inscribed one
    return pymupdf.Rect(inner.x0, min(inner.y0, box.y0), inner.x1, max(inner.y1, box.y1)) & bubble


def font_size(region: dict) -> float:
    """About the original letters' size: the box's area shared by its characters."""
    box = pymupdf.Rect(region["box"])
    size = 1.2 * (box.get_area() / max(len(region["text"]), 1)) ** 0.5
    return min(max(size, SIZES[0]), SIZES[1])


def letter(page: pymupdf.Page, region: dict, text: str) -> None:
    """Write ``text`` for ``region`` on ``page``, fitted and centred in its room."""
    if region["text"].isupper():
        text = text.upper()  # the lettering's own case
    rect = room(region)
    body = f'<p style="font-size:{font_size(region):.1f}px">{html.escape(text)}</p>'
    scratch = pymupdf.open()  # measure first: the unused height centres the text
    probe = scratch.new_page(width=page.rect.width, height=page.rect.height)
    spare, _ = probe.insert_htmlbox(rect, body, css=CSS, archive=_ARCHIVE, scale_low=0)
    rect += (0, max(spare, 0) / 2, 0, max(spare, 0) / 2)  # vertically centred
    if not region.get("bubble"):
        halo = body.replace("<p ", '<p class="halo" ')
        css = CSS + "p.halo {color: white;}"
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
            shift = (dx * OUTLINE, dy * OUTLINE, dx * OUTLINE, dy * OUTLINE)
            page.insert_htmlbox(rect + shift, halo, css=css, archive=_ARCHIVE, scale_low=0)
    page.insert_htmlbox(rect, body, css=CSS, archive=_ARCHIVE, scale_low=0)
