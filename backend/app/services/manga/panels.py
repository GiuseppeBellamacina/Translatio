"""Panels of a manga page in reading order, from the white gutters between them.

A recursive cut: a block splits at full-width white rows first (pieces top to bottom), otherwise
at full-height white columns (right to left for manga); a block that does not split is a panel.
Detected bubbles and text are painted out first, since bubbles often cross a gutter.

shortcut: gutters must be white and straight, so dark pages and slanted gutters stay one panel
(the row rule inside regions still orders them); painting out a big bubble can open a fake gutter
inside its panel, which then splits in two (still read in a sensible order).
"""

import numpy as np
import pymupdf

LIGHT = 200  # grey level from which a pixel counts as paper
GUTTER_INK = 0.004  # a row or column with at most this share of ink is gutter
MIN_PANEL = 0.005  # smaller pieces (stray lines, specks) are not panels
MIN_GAP = (
    0.002  # a gutter is at least this share of the page's longer side (vertical ones are thin)
)


def _gray(image: bytes) -> np.ndarray:
    pix = pymupdf.Pixmap(image)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.n != 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def _pieces(ink_share: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Runs along an axis that hold ink, split by gutters at least ``min_gap`` long."""
    pieces: list[tuple[int, int]] = []
    start = end = gap = 0
    inside = False
    for i, inked in enumerate(ink_share > GUTTER_INK):
        if inked:
            if not inside:
                start, inside = i, True
            end, gap = i + 1, 0
        elif inside:
            gap += 1
            if gap >= min_gap:
                pieces.append((start, end))
                inside = False
    if inside:
        pieces.append((start, end))
    return pieces


def panels(image: bytes, covered: list[list[float]], right_to_left: bool = True) -> list[list[int]]:
    """Panel boxes ``[x0, y0, x1, y1]`` in reading order; ``covered``: boxes painted out."""
    ink = _gray(image) < LIGHT
    for x0, y0, x1, y1 in covered:
        ink[max(int(y0), 0) : int(y1) + 1, max(int(x0), 0) : int(x1) + 1] = False
    min_gap = max(2, round(MIN_GAP * max(ink.shape)))
    found: list[list[int]] = []

    def cut(x0: int, y0: int, x1: int, y1: int) -> None:
        block = ink[y0:y1, x0:x1]
        rows = _pieces(block.mean(axis=1), min_gap)
        if len(rows) > 1:
            for a, b in rows:
                cut(x0, y0 + a, x1, y0 + b)
            return
        cols = _pieces(block.mean(axis=0), min_gap)
        if len(cols) > 1:
            for a, b in reversed(cols) if right_to_left else cols:
                cut(x0 + a, y0, x0 + b, y1)
            return
        if (
            rows
            and cols
            and ((cols[0][1] - cols[0][0]) * (rows[0][1] - rows[0][0]) >= MIN_PANEL * ink.size)
        ):
            found.append([x0 + cols[0][0], y0 + rows[0][0], x0 + cols[0][1], y0 + rows[0][1]])

    cut(0, 0, ink.shape[1], ink.shape[0])
    return found
