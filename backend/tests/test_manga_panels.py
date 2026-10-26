"""Manga panels from the gutters, and regions read panel by panel."""

import pymupdf

from app.services.manga.panels import panels
from app.services.manga.regions import regions
from scripts.make_sample_manga import sample_page

# a tall panel on the right, two stacked panels on the left (gutters 10 px)
RIGHT, TOP_LEFT, BOTTOM_LEFT = (310, 20, 580, 780), (20, 20, 300, 390), (20, 400, 300, 780)


def _page(bubble: tuple | None = None) -> bytes:
    page = pymupdf.open().new_page(width=600, height=800)
    for rect in (RIGHT, TOP_LEFT, BOTTOM_LEFT):
        page.draw_rect(rect, width=2)
        for x in range(int(rect[0]) + 10, int(rect[2]) - 10, 25):  # some art in each panel
            page.draw_line((x, rect[1] + 20), (x + 8, rect[3] - 20), width=1)
    if bubble:
        page.draw_oval(bubble, color=(0, 0, 0), fill=(1, 1, 1), width=2)
    return page.get_pixmap(dpi=72).tobytes("png")


def _near(found, *expected):
    assert len(found) == len(expected)
    for got, want in zip(found, expected, strict=True):
        assert all(abs(g - w) <= 3 for g, w in zip(got, want, strict=True)), (got, want)


def test_panels_in_reading_order():
    _near(panels(_page(), []), RIGHT, TOP_LEFT, BOTTOM_LEFT)
    _near(panels(_page(), [], right_to_left=False), TOP_LEFT, BOTTOM_LEFT, RIGHT)
    assert len(panels(sample_page()[0], [])) == 2  # the sample page's two stacked panels


def test_a_bubble_across_a_gutter_is_painted_out():
    bubble = (60, 340, 260, 450)  # over the gutter between the two left panels
    assert len(panels(_page(bubble), [])) == 2  # the bubble joins them
    _near(panels(_page(bubble), [list(bubble)]), RIGHT, TOP_LEFT, BOTTOM_LEFT)


def test_regions_are_read_panel_by_panel():
    def text(*box):
        return {"box": list(box), "label": "text_free", "score": 1}

    # the row rule alone would read the top-left text before the lower right one
    page = [text(100, 50, 200, 100), text(400, 300, 500, 350), text(100, 500, 200, 550)]
    order = [r["box"][:2] for r in regions(page, panels=[RIGHT, TOP_LEFT, BOTTOM_LEFT])]
    assert order == [[400, 300], [100, 50], [100, 500]]
    assert [r["box"][:2] for r in regions(page)] == [[100, 50], [400, 300], [100, 500]]
