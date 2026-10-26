"""Manga text regions: bubbles, kinds, SFX filter, reading order."""

from app.services.manga.regions import regions, text_regions

# what the detector returned on scripts/make_sample_manga.py's page
SAMPLE = [
    {"box": [434.7, 134.3, 745.5, 323.9], "label": "bubble", "score": 0.982},
    {"box": [54.2, 74.2, 365.5, 284.4], "label": "bubble", "score": 0.982},
    {"box": [503.2, 167.7, 675.9, 267.5], "label": "text_bubble", "score": 0.95},
    {"box": [148.3, 108.6, 271.0, 183.5], "label": "text_bubble", "score": 0.923},
    {"box": [478.3, 808.7, 733.1, 920.7], "label": "text_free", "score": 0.836},
    {"box": [127.3, 696.9, 370.2, 730.6], "label": "text_free", "score": 0.804},
]


def test_bubbles_and_kinds():
    found = {tuple(r["box"]): r for r in regions(SAMPLE)}
    assert len(found) == 4
    assert found[(503.2, 167.7, 675.9, 267.5)]["bubble"] == SAMPLE[0]["box"]
    assert found[(148.3, 108.6, 271.0, 183.5)]["bubble"] == SAMPLE[1]["box"]
    assert found[(127.3, 696.9, 370.2, 730.6)] == {
        "box": [127.3, 696.9, 370.2, 730.6],
        "kind": "caption",
        "bubble": None,
        "score": 0.804,
    }
    # manga order by default: the right bubble first, although its text starts lower
    order = [r["box"] for r in text_regions(SAMPLE)]
    assert order == [s["box"] for s in SAMPLE[2:4] + [SAMPLE[5], SAMPLE[4]]]
    # a text box mostly outside every bubble has none
    loose = [SAMPLE[0], {"box": [300, 200, 500, 300], "label": "text_bubble", "score": 0.5}]
    assert regions(loose)[0]["bubble"] is None


def test_free_text_is_a_caption():
    # sizes do not tell sound effects from narration (real pages: big captions, small SFX);
    # the synthetic "BAM!" is lettered like text, so it stays until OCR can recognise it
    assert [r["kind"] for r in regions(SAMPLE)] == ["dialogue", "dialogue", "caption", "caption"]
    assert text_regions([{**SAMPLE[4], "label": "text_bubble"}])[0]["kind"] == "dialogue"


def test_reading_order():
    def box(x0, y0, x1, y1):
        return {"box": [x0, y0, x1, y1], "label": "text_free", "score": 1}

    # two panels side by side (one taller), then a wide one below
    page = [box(0, 300, 300, 360), box(400, 10, 700, 80), box(20, 20, 300, 100)]
    page += [box(410, 120, 700, 200), box(100, 140, 380, 210)]
    order = lambda rtl: [r["box"][:2] for r in regions(page, rtl)]  # noqa: E731
    assert order(False) == [[20, 20], [400, 10], [100, 140], [410, 120], [0, 300]]
    assert order(True) == [[400, 10], [20, 20], [410, 120], [100, 140], [0, 300]]


def test_merged_boxes_are_dropped():
    def text(*box):
        return {"box": list(box), "label": "text_free", "score": 1}

    # a box around two captions (Spy x Family p. 21), an exact duplicate, a box with a speck
    page = [text(95, 45, 285, 285), text(100, 50, 280, 145), text(95, 180, 265, 280)]
    page += [text(500, 50, 600, 100), text(500, 50, 600, 100)]
    page += [text(500, 500, 700, 700), text(600, 600, 610, 610)]
    kept = sorted(r["box"] for r in regions(page))
    assert kept == sorted(
        [[100, 50, 280, 145], [95, 180, 265, 280], [500, 50, 600, 100]]
        + [[500, 500, 700, 700], [600, 600, 610, 610]]
    )
