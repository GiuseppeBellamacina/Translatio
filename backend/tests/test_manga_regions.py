"""Manga text regions: bubbles, kinds, SFX filter, reading order."""

from app.services.manga.regions import regions, text_regions

# what the detector returned on scripts/make_sample_manga.py's page
SAMPLE = [
    {"box": [54.6, 74.5, 365.1, 284.4], "label": "bubble", "score": 0.982},
    {"box": [434.4, 133.6, 745.7, 324.6], "label": "bubble", "score": 0.981},
    {"box": [122.9, 107.6, 296.4, 209.5], "label": "text_bubble", "score": 0.948},
    {"box": [528.4, 168.5, 651.2, 242.8], "label": "text_bubble", "score": 0.927},
    {"box": [478.0, 809.4, 734.9, 921.8], "label": "text_free", "score": 0.832},
    {"box": [127.4, 696.7, 370.0, 730.7], "label": "text_free", "score": 0.802},
]


def test_bubbles_and_kinds():
    found = {tuple(r["box"]): r for r in regions(SAMPLE)}
    assert len(found) == 4
    assert found[(122.9, 107.6, 296.4, 209.5)]["bubble"] == SAMPLE[0]["box"]
    assert found[(528.4, 168.5, 651.2, 242.8)]["bubble"] == SAMPLE[1]["box"]
    assert found[(127.4, 696.7, 370.0, 730.7)] == {
        "box": [127.4, 696.7, 370.0, 730.7],
        "kind": "caption",
        "bubble": None,
        "score": 0.802,
    }
    # a text box mostly outside every bubble has none
    loose = [SAMPLE[0], {"box": [300, 200, 500, 300], "label": "text_bubble", "score": 0.5}]
    assert regions(loose)[0]["bubble"] is None


def test_sound_effects_are_left_alone():
    kinds = [r["kind"] for r in regions(SAMPLE)]
    assert kinds.count("sfx") == 1 and kinds.count("caption") == 1
    kept = text_regions(SAMPLE)
    assert [r["box"] for r in kept if r["kind"] == "caption"] == [SAMPLE[5]["box"]]
    assert all(r["kind"] != "sfx" for r in kept)
    # without bubble text on the page there is no scale to compare with: free text is kept
    assert [r["kind"] for r in regions(SAMPLE[4:])] == ["caption", "caption"]


def test_reading_order():
    def box(x0, y0, x1, y1):
        return {"box": [x0, y0, x1, y1], "label": "text_free", "score": 1}

    # two panels side by side (one taller), then a wide one below
    page = [box(0, 300, 300, 360), box(400, 10, 700, 80), box(20, 20, 300, 100)]
    page += [box(410, 120, 700, 200), box(450, 140, 690, 210)]
    order = lambda rtl: [r["box"][:2] for r in regions(page, rtl)]  # noqa: E731
    assert order(False) == [[20, 20], [400, 10], [410, 120], [450, 140], [0, 300]]
    assert order(True) == [[400, 10], [20, 20], [450, 140], [410, 120], [0, 300]]
