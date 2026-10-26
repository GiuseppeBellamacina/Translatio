"""Manga evaluation harness: metrics, sample pages, eval script."""

from collections import Counter

from app.services.manga.metrics import edit_distance, iou, match, rates, score_page


def test_iou_and_match():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0
    assert round(iou([0, 0, 10, 10], [5, 0, 15, 10]), 3) == round(50 / 150, 3)
    # the better pair wins, each box is used once, weak overlaps are not matches
    pred = [[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]]
    truth = [[1, 0, 11, 10], [50, 50, 60, 60]]
    assert match(pred, truth) == [(1, 0)]


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("スパイ", "スパイ") == 0


def test_score_page_counts_sfx_and_cer():
    truth = [
        {"box": [0, 0, 100, 50], "kind": "dialogue", "text": "HELLO  THERE"},
        {"box": [0, 100, 100, 150], "kind": "caption", "text": "Meanwhile"},
        {"box": [200, 0, 400, 200], "kind": "sfx", "text": "BAM"},
    ]
    pred = [
        {"box": [2, 0, 100, 50], "text": "HELLO\nTHERE!"},  # one extra char
        {"box": [200, 0, 400, 200], "text": "BAM"},  # SFX kept by mistake
    ]
    counts = score_page(pred, truth)
    r = rates(counts)
    assert (counts["hits"], counts["pred"], counts["truth"]) == (1, 2, 2)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["sfx_included"] == 1
    assert r["cer"] == 1 / len("HELLO THERE")
    # pages add up; no SFX and no OCR text means n/a, not zero
    total = counts + score_page([], truth[:1])
    assert total["truth"] == 3
    assert rates(Counter(pred=1, hits=1, truth=1))["sfx_included"] is None
