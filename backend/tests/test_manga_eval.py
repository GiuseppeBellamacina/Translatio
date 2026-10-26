"""Manga evaluation harness: metrics, sample pages, eval script."""

import json
from collections import Counter

import pymupdf

from app.services.manga.metrics import edit_distance, iou, match, rates, score_page
from scripts.make_sample_manga import HEIGHT, REGIONS, WIDTH, sample_page
from scripts.manga_eval import DETECTORS, evaluate


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


def test_sample_page_has_ground_truth():
    png, truth = sample_page()
    pix = pymupdf.Pixmap(png)
    assert (pix.width, pix.height) == (WIDTH, HEIGHT)
    assert [r["kind"] for r in truth] == ["dialogue", "dialogue", "caption", "sfx"]
    for region, (_, area, text, *_) in zip(truth, REGIONS, strict=True):
        assert pymupdf.Rect(area).contains(pymupdf.Rect(region["box"]))
        assert region["text"] == " ".join(text.split())


async def test_eval_scores_and_bootstraps(tmp_path):
    png, truth = sample_page()
    for name in ("page-2", "page-10"):
        (tmp_path / f"{name}.png").write_bytes(png)
    (tmp_path / "page-10.json").write_text(json.dumps({"boxes": truth}))

    rows, total = await evaluate(tmp_path, DETECTORS["truth"])
    assert [name for name, _ in rows] == ["page-10.png"]  # no ground truth, not scored
    assert rates(total) == {"precision": 1, "recall": 1, "sfx_included": 0, "cer": 0}

    async def everything(image):
        return truth

    _, total = await evaluate(tmp_path, everything)
    assert rates(total)["sfx_included"] == 1 and rates(total)["precision"] == 0.75

    rows, _ = await evaluate(tmp_path, everything, bootstrap=True)
    assert json.loads((tmp_path / "page-2.json").read_text())["boxes"] == truth
    assert [name for name, _ in rows] == ["page-10.png"]  # natural order, draft not scored
    rows, _ = await evaluate(tmp_path, everything)
    assert [name for name, _ in rows] == ["page-2.png", "page-10.png"]
