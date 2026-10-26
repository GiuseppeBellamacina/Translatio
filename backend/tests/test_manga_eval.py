"""Manga evaluation harness: metrics, sample pages, eval script."""

import base64
import json
from collections import Counter

import httpx
import pymupdf
import pytest

from app.core.config import settings
from app.services.manga.detect import detect
from app.services.manga.metrics import edit_distance, iou, match, rates, score_page
from app.services.manga.ocr import clean, read
from scripts import manga_eval
from scripts.make_sample_manga import HEIGHT, REGIONS, WIDTH, sample_page
from scripts.manga_eval import DETECTORS, evaluate
from tests.test_manga_regions import SAMPLE


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
    assert rates(total) == {"precision": 1, "recall": 1, "sfx_included": 0, "cer": 0, "order": 1}

    async def everything(image):
        return truth

    _, total = await evaluate(tmp_path, everything)
    assert rates(total)["sfx_included"] == 1 and rates(total)["precision"] == 0.75

    rows, _ = await evaluate(tmp_path, everything, draft=everything)
    assert json.loads((tmp_path / "page-2.json").read_text())["boxes"] == truth
    assert [name for name, _ in rows] == ["page-10.png"]  # natural order, draft not scored
    rows, _ = await evaluate(tmp_path, everything)
    assert [name for name, _ in rows] == ["page-2.png", "page-10.png"]


async def test_detect_client(monkeypatch):
    regions = [{"box": [1, 2, 3, 4], "label": "text_free", "score": 0.9}]

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/detect" and request.content == b"png"
        return httpx.Response(200, json={"width": 8, "height": 8, "regions": regions})

    monkeypatch.setattr(settings, "manga_ml_url", "")
    with pytest.raises(RuntimeError, match="TRANSLATIO_MANGA_ML_URL"):
        await detect(b"png")
    monkeypatch.setattr(settings, "manga_ml_url", "http://ml:8100")
    assert await detect(b"png", transport=httpx.MockTransport(reply)) == regions


def test_order_score():
    truth = [{"box": [0, i * 100, 50, i * 100 + 50], "kind": "dialogue"} for i in range(3)]
    assert rates(score_page(truth, truth))["order"] == 1
    assert rates(score_page(truth[::-1], truth))["order"] == 0
    assert rates(score_page([truth[1], truth[0], truth[2]], truth))["order"] == 0.5


async def test_drafts_list_every_region(tmp_path, monkeypatch):
    async def detected(image: bytes) -> list[dict]:
        return SAMPLE

    monkeypatch.setattr(manga_eval, "ml_detect", detected)
    (tmp_path / "page-1.png").write_bytes(sample_page()[0])
    rows, _ = await evaluate(tmp_path, DETECTORS["regions"], draft=manga_eval._all_regions)
    assert rows == []  # drafted, not scored
    boxes = json.loads((tmp_path / "page-1.json").read_text())["boxes"]
    assert [b["kind"] for b in boxes] == ["dialogue", "dialogue", "caption", "caption"]
    check = pymupdf.Pixmap(str(tmp_path / "check" / "page-1.png"))
    assert (check.width, check.height) == (WIDTH, HEIGHT)


def test_ocr_text_is_cleaned():
    assert clean("PROMES-\nSO") == "PROMESSO"
    assert clean("RIPOR - TARLA IN\nTEMPO") == "RIPORTARLA IN TEMPO"
    assert clean("Forger-sama, ＢＡＭ！") == "Forger-sama, BAM!"
    # case and apostrophe style do not count as OCR errors
    pred = [{"box": [0, 0, 10, 10], "text": "Perche’ sei"}]
    truth = [{"box": [0, 0, 10, 10], "kind": "dialogue", "text": "PERCHE' SEI"}]
    assert rates(score_page(pred, truth))["cer"] == 0


async def test_ocr_client(monkeypatch):
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.params["engine"] == "easyocr" and request.url.params["lang"] == "it"
        assert base64.b64decode(body["image"]) == b"png" and body["boxes"] == [[1, 2, 3, 4]]
        return httpx.Response(200, json={"texts": ["CIAO-\nNE"]})

    monkeypatch.setattr(settings, "manga_ml_url", "http://ml:8100")
    texts = await read(b"png", [[1, 2, 3, 4]], "easyocr", "it", httpx.MockTransport(reply))
    assert texts == ["CIAONE"]
