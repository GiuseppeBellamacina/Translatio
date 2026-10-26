# Manga mode: measured results

Numbers from `backend/scripts/manga_eval.py` (see [the plan](manga-plan.md)).

## Test set

Spy x Family ch. 1, the official Italian edition (74 pages, 800×1200). The pages are copyrighted and stay outside git, in `data/manga-test/it/`.

13 pages are labelled by hand in `sxf1-dev/` and `sxf1-test/`, with:
- the reading order;
- sound effects lettered as text;
- false positives removed (signs, a chandelier);
- a transcription of 162 regions (de-hyphenated, as printed).

The 7 development pages (10, 14, 18, 21, 29, 48, 54) are used for tuning. The 6 held-out pages (13, 25, 32, 37, 44, 65) are only measured.

The labels start from the detector's drafts and missed boxes were not added, so recall is not meaningful here. A check of the remaining 61 pages found no missed text.

## Detection and order

The detector is comic-text-and-bubble-detector, on an RTX 4070 SUPER.

| | dev | held-out |
| --- | --- | --- |
| precision | 0.92 | 0.90 |
| order, row rule | 0.90 | 0.90 |
| order, panels + row rule | 0.94 | 0.91 |

Precision misses come from sound effects lettered as text ("TOSS", "DING DONG", "SNIFF") and from signs in the art.

## OCR (local engines, Italian text)

CER is the character error rate on matched regions, ignoring case and apostrophe style. Crops are enlarged 2× and words hyphenated at a line end are rejoined.

| engine | CER dev | CER held-out | ms/page (detection + OCR) | notes |
| --- | --- | --- | --- | --- |
| easyocr (it, en) | **0.08** | **0.09** | 400–850 | U read as L, lost dots, the odd first letter |
| doctr | 0.23 | 0.27 | 450–700 | weak on comic lettering |
| manga-ocr | 0.44 | 0.37 | 1100–1500 | Japanese-only model: for Japanese sources |

Before the 2× enlargement and the " - " hyphen joining, easyocr scored 0.13 / 0.12.

## OCR (vision LLMs, Italian text)

Each page's crops are enlarged 2× and numbered on sheets of at most 1600 px (`app/services/manga/vision.py`), with one structured call per sheet. A sheet the model miscounts is read again box by box.

| model | CER dev | CER held-out | $/page | s/page | notes |
| --- | --- | --- | --- | --- | --- |
| gpt-6-luna (OpenAI) | **0.01** | **0.01** | 0.0009–0.0012 | 18–24 | reasoning tokens make it slow |
| gpt-4o-mini (OpenAI) | 0.03 | 0.03 | 0.006–0.008 | 5–6 | images cost it many input tokens |
| nvidia/nemotron-3.5-super-vl:free (OpenRouter) | 0.01 | — | 0 | ~97 | free tier rate limits |

The OpenAI credit ran out before gpt-4.1-mini and gpt-5-nano could be measured. On a first try, gpt-4.1-mini looped up to its output limit on one sheet; the box-by-box fallback now covers that case.

**Choice**:
- a vision LLM is the accurate default, at about 1/10 of a cent per page;
- easyocr is the free, fast, offline option;
- the translator tolerates its errors better than a human reader would, but names and short exclamations suffer.

Not measured yet:
- PaddleOCR, which needs its own framework;
- Surya, whose weights licence is restricted;
- Google Vision and Azure, which need keys;
- Japanese pages, for which a chapter is needed.
