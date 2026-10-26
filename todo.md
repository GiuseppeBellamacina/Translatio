# To do

Future work, most of it on manga mode. Where things stand: [docs/manga-plan.md](docs/manga-plan.md), with the measured results in [docs/manga-eval.md](docs/manga-eval.md).

## Needs model calls or new data

- [ ] **End-to-end evaluation on real chapters** ([#186](https://github.com/GiuseppeBellamacina/Translatio/issues/186)).
  - Translate a whole chapter and score every page: readability, fit in the bubble, erasure, sound effects untouched, cost and time per page.
  - So far every number comes from one Italian chapter (Spy x Family ch. 1); measure Japanese and English sources too.
  - OpenRouter's free tier allows 50 requests a day without credits. To try a few pages cheaply, upload a CBZ of those pages and copy their `<file>.pages/` cache next to it.
- [ ] **manga-ocr on Japanese pages**: never measured; it needs a Japanese chapter with labelled ground truth (`scripts/manga_eval.py --bootstrap`, then correct the drafts by hand).
- [ ] **Other OCR engines**:
  - PaddleOCR, which needs its own framework (paddlepaddle);
  - Surya, after checking its weights licence;
  - Google Cloud Vision and Azure Read, which need keys.
  - Compare them on the same test set (`--ocr`).
- [ ] **More vision models**: gpt-4.1-mini and gpt-5-nano were not measured (the OpenAI credit ran out), nor cheap paid OpenRouter models.

## Known limits

- [ ] **Reading order inside a panel**: 0.94 on development pages and 0.91 on held-out ones. The cuts along the gutters are right; the errors are the order of bubbles within one panel and complex layouts (page 32: a big bubble opens a fake gutter once it is painted out). Options:
  - order by bubble tails or speaker position;
  - a panel detector without a non-commercial licence (Magi v2 has one).
- [ ] **Lettered sound effects with the local OCR**: only the vision OCR marks them (`[BAM]`), so easyocr lets "TOSS" or "SNIFF" through. Options:
  - a list of onomatopoeia per language plus short all-caps words outside dialogue;
  - a small classifier on the crops.
- [ ] **Vertical lettering** for Japanese and Chinese targets: CJK is lettered horizontally today, through MuPDF's fallback font.
- [ ] **Bubble shape**: the room is a rectangle, the bubble's inscribed width over its height. Use the bubble's real outline (a mask from the detector, or a distance transform of the clean page) and per-line widths.
- [ ] **Hyphenation** of long words in narrow bubbles (e.g. pyphen for the target language).
- [ ] **Erasing over gradients and screentone inside bubbles**: they go to LaMa and may leave smudges. Check on more pages; maybe a dedicated screentone fill.
- [ ] **OCR leftovers**: the vision OCR sometimes keeps a hyphen inside a word ("COMUN-QUE"), which a translation hides. Handle it if it shows in untranslated text.
- [ ] **Cost estimate**: unread pages are reported but the OCR itself is not priced. Add the vision model's per-page cost once the OCR model is chosen in the estimate.
- [ ] **Speakers → compendium**: knowing who says each line would fix gender agreement in dialogue. The plan's source, Magi v2, is non-commercial; look for an alternative.

## Improvements

- [ ] **Webtoons** (long vertical strips): split tall images into page-sized slices for detection, then stitch them back.
- [ ] **CPU-only machines**: the ML service falls back to CPU but was never tried there; measure it and document what to expect.
- [ ] **README screenshots of manga mode**: the test chapter is copyrighted. Use pages with an open licence, or synthetic ones.
- [ ] **Per-book reading direction**: manga are read right to left; add a switch for western comics (left to right), which today exists only as `--ltr` in the eval script.
- [ ] **Delete the merged `manga` branch on GitHub**: `git push origin --delete manga`.
