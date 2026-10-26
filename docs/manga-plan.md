# Plan: Manga mode (OCR + bubble-aware typesetting), on a dedicated `manga` branch

## Context
Translatio translates text-layer EPUB/PDF. The user wants a **manga mode**:
- detect text boxes (in bubbles, captions, and text drawn over the art);
- OCR them;
- translate with the existing pipeline (glossary, compendium, doubts, review);
- erase the original and re-typeset the translation in place, recomputing font size and line breaks to fit.

**Requirements:**
- Onomatopoeia/SFX are excluded.
- OCR must be evaluated on both paid and free local models, with tests and measured results.
- **Sources:** Japanese raw and English.
- **Inputs:** CBZ/ZIP, PDF of images, and folders of images, written out in the same format.
- **Hardware:** an NVIDIA GPU is available, so local models run in Docker with GPU passthrough.

**Deliverables of the planning step** (done):
- a GitHub issue (epic) with the roadmap;
- a new long-lived branch `manga` from `main`;
- the plan committed there as `docs/manga-plan.md`.

The roadmap is tracked in epic #143: each item is a small issue with a feature branch from `manga`, merged into `manga` by a PR.

## Research summary
All sources, with links and what we take from each: [manga-sources.md](manga-sources.md).

| Stage | Local / free | Paid / API | Notes |
| --- | --- | --- | --- |
| Text + bubble detection | **comic-text-and-bubble-detector** (ogkalu, RT-DETR, ~11k manga/webtoon/comic images; separates bubbles from free text); **comic-text-detector** (used by manga-image-translator and BallonsTranslator; pixel text masks); **Magi v2** (Oxford, CVPR/ACCV 2024: text boxes, reading order, speaker links, keeps "dialogue-essential" text) | — | Magi's speaker links can feed the compendium (gender agreement) |
| OCR (Japanese) | **manga-ocr** (vertical text, furigana, text over images, multi-line bubbles in one pass); PaddleOCR / PaddleOCR-VL | Google Cloud Vision, Azure AI Vision Read; vision LLMs through the existing providers (GPT/Claude/Gemini via OpenRouter) | vendor numbers: plain PaddleOCR is weak on manga, the cloud APIs are strong; the MangaVQA paper reports GPT-4o/Gemini near zero on in-page manga OCR, so test it |
| OCR (English) | PaddleOCR (PP-OCRv5), Surya, docTR | the same cloud APIs and vision LLMs | comic fonts and all-caps text; must be measured |
| Erasing | white/plain bubble: fill with the sampled background; text over art: **LaMa** (anime/manga big-lama weights) or AOT | — | as manga-image-translator, BallonsTranslator and Koharu do |
| Typesetting | own: fit text to the bubble's shape (inscribed area), binary search on font size, hyphenation, centered; vertical JP → horizontal IT | — | the hard part; reuse what we do for PDF regions |

**Reference projects** for design and weights (not dependencies):
- manga-image-translator: GPL, the pipeline and model zoo;
- comic-translate: Apache, the detector and OCR choice;
- BallonsTranslator: GPL, the UI ideas;
- Koharu: MIT/Apache, Rust.

Check each model's licence before bundling it.

## Fit with the current architecture (reuse, do not fork)
- **New `Document` adapter** `MangaDocument` (`app/services/manga/document.py`), picked by `open_document()` for `.cbz/.zip`, image folders, and PDFs whose pages have no text layer and are images (today rejected with `ScannedPdfError`; with manga mode on they go here).
- **Unit and block mapping**:
  - a unit is a page;
  - a block is one detected text region in reading order;
  - translations stay `list[str | None]` per unit, so `translate_document`, grouping (`unit_groups`), resume checkpoints, `JobStore`, glossary, compendium and doubt review all work unchanged;
  - `lead_context` (the previous page) keeps cross-page dialogue coherent.
- **Extraction is the expensive step** (detection + OCR), run once per page and cached on disk (`data/manga/<book>/page-N.json`: boxes, masks, OCR text, kind), so resume or re-translation never re-OCRs.
- **`save()`** = erase + typeset per page, then pack back as CBZ / PDF / folder.
- **Run settings**: the `RunConfig` switch `manga` plus choices `ocr_engine` and `detector`, shown in `RunSettingsFields` like the feature switches. The OCR engine reuses the provider registry (`llm_factory`, `model_catalog`) for vision-LLM OCR. The cloud OCR keys go in `.env` / `config.yaml` like today.
- **Dependencies**: heavy ML (torch, the detector, manga-ocr, LaMa) goes in a separate Docker target or optional service (`backend` image stays slim): `docker-compose.manga.yml` with GPU (`deploy.resources.reservations.devices` nvidia). It exposes a small internal HTTP API (`/detect`, `/ocr`, `/inpaint`) that the backend calls. This keeps the main image and tests light, and the models swappable.
- **SFX exclusion**:
  - classes from the detector (bubble text vs free text) plus Magi's "essential text" flag;
  - a heuristic fallback (very large or stylised glyphs, rotated, outside bubbles, short repeated kana or onomatopoeic English such as "BAM", "DOKI", "ZZZ");
  - excluded regions stay untouched (blocks with translation `None`).

## Work items (each an issue + PR into `manga`)
1. **Branch, epic, plan doc.** `git checkout -b manga main`; commit `docs/manga-plan.md` (this plan, as the shared reference); push; open the epic issue with the checklist below.
2. **Evaluation harness first.**
   - **Test set**: 30–40 pages, half JP raw and half EN. The user provides them; we store only their paths and labels, outside git. The pages should mix white bubbles, text over art, vertical JP, all-caps EN, SFX and captions.
   - **Ground truth**: boxes, text, kind (dialogue / caption / SFX), made with a small labelling script (or bootstrapped from a strong engine and corrected by hand).
   - **Script**: `scripts/manga_eval.py` reports, per engine, detection precision/recall (IoU ≥ 0.5), SFX false-include rate, OCR CER/NLS, seconds per page and cost per page.
3. **Detection.**
   - **Service**: the ML service with comic-text-and-bubble-detector (and Magi v2 as the comparison).
   - **Output**: boxes, bubble masks, kind and reading order (right to left for manga in any language, as published editions keep the layout; left to right for western comics), with SFX filtering.
   - **Gate**: evaluate before picking the default.
4. **OCR engines behind one interface** (`ocr(image_crop, lang) -> text`):
   - local: manga-ocr (JP), PaddleOCR / PaddleOCR-VL (JP + EN), Surya (EN);
   - paid: Google Cloud Vision, Azure Read, and a vision LLM (cheap model via OpenRouter/OpenAI, with a strict prompt "transcribe exactly, no translation").
   - Run the harness and publish a comparison table: accuracy, speed, cost. Pick defaults per language, and offer the others as a run setting.
5. **`MangaDocument` + extraction cache + translation through the existing pipeline.**
   - JP vertical lines joined without spaces, EN line breaks unwrapped;
   - the translator prompt gets a short "comic dialogue, keep it short to fit a bubble" style hint;
   - Magi speakers become compendium hints (so a female speaker's lines agree).
6. **Erasing.**
   - bubble interiors: fill with the median background of the bubble mask;
   - text over art: LaMa (manga weights), only on the text mask, dilated a few px.
   - Compare against originals visually.
7. **Typesetting.**
   - fit horizontal text into the bubble's inscribed area (distance transform of the mask, or an ellipse fit);
   - binary-search the font size between bounds tied to the original glyph height;
   - Italian hyphenation (pyphen), centered lines, outline/stroke for text over art;
   - a comic font with an open licence (Comic Neue / Anime Ace-like OFL font);
   - if the text still does not fit: shrink to a floor, then flag a doubt bookmark ("text too long for its bubble") so the doubt review can shorten it.
8. **Output and UI.**
   - CBZ / PDF / folder writers;
   - preview side by side (reuse `PreviewPage` with page images);
   - run switches; the upload accepts `.cbz/.zip` and folders (zip upload).
9. **End-to-end evaluation on real chapters** (JP and EN): a human score sheet on readability, fit, erasure and SFX untouched, plus cost and time per page. Then decide what merges to `main`.

## Critical files
- **New**: `app/services/manga/{document,extract,ocr,inpaint,typeset,pack}.py`, `ml/` (GPU service: Dockerfile, FastAPI app, model loaders), `docker-compose.manga.yml`, `scripts/manga_eval.py`, `docs/manga-plan.md`.
- **Touched**: `app/services/documents.py` (`open_document`), `app/api/routers/books.py` (upload types), `app/jobs/models.py` (`RunConfig`), `app/core/config.py` (OCR keys and the ML service URL), `frontend/src/lib/api.ts` (`FEATURES`), `components/RunSettingsFields.tsx`, `pages/UploadPage.tsx`, `pages/PreviewPage.tsx`.
- **Reused as is**: `translation_service.translate_document`, `agents/pipeline.py`, `knowledge/*`, `job_store.py`, `jobs/*`, `structured_chain`, the provider registry.

## Verification
- **Now**:
  - the issue exists with the roadmap;
  - `git branch -a` shows `manga` pushed;
  - `docs/manga-plan.md` is committed on it;
  - `main` is untouched.
- **Each later item**:
  - unit tests with small synthetic pages (generated bubbles + text, deterministic fake OCR/detector like `FakeChatModel`);
  - `scripts/manga_eval.py` numbers recorded in the PR;
  - before/after page images in the PR;
  - the full suite stays green;
  - cheap or free models only for live tests.
