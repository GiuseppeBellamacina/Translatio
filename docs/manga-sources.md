# Manga mode: research sources

The sources behind [the manga plan](manga-plan.md), grouped by stage. Each line says what we take from the source. Reference projects are for design and model choice, not dependencies: check each licence before bundling any code or weights.

## Overviews and comparisons
- [OpenL – Best Manga Translator 2026](https://blog.openl.io/best-manga-translator-2026/) — the usual pipeline (detect → OCR → translate → clean → typeset) and where tools fail.
- [Kitsu – Manga Translation Tools Compared 2026](https://www.kitsutl.com/blog/manga-translation-tools-comparison-2026) — feature comparison of the current tools.
- [Local AI Master – Translate Manga Locally](https://localaimaster.com/blog/translate-manga-locally) — a fully local setup on a consumer GPU.
- [aimangatranslator – Scanlation toolchain 2026](https://aimangatranslator.app/blog/scanlation-tools-workflow-2026) — how scanlation groups split the work (clean, typeset, QA).

## Reference projects
- [comic-translate](https://github.com/ogkalu2/comic-translate) — Apache; its RT-DETR text/bubble detector (comic-text-and-bubble-detector) and OCR choices per language.
- [Koharu](https://github.com/koharu-rs/koharu) — MIT/Apache, Rust; detection, OCR, inpainting and rendering in one local app.
- [MangaQuick](https://github.com/DCY1117/MangaQuick) — a compact end-to-end pipeline for comparison.
- [pcleaner](https://pypi.org/project/pcleaner/2.10.0) — text removal with masks; how to tell plain bubble backgrounds from art.
- [lama-cleaner](https://www.pypi.org/project/lama-cleaner/0.29.0/) — LaMa inpainting, including the anime/manga weights, for text over art.
- [manga_globes on Replicate](https://replicate.com/eiby777/manga_globes/readme) — a hosted bubble detector, a paid option to compare.

## OCR
- [manga-ocr on PyPI](https://pypi.org/project/manga-ocr/0.1.10) — the local reference for Japanese: vertical text, furigana, multi-line bubbles in one pass.
- [aimangatranslator – Manga OCR Engine Comparison](https://aimangatranslator.app/blog/manga-ocr-engine-comparison) — engines compared on manga pages (vendor numbers, to verify).
- [Sakasegawa – 23 OCR models on handwritten Japanese](https://nyosegawa.com/en/posts/japanese-handwriting-ocr-comparison/) — accuracy of 23 OCR models on hard Japanese text.
- [Best Open-Source OCR Models 2026](https://www.aimadetools.com/blog/best-open-source-ocr-models-2026/) — the English candidates: PaddleOCR, Surya, docTR.
- [MangaVQA / MangaOCR](https://arxiv.org/abs/2505.20298v3) — benchmark where general vision LLMs score poorly on in-page manga OCR: measure them before trusting them.

## Detection, reading order, speakers
- [Magi / The Manga Whisperer, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sachdeva_The_Manga_Whisperer_Automatically_Generating_Transcriptions_for_Comics_CVPR_2024_paper.html) — panels, text boxes, reading order and speaker links.
- [Magiv2, ACCV 2024](https://arxiv.org/html/2408.00298v1) — adds character names and "essential text" (helps leave out SFX).
- [Unconstrained Text Detection in Manga](https://arxiv.org/pdf/2010.03997) — text over art and stylised text; the dataset and the difficulty of SFX.
- [Survey on comics understanding](https://arxiv.org/pdf/2409.09502) — overview of the tasks and datasets (Manga109, COMICS).
- [Indonesian Manhwa Translation pipeline](https://arxiv.org/pdf/2501.01629) — a full pipeline paper with per-stage evaluation, a model for our harness.
