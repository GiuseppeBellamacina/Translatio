"""Reflow translated PDFs as continuous text (see app/services/pdf_reflow.py).

    docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm \
        -v /path/to/pdfs:/pdfs backend python scripts/reflow_pdf.py /pdfs /pdfs/reflow

Every PDF of the input folder (or the single file given) is written to the output
folder with the same name; the input files are left as they are.
"""

import sys
import time
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.pdf_reflow import reflow  # noqa: E402


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(source.glob("*.pdf")) if source.is_dir() else [source]
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        start = time.time()
        src = pymupdf.open(path)
        out = reflow(src)
        out.subset_fonts()
        out.save(target / path.name, garbage=4, deflate=True, use_objstms=1)
        print(f"{path.name}: {len(src)} -> {len(out)} pages ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
