"""
Continuous text: rebuild a translated PDF as one flow instead of page by page.

Typesetting each translation on its original page (``pdf_service.typeset``) keeps the
pages aligned with the original, but a longer language has to shrink, so sizes vary from
page to page and a page cut before an illustration or a scene break may stay half empty.
``reflow`` lays the text out again, the page count may change:

- body text flows across pages at the book's own size (the one of the pages that did not
  shrink), with uniform spacing, indent and alignment; a paragraph cut by a page end is
  joined again;
- a chapter (a heading at the top of a page, as text or drawn as an image) starts a page;
- full-page illustrations float to right after the page where the text reaches them;
- small images (headings drawn as images, ornaments) stay in the flow;
- the front matter (cover, plates, contents, credits) is kept as it is;
- footers ("12 | Pagina") are renumbered; the outline and the links of the contents page
  point to the new pages where the chapters start.
"""

import io
import re
import statistics
from collections import Counter

import pymupdf

from app.services.pdf_service import (
    _FONT_CSS,
    _PLACEHOLDER,
    _SENTENCE_END,
    PdfBlock,
    Style,
    _archive,
    _css_style,
    block_html,
    extract_page_blocks,
)

FOOTER = re.compile(r"P\s*a\s*g\s*(?:i\s*n\s*a|e)\b", re.I)  # "Pagina", or "Page" left untranslated
IMAGE_PAGE = 0.4  # share of a page covered by images that makes it an illustration
FRONT_TEXT = 600  # characters of the first page that is not front matter


def _plain(text: str) -> str:
    return _PLACEHOLDER.sub("", text).strip()


def _is_footer(block: PdfBlock, height: float) -> bool:
    return block.bbox[1] > height * 0.88 and bool(
        FOOTER.search(block.text) or _plain(block.text).isdigit()
    )


def _image_page(page: pymupdf.Page) -> bool:
    area = sum(pymupdf.Rect(i["bbox"]).get_area() for i in page.get_image_info())
    return area > IMAGE_PAGE * page.rect.get_area()


def _body_size(blocks: list[PdfBlock]) -> float | None:
    sizes = Counter()
    for b in blocks:
        if not b.heading:
            sizes[b.style.size] += len(b.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _clusters(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Small images of a text page, joined when they touch (a heading is several)."""
    out: list[pymupdf.Rect] = []
    for r in sorted((pymupdf.Rect(i["bbox"]) for i in page.get_image_info()), key=lambda r: r.y0):
        if out and r.y0 <= out[-1].y1 + 6:
            out[-1] |= r
        else:
            out.append(pymupdf.Rect(r))
    return out


def reflow(src: pymupdf.Document) -> pymupdf.Document:
    """A new document with the text of ``src`` reflowed (``src`` is left untouched)."""
    pages = [extract_page_blocks(p) for p in src]
    heights = [p.rect.height for p in src]
    bodies = [[b for b in bl if not _is_footer(b, h)] for bl, h in zip(pages, heights)]
    full = [i for i, p in enumerate(src) if not _image_page(p) and len(bodies[i]) >= 6]
    if not full:  # nothing that looks like running text: leave it as it is
        out = pymupdf.open()
        out.insert_pdf(src)
        return out

    sizes = Counter(s for s in map(_body_size, bodies) if s)
    common = [s for s, n in sizes.items() if n >= 5]
    body = max(common) if common else sizes.most_common(1)[0][0]  # pages that did not shrink

    page_rect = src[full[0]].rect
    footers = [b for i in full for b in pages[i] if _is_footer(b, heights[i])]
    wordings = Counter(re.sub(r"\d+", "#", b.text) for b in footers)
    footer = next(
        (b for b in footers if re.sub(r"\d+", "#", b.text) == wordings.most_common(1)[0][0]),
        None,
    )  # the most common one: a few may have been left untranslated ("Page")
    rules = [
        d["rect"] for d in src[full[0]].get_drawings() if d["rect"].y0 > page_rect.height * 0.88
    ]
    bottom = min([r.y0 for r in rules] + ([footer.bbox[1]] if footer else []), default=page_rect.y1)
    lows = [max(b.bbox[3] for b in bodies[i]) for i in full]
    y1 = min(statistics.quantiles(lows, n=20)[-1] if len(lows) > 1 else lows[0], bottom - 8)
    content = pymupdf.Rect(
        statistics.median(min(b.bbox[0] for b in bodies[i]) for i in full),
        statistics.median(min(b.bbox[1] for b in bodies[i]) for i in full),
        statistics.median(max(b.bbox[2] for b in bodies[i]) for i in full) + 1,
        y1,
    )
    flowing = [b for i in full for b in bodies[i] if not b.heading]
    gaps = [
        b.bbox[1] - a.bbox[3]
        for i in full
        for a, b in zip(bodies[i], bodies[i][1:])
        if not a.heading and not b.heading and 0 < b.bbox[1] - a.bbox[3] < 2 * body
    ]
    gap = round(statistics.median(gaps), 1) if gaps else round(body * 0.6, 1)
    line = round(statistics.median(b.line_height for b in flowing), 2)
    align = Counter(b.align for b in flowing if len(b.text) > 200).most_common(1)
    align = align[0][0] if align else "left"
    indents = [b.indent for b in flowing if b.indent > 0]
    indent = round(statistics.median(indents), 1) if indents else 0.0
    front = next((i for i, bl in enumerate(bodies) if sum(len(b.text) for b in bl) > FRONT_TEXT), 0)

    archive = _archive()

    def picture(page: pymupdf.Page, rect: pymupdf.Rect, name: str) -> str:
        pix = page.get_pixmap(clip=rect, dpi=300)
        archive.add((pix.tobytes("png"), name))
        return (
            f'<p style="margin-top: 0;"><img src="{name}" '
            f'style="width: {rect.width:.1f}px; height: {rect.height:.1f}px;"/></p>'
        )

    def paragraph(b: PdfBlock, factor: float) -> tuple[str, str]:
        scale = 1 / factor
        style = Style(
            b.style.family,
            round(b.style.size * scale, 1),
            b.style.bold,
            b.style.italic,
            b.style.color,
        )
        centered = b.align == "center"
        css = (
            f"{_css_style(style)} text-align: {'center' if centered else (b.align if b.heading else align)};"
            f" line-height: {b.line_height if b.heading else line:.2f};"
            f" text-indent: {0 if centered or b.heading else indent}px;"
        )
        return css, block_html(b, b.text, scale)

    # --- the flow: ("p", css, html, plain, heading) | ("pic", html) | ("img", page) | ("break", page)
    items: list[tuple] = []
    for i, page in enumerate(src):
        if i < front:
            continue
        if _image_page(page):
            items.append(("img", i))
            continue
        blocks = bodies[i]
        factor = (_body_size(blocks) or body) / body
        pics = _clusters(page)
        first_y = min((b.bbox[1] for b in blocks), default=page.rect.height)
        if (pics and pics[0].y0 < content.y0 + 40 and pics[0].y1 <= first_y + 2) or (
            blocks and blocks[0].heading
        ):
            items.append(("break", i))
        for k, b in enumerate(blocks):
            while pics and pics[0].y0 < b.bbox[1]:
                items.append(("pic", picture(page, pics.pop(0), f"p{i}-{len(items)}.png")))
            css, inner = paragraph(b, factor)
            text = _plain(b.text)
            last = items[-1] if items else None
            if (
                k == 0
                and last
                and last[0] == "p"
                and not last[4]
                and not b.heading
                and not _SENTENCE_END.search(last[3])
            ):  # a paragraph cut by the page end
                joint = "" if last[3].endswith("-") else " "
                items[-1] = (
                    "p",
                    last[1],
                    last[2].rstrip("-") + joint + inner,
                    last[3] + joint + text,
                    False,
                )
                continue
            items.append(("p", css, inner, text, b.heading))
        for rect in pics:
            items.append(("pic", picture(page, rect, f"p{i}-{len(items)}.png")))

    # --- placing it -----------------------------------------------------------------------
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)  # a Story draws on a writer's page device
    order: list[tuple[str, int]] = [("src", i) for i in range(front)]
    starts = {i: i for i in range(front)}  # source page -> new page where it starts
    pending: list[int] = []
    state = {"device": None, "y": content.y0, "pages": 0}

    def new_page():
        if state["device"] is not None:
            writer.end_page()
            for i in pending:
                starts[i] = len(order)
                order.append(("src", i))
            pending.clear()
        state["device"] = writer.begin_page(page_rect)
        order.append(("text", state["pages"]))
        state["pages"] += 1
        state["y"] = content.y0

    min_room = body * line * 1.5  # a line and its spacing

    def place(html: str):
        # a story given (almost) no room reports "done" without placing anything: its whole
        # text would be lost, so it always starts with at least a line of room
        if state["device"] is None or content.y1 - state["y"] < min_room:
            new_page()
        story = pymupdf.Story(html=html, user_css=_FONT_CSS, archive=archive)
        more = 1
        while more:
            rect = pymupdf.Rect(content.x0, state["y"], content.x1, content.y1)
            more, filled = story.place(rect)
            if filled[3] <= rect.y0 and more:  # nothing fitted: next page
                if state["y"] == content.y0:
                    raise ValueError("an element taller than a whole page")  # not a loop
                new_page()
                continue
            story.draw(state["device"])
            state["y"] = filled[3]
            if more:
                new_page()

    run: list[str] = []

    def flush():
        if run:
            place("".join(run))
            run.clear()

    for it in items:
        if it[0] == "img":
            flush()
            pending.append(it[1])
        elif it[0] == "break":
            flush()
            if state["device"] is None or state["y"] > content.y0:
                new_page()
            starts[it[1]] = len(order) - 1
        elif it[0] == "pic":
            run.append(it[1])
        else:
            top = 0 if state["y"] == content.y0 and not run else gap
            below = f" margin-bottom: {gap}px;" if it[4] else ""
            run.append(f'<p style="{it[1]} margin-top: {top}px;{below}">{it[2]}</p>')
    flush()
    if state["device"] is not None:
        writer.end_page()
    writer.close()
    for i in pending:
        starts[i] = len(order)
        order.append(("src", i))
    text_doc = pymupdf.open("pdf", buffer.getvalue())

    # --- assembling, footers, outline and contents links -----------------------------------
    out = pymupdf.open()
    for kind, i in order:
        out.insert_pdf(src if kind == "src" else text_doc, from_page=i, to_page=i, links=False)

    def moved(page: int) -> int:
        """New page of an old one: exact for chapter starts and kept pages, else the
        nearest known start before it."""
        known = [p for p in starts if p <= page]
        return starts[max(known)] if known else page

    for n, page in enumerate(out):
        kind, i = order[n]
        label_from = footer
        if kind == "src":
            own = [b for b in pages[i] if _is_footer(b, heights[i])]
            for b in own:  # the old number would be wrong now: same place, new number
                for r in b.span_rects:
                    page.add_redact_annot(r, fill=False)
            if own:
                page.apply_redactions(
                    images=pymupdf.PDF_REDACT_IMAGE_NONE,
                    graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                    text=pymupdf.PDF_REDACT_TEXT_REMOVE,
                )
            label_from = own[0] if own else None
            for link in src[i].get_links():
                if link.get("kind") == pymupdf.LINK_GOTO and link.get("page", -1) >= 0:
                    page.insert_link({**link, "page": moved(link["page"])})
        else:
            for r in rules:
                page.draw_rect(r, color=None, fill=(0, 0, 0), width=0)
        if label_from is not None and footer is not None:
            # the book's usual wording ("Pagina | 29" or "29 | Pagina"), at this page's place
            label = re.sub(r"\d+", str(n + 1), footer.text, count=1)
            box = pymupdf.Rect(label_from.bbox) + (-60, -2, 2, 4)
            css = f"{_css_style(footer.style)} text-align: right;"
            page.insert_htmlbox(
                box,
                f'<p style="{css}">{block_html(footer, label)}</p>',
                css=_FONT_CSS,
                archive=archive,
            )

    toc = src.get_toc(simple=True)
    if toc:
        out.set_toc([[level, title, moved(page - 1) + 1] for level, title, page in toc])
    out.set_metadata(src.metadata or {})
    return out
