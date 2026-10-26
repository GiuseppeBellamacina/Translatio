"""Lettering of translated manga regions."""

import pymupdf

from app.services.manga.typeset import fits, letter, room

BUBBLE = [100, 100, 300, 260]
IN_BUBBLE = {"box": [150, 150, 250, 210], "bubble": BUBBLE, "text": "CIAO A TUTTI"}
OVER_ART = {"box": [400, 400, 500, 440], "bubble": None, "text": "Intanto, alla stazione..."}


def test_room_is_inside_the_bubble_or_around_the_text():
    inside = room(IN_BUBBLE)
    assert pymupdf.Rect(BUBBLE).contains(inside) and inside.contains(IN_BUBBLE["box"])
    assert inside.width < pymupdf.Rect(BUBBLE).width * 0.8  # the oval's inscribed width
    assert room(OVER_ART) == pymupdf.Rect(396, 396, 504, 444)


def test_letter_is_fitted_centred_and_keeps_the_case():
    page = pymupdf.open().new_page(width=600, height=600)
    letter(page, IN_BUBBLE, "Hello everyone, how are you doing today?")
    words = page.get_text("words")
    assert " ".join(w[4] for w in words) == "HELLO EVERYONE, HOW ARE YOU DOING TODAY?"
    text = pymupdf.Rect(words[0][:4])
    for w in words[1:]:
        text |= pymupdf.Rect(w[:4])
    area = room(IN_BUBBLE)
    assert area.contains(text)
    assert abs((text.y0 + text.y1) / 2 - (area.y0 + area.y1) / 2) < 6  # centred

    page = pymupdf.open().new_page(width=600, height=600)
    letter(page, OVER_ART, "Meanwhile, at the station...")
    words = [w[4] for w in page.get_text("words")]
    assert words.count("Meanwhile,") == 9  # 8 white copies for the outline, then the text


def test_fits_and_the_readable_minimum():
    small = {"box": [100, 100, 160, 120], "bubble": None, "text": "OK"}
    assert fits(small, "Ok") and not fits(small, "This is far too long for so small a box " * 3)
    page = pymupdf.open().new_page(width=600, height=600)
    letter(page, small, "This is far too long for so small a box " * 3)  # still inside its room
    words = page.get_text("words")
    inside = room(small) + (-2, -2, 2, 2)  # the outline copies are 1.5 px off
    assert words and all(inside.contains(pymupdf.Rect(w[:4])) for w in words)
