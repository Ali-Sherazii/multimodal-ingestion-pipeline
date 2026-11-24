"""PDF checks with PDFium, run before Docling: why a file can't be opened, and invisible text.

Invisible text is text a reader can't see but a chatbot would read: text the same colour as
what is behind it (e.g. white on white) or smaller than 3 pt. Invisible OCR layers on scans
(render mode 3) are NOT treated as hidden: real searchable scans rely on them.
"""
import colorsys
import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

TINY_PT = 3.0          # smaller than this can't be read in print
MIN_CONTRAST = 1.5     # text vs background; normal text is 4.5+, white on white is 1.0


class UnreadablePdf(Exception):
    """The PDF can't be read; the message says why in plain words."""


@dataclass
class PdfInfo:
    pages: int
    cut_off: bool                 # no end-of-file marker: the file was probably truncated
    open_error: str | None = None  # PDFium couldn't open it (Docling may still read some of it)
    hidden_text: list[str] = field(default_factory=list)
    chars_per_page: list[int] = field(default_factory=list)  # text the PDF itself carries
    # by page number: on pages with text turned sideways, every text run's place and turn
    turned: dict[int, list[tuple[tuple, int | None]]] = field(default_factory=dict)
    # by page number: shapes filled with a strong colour, (place, colour name)
    colours: dict[int, list[tuple[tuple, str]]] = field(default_factory=dict)


def _text(obj, textpage) -> str:
    n = pdfium_c.FPDFTextObj_GetText(obj.raw, textpage.raw, None, 0)
    if n <= 2:
        return ""
    buf = ctypes.create_string_buffer(n)
    pdfium_c.FPDFTextObj_GetText(obj.raw, textpage.raw,
                                 ctypes.cast(buf, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), n)
    return buf.raw.decode("utf-16-le", errors="ignore").rstrip("\x00")


def _fill_rgba(obj) -> tuple[int, int, int, int]:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    pdfium_c.FPDFPageObj_GetFillColor(obj.raw, *(ctypes.byref(x) for x in (r, g, b, a)))
    return r.value, g.value, b.value, a.value


def _font_size(obj) -> float:
    size = ctypes.c_float()
    pdfium_c.FPDFTextObj_GetFontSize(obj.raw, ctypes.byref(size))
    return size.value


def _bounds(obj) -> tuple[float, float, float, float]:
    l, b, r, t = (ctypes.c_float() for _ in range(4))
    pdfium_c.FPDFPageObj_GetBounds(obj.raw, *(ctypes.byref(x) for x in (l, b, r, t)))
    return l.value, b.value, r.value, t.value


def _luminance(rgb) -> float:
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(c1, c2) -> float:
    hi, lo = sorted((_luminance(c1), _luminance(c2)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _to_pixel(page, image, x: float, y: float) -> tuple[int, int]:
    # PDFium's own mapping, so crop boxes, inherited page sizes and offsets are all handled
    dx, dy = ctypes.c_int(), ctypes.c_int()
    pdfium_c.FPDF_PageToDevice(page.raw, 0, 0, image.width, image.height, 0,
                               ctypes.c_double(x), ctypes.c_double(y),
                               ctypes.byref(dx), ctypes.byref(dy))
    return dx.value, dy.value


def _background(image, page, box) -> tuple[int, int, int]:
    """Median colour of the rendered page under a text object's box."""
    l, b, r, t = box
    (xa, ya), (xb, yb) = _to_pixel(page, image, l, b), _to_pixel(page, image, r, t)
    x0, x1 = max(min(xa, xb), 0), min(max(xa, xb) + 1, image.width)
    y0, y1 = max(min(ya, yb), 0), min(max(ya, yb) + 1, image.height)
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255)
    pixels = list(image.crop((x0, y0, x1, y1)).getdata())
    return tuple(int(median(p[i] for p in pixels)) for i in range(3))


def scan_pages(pdf: pdfium.PdfDocument, check_hidden: bool) -> tuple[list[str], list[int]]:
    """(invisible text fragments, number of text characters on each page)."""
    hidden: list[str] = []
    chars: list[int] = []
    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        chars.append(textpage.count_chars())
        if not check_hidden:
            textpage.close()
            page.close()
            continue
        rendered = None
        for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_TEXT,)):
            text = _text(obj, textpage).strip()
            if len(text.replace(" ", "")) < 3:
                continue
            box = _bounds(obj)
            if _font_size(obj) < TINY_PT and (box[3] - box[1]) < TINY_PT + 1:
                hidden.append(text)
                continue
            rgba = _fill_rgba(obj)
            if rgba[3] == 0:  # fully transparent
                hidden.append(text)
                continue
            if _luminance(rgba) > 0.6:  # light text: is the background light too?
                if rendered is None:
                    rendered = page.render(scale=1).to_pil().convert("RGB")
                if _contrast(rgba, _background(rendered, page, box)) < MIN_CONTRAST:
                    hidden.append(text)
        textpage.close()
        page.close()
    return hidden, chars


def inspect_pdf(path: Path, check_hidden: bool = True) -> PdfInfo:
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(f.tell() - 1024, 0))
        cut_off = b"%%EOF" not in f.read()
    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise UnreadablePdf("password-protected PDF: save a copy without the password and "
                                "ingest that") from None
        return PdfInfo(pages=0, cut_off=cut_off, open_error=str(exc))
    try:
        hidden, chars = scan_pages(pdf, check_hidden)
        turned, colours = page_marks(pdf)
        return PdfInfo(pages=len(pdf), cut_off=cut_off, hidden_text=hidden, chars_per_page=chars,
                       turned=turned, colours=colours)
    finally:
        pdf.close()


SLANT = 5.0  # degrees away from horizontal or vertical: a diagonal watermark, not a table


def remove_slanted_text(path: Path, out: Path) -> list[str]:
    """Write a copy of the PDF at `out` without text drawn at a slant (not at 0, 90, 180 or
    270 degrees): a diagonal watermark such as "DRAFT - CONFIDENTIAL" runs across headings
    and tables, which Docling then reads merged with it. Text turned a quarter (a table
    printed sideways) is kept. Returns the texts removed; with none, no copy is written."""
    import math
    pdf = pdfium.PdfDocument(str(path))
    removed: list[str] = []
    try:
        for page in pdf:
            textpage = page.get_textpage()
            slanted = []
            for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_TEXT,), max_depth=1):
                m = obj.get_matrix()
                if SLANT < math.degrees(math.atan2(m.b, m.a)) % 90 < 90 - SLANT:
                    slanted.append(obj)
            for obj in slanted:
                removed.append(_text(obj, textpage).strip())
                page.remove_obj(obj)
                obj.close()  # no longer part of the page: freed here
            textpage.close()
            if slanted:
                page.gen_content()
            page.close()
        if removed:
            pdf.save(str(out))
    finally:
        pdf.close()
    return removed


def quarter(obj) -> int | None:
    """0, 90, 180 or 270 for text drawn upright or turned a quarter; None for a slant."""
    m = obj.get_matrix()
    angle = math.degrees(math.atan2(m.b, m.a)) % 360
    nearest = round(angle / 90) * 90 % 360
    return nearest if abs((angle - nearest + 180) % 360 - 180) <= SLANT else None


def colour_name(r: int, g: int, b: int) -> str | None:
    """The name of a strong colour (status colours: red, amber, green ...); None for white,
    greys and pale shading such as header and zebra stripes."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if s < 0.35 or v < 0.25:
        return None
    for limit, name in ((15, "red"), (50, "amber"), (70, "yellow"), (170, "green"),
                        (260, "blue"), (330, "purple"), (361, "red")):
        if h * 360 < limit:
            return name
    return None


def page_marks(pdf: pdfium.PdfDocument) -> tuple[dict, dict]:
    """Per page (numbered from 1), in points from the top-left: on pages with text turned a
    quarter (a table printed sideways), every text run's box and turn; and shapes filled
    with a strong colour (cells whose status is shown only by colour)."""
    turned: dict[int, list] = {}
    colours: dict[int, list] = {}
    for number, page in enumerate(pdf, 1):
        width, height = page.get_size()
        texts, fills = [], []
        for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_PATH),
                                    max_depth=1):
            left, bottom, right, top = obj.get_bounds()
            box = (left, height - top, right, height - bottom)
            if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                texts.append((box, quarter(obj)))
                continue
            mode, stroke = ctypes.c_int(), ctypes.c_int()
            pdfium_c.FPDFPath_GetDrawMode(obj.raw, ctypes.byref(mode), ctypes.byref(stroke))
            if not mode.value or (right - left) * (top - bottom) > 0.25 * width * height:
                continue
            r, g, b, a = (ctypes.c_uint() for _ in range(4))
            pdfium_c.FPDFPageObj_GetFillColor(obj.raw, *(ctypes.byref(x) for x in (r, g, b, a)))
            name = colour_name(r.value, g.value, b.value) if a.value > 64 else None
            if name:
                fills.append((box, name))
        if any(q in (90, 270) for _, q in texts):
            turned[number] = texts
        if fills:
            colours[number] = fills
        page.close()
    return turned, colours


def sideways(texts: list[tuple[tuple, int | None]], box: tuple) -> int | None:
    """90 or 270 if most of the text inside `box` (a table) is turned that way."""
    inside = [q for (l, t, r, b), q in texts
              if box[0] <= (l + r) / 2 <= box[2] and box[1] <= (t + b) / 2 <= box[3]]
    for q in (90, 270):
        if inside and inside.count(q) >= 0.6 * len(inside):
            return q
    return None
