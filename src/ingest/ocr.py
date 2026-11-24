"""OCR for pictures: Word images and video keyframes (Docling OCRs PDF pages itself)."""
import logging
import re
from functools import lru_cache

import numpy as np
from PIL import Image

# text, confidence, (left, top, right, bottom) in pixels
TextBox = tuple[str, float, tuple[float, float, float, float]]


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    logging.getLogger("RapidOCR").disabled = True
    return RapidOCR()


# Real text on slides and screenshots scored 0.98-1.00 in testing; shapes misread as letters
# (building windows read as "M") scored 0.66-0.77
MIN_SCORE = 0.85


def read_text(image: Image.Image) -> list[TextBox]:
    """Every piece of text found, with its confidence and position, top to bottom."""
    result = _engine()(np.array(image.convert("RGB")))
    boxes = result.boxes if result.boxes is not None else []
    return [(text.strip(), float(score),
             (min(p[0] for p in box), min(p[1] for p in box),
              max(p[0] for p in box), max(p[1] for p in box)))
            for text, score, box in zip(result.txts or [], result.scores or [], boxes)]


def confident_lines(found: list[TextBox]) -> list[str]:
    return [text for text, score, _ in found
            if score >= MIN_SCORE and sum(ch.isalnum() for ch in text) >= 2]


def ocr_lines(image: Image.Image) -> list[str]:
    """Confident text lines found in the image, top to bottom."""
    return confident_lines(read_text(image))


def ocr_image(image: Image.Image) -> str:
    return "\n".join(ocr_lines(image))


Region = tuple[float, float, float, float]  # left, top, right, bottom in pixels


def _inside(box: Region, region: Region) -> bool:
    """The box's centre lies in the region."""
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def texts_inside(found: list[TextBox], regions: list[Region]) -> set[str]:
    """The text read inside any of the regions (e.g. an equation another model has read)."""
    return {text for text, _, box in found if any(_inside(box, r) for r in regions)}


MATH_SIGNS = set("=≤≥≠≈∑∏√∫∂±∞∩∪¬∈∝·×÷|")
# a one-letter function or probability: P(A), f(x)
_FUNCTION = re.compile(r"(?<![A-Za-z])[A-Za-z]\(")


def looks_mathematical(found: list[TextBox]) -> bool:
    """Text with maths in it (=, ≤, ∑, P(A), ...): worth Docling's formula model, which takes
    about 11 s per frame even on a GPU. OCR often misses the thin "=" of a fraction, so the
    other signs count too."""
    texts = [text for text, score, _ in found if score >= 0.5]
    return any(ch in MATH_SIGNS for text in texts for ch in text) or any(
        _FUNCTION.search(text) for text in texts)


def _looks_mathematical(text: str) -> bool:
    return any(ch in MATH_SIGNS for ch in text) or bool(_FUNCTION.search(text))


def stacked_fractions(found: list[TextBox], keep: set[str] | None = None) -> list[tuple[str, set[str]]]:
    """Fractions written with words, which OCR reads as three separate lines and the formula
    model doesn't recognise as an equation, e.g.

        P(star|Green) = Total Green Stars        ->  P(star|Green)=\\frac{\\text{Total Green Stars}}
                        -----------------                                 {\\text{Total Green}}
                           Total Green

    A mathematical-looking piece with two pieces to its right, one centred above the other,
    close together, its middle between theirs. Returns each fraction as LaTeX with the OCR
    lines it replaces. Only pieces still in `keep` are used (not ones already read as part
    of another formula)."""
    pieces = [(text, box) for text, score, box in found
              if score >= MIN_SCORE and text and (keep is None or text in keep)]
    if len(pieces) < 3:
        return []
    height = float(np.median([box[3] - box[1] for _, box in pieces]))
    middle = lambda box: (box[1] + box[3]) / 2
    centre = lambda box: (box[0] + box[2]) / 2
    wrap = lambda text: f"\\text{{{text}}}" if any(ch.isalpha() for ch in text) else text

    def awaits_value(text: str) -> bool:
        # "P(star|Green)" or "P(Star|Green) =", not "P(Yes) = 0.625" (already complete)
        return "=" not in text or not text.rsplit("=", 1)[1].strip(" .:")

    def numerator_and_denominator(top: Region, bottom: Region) -> bool:
        widths = sorted((top[2] - top[0], bottom[2] - bottom[0]))
        heights = sorted((top[3] - top[1], bottom[3] - bottom[1]))
        return (abs(centre(top) - centre(bottom)) <= 0.25 * widths[1]   # centred on each other
                and bottom[1] - top[3] <= 1.5 * height                  # one fraction bar apart
                and widths[0] >= 0.25 * widths[1]                       # not a tiny sub-index
                and heights[0] >= 0.6 * heights[1])                     # same size of text

    out, used = [], set()
    for left_text, left in pieces:
        if left_text in used or not _looks_mathematical(left_text) or not awaits_value(left_text):
            continue
        right = [(t, b) for t, b in pieces
                 if t not in used and t != left_text and b[0] >= left[2] - height / 2]
        tops = sorted((p for p in right if middle(p[1]) < middle(left)),
                      key=lambda p: middle(left) - middle(p[1]))
        bottoms = sorted((p for p in right if middle(p[1]) > middle(left)),
                         key=lambda p: middle(p[1]) - middle(left))
        match = next(((top, bottom) for top in tops[:2] for bottom in bottoms[:2]
                      if numerator_and_denominator(top[1], bottom[1])), None)
        if match:
            (top_text, _), (bottom_text, _) = match
            name = left_text.rstrip(" =.:")
            out.append((f"{name}=\\frac{{{wrap(top_text)}}}{{{wrap(bottom_text)}}}",
                        {left_text, top_text, bottom_text}))
            used |= {left_text, top_text, bottom_text}
    return out


def column_tables(found: list[TextBox], skip: list[Region] = ()) -> list[str]:
    """One-column tables, as Markdown: a header with 3+ short cells stacked under it, centred
    on one line and evenly spaced (e.g. a list of sample ids). Docling's layout model takes
    these for pictures. Text inside `skip` (tables Docling already read) is left out.
    """
    pieces = [(text, box) for text, score, box in found
              if score >= MIN_SCORE and text and len(text) <= 12
              and not any(_inside(box, region) for region in skip)]
    if len(pieces) < 4:
        return []
    height = float(np.median([box[3] - box[1] for _, box in pieces]))
    centre_x = lambda p: (p[1][0] + p[1][2]) / 2
    centre_y = lambda p: (p[1][1] + p[1][3]) / 2

    columns: list[list] = []  # pieces whose centres line up vertically
    for piece in sorted(pieces, key=centre_x):
        if columns and abs(centre_x(piece) - np.mean([centre_x(p) for p in columns[-1]])) <= height * 0.6:
            columns[-1].append(piece)
        else:
            columns.append([piece])

    tables = []
    for column in columns:
        column.sort(key=centre_y)
        runs, run = [], [column[0]]  # split where the column has a gap of more than a row
        for piece in column[1:]:
            if centre_y(piece) - centre_y(run[-1]) <= height * 2:
                run.append(piece)
            else:
                runs.append(run)
                run = [piece]
        runs.append(run)
        for run in runs:
            gaps = np.diff([centre_y(p) for p in run])
            header = run[0][0]
            if (len(run) >= 4 and any(ch.isalpha() for ch in header)
                    and gaps.min() >= height * 0.8 and gaps.max() <= gaps.min() * 1.6):
                rows = "\n".join(f"| {text} |" for text, _ in run[1:])
                tables.append(f"| {header} |\n|---|\n{rows}")
    return tables


def looks_like_table(found: list[TextBox]) -> bool:
    """Text laid out in rows and columns: at least 3 rows holding 2+ separate pieces of text.

    A cheap check before running Docling's table model (seconds per frame). Bullet lists give
    one piece per row; a two-column slide can pass, and Docling then simply finds no table.
    """
    boxes = [box for text, score, box in found if score >= MIN_SCORE and text]
    if len(boxes) < 6:
        return False
    line_height = float(np.median([bottom - top for _, top, _, bottom in boxes]))
    rows: list[list[float]] = []  # vertical centres of the pieces in each row
    for _, top, _, bottom in sorted(boxes, key=lambda b: b[1] + b[3]):
        centre = (top + bottom) / 2
        if rows and abs(centre - np.mean(rows[-1])) <= line_height / 2:
            rows[-1].append(centre)
        else:
            rows.append([centre])
    return sum(len(row) >= 2 for row in rows) >= 3
