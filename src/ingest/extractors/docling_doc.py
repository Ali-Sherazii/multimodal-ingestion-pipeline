"""PDF and Word extractor (FR-04, FR-05): Docling reads layout, reading order, tables and OCR.

PDFs are checked with PDFium first (pdf_checks.py) and converted in page batches, so a very
long PDF can't exhaust memory. Problems that don't stop extraction (damaged pages, a garbage
text layer, invisible text) come back as warnings on the file instead of being hidden.
"""
import gc
import re
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, OcrMode, PdfPipelineOptions
from docling.document_converter import (DocumentConverter, ImageFormatOption, PdfFormatOption,
                                        WordFormatOption)
from docling_core.types.doc import (DocItemLabel, DoclingDocument, ListItem, PictureItem,
                                    SectionHeaderItem, TableItem, TextItem, TitleItem)
from PIL import Image

from ingest.config import Settings
from ingest.logs import log
from ingest.models import Extraction, Modality, Segment
from ingest.ocr import ocr_image
from ingest.pdf_checks import PdfInfo, UnreadablePdf, inspect_pdf, remove_slanted_text, sideways
from ingest.quality import garbage_share, looks_like_garbage
from ingest.tables import (Block, attach, contents_list, continuation, docling_table, finish,
                           form_lines, join)
from ingest.tables import parts as table_parts


def formulas_on(settings: Settings) -> bool:
    """Equations are written as LaTeX by Docling's formula model (CodeFormulaV2), a vision-
    language model too slow for a CPU (about 11 s per page or frame even on a GPU), so only
    where a GPU is available: on Modal, or locally with an NVIDIA card."""
    return settings.formulas and settings.use_cuda()


def compact_latex(latex: str) -> str:
    """CodeFormula separates every token ("P ( G r e e n )"); join them so words stay words
    for search, keeping a space only where LaTeX needs one (a command before a letter)."""
    out = ""
    for token in latex.replace("&", " ").split():
        if out and re.search(r"\\[A-Za-z]+$", out) and token[0].isalpha():
            out += " "
        out += token
    return out


@lru_cache(maxsize=2)
def _converter(ocr: bool, force_full_page_ocr: bool, crop_pictures: bool,
               device: str, formulas: bool = False,
               artifacts: str | None = None) -> DocumentConverter:
    pdf_options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True,
                                     do_formula_enrichment=formulas, artifacts_path=artifacts)
    pdf_options.ocr_options.mode = OcrMode.FULL_PAGE if force_full_page_ocr else OcrMode.DEFAULT
    pdf_options.accelerator_options = AcceleratorOptions(device=device)
    if crop_pictures:  # keep each PDF picture as an image so it can be described
        pdf_options.generate_picture_images = True
        pdf_options.images_scale = 2.0
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.DOCX: WordFormatOption(),  # Word files carry their pictures already
        },
    )


@lru_cache(maxsize=2)
def _image_converter(device: str, formulas: bool, artifacts: str | None = None) -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=True, do_table_structure=True,
                                 do_formula_enrichment=formulas, artifacts_path=artifacts)
    options.accelerator_options = AcceleratorOptions(device=device)
    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)})


def read_image(path: Path, settings: Settings, formulas: bool = False) -> dict:
    """Tables, and equations if `formulas`, in a picture such as a video frame, read with
    Docling's layout, table-structure and formula models. Each comes with where it is in the
    picture (left, top, right, bottom in pixels), so the plain OCR of it can be dropped:
    {"tables": [(markdown, region)], "formulas": [(latex, region)]}. Tables of fewer than two
    rows are left to plain OCR."""
    out = {"tables": [], "formulas": []}
    result = _image_converter(settings.device, formulas,
                              settings.docling_artifacts_path).convert(path, raises_on_error=False)
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        return out
    doc = result.document
    with Image.open(path) as image:
        width, height = image.size

    def region(item):
        if not item.prov:
            return None
        page = doc.pages[item.prov[0].page_no].size
        box = item.prov[0].bbox.to_top_left_origin(page_height=page.height)
        sx, sy = width / page.width, height / page.height
        return (box.l * sx, box.t * sy, box.r * sx, box.b * sy)

    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            markdown = item.export_to_markdown(doc).strip()
            if len(markdown.splitlines()) >= 3:  # header, separator, at least one row
                out["tables"].append((markdown, region(item)))
        elif formulas and getattr(item, "label", None) == DocItemLabel.FORMULA and item.text.strip():
            out["formulas"].append((compact_latex(item.text), region(item)))
    return out


def release_image_converter() -> None:
    """Free the table models (about 1 GB) once a video's frames are done."""
    _image_converter.cache_clear()
    gc.collect()


def _page_of(item) -> int | None:
    # Word files have no page layout, so their items carry no page provenance
    return item.prov[0].page_no if getattr(item, "prov", None) else None


def _strip_hidden(text: str, hidden: list[str], removed: list[str]) -> str:
    """Cut invisible fragments (found by PDFium) out of a piece of extracted text."""
    if not hidden:
        return text
    flat = " ".join(text.split())
    found = False
    for fragment in hidden:
        frag = " ".join(fragment.split())
        if frag and frag in flat:
            flat = flat.replace(frag, " ")
            removed.append(frag)
            found = True
    return " ".join(flat.split()) if found else text


def _block(item, doc: DoclingDocument, text: str) -> Block:
    """What tables.attach needs to know about an item: its kind, text and place."""
    if isinstance(item, TableItem):
        kind = "table"
    elif isinstance(item, (TitleItem, SectionHeaderItem)):
        kind = "heading"
    elif isinstance(item, PictureItem):
        kind = "picture"
    elif isinstance(item, TextItem):
        kind = {DocItemLabel.CAPTION: "caption", DocItemLabel.FOOTNOTE: "footnote"}.get(
            item.label, "text")
    else:
        kind = "other"
    page, box = _page_of(item), None
    if page is not None and doc.pages.get(page) is not None:
        b = item.prov[0].bbox.to_top_left_origin(doc.pages[page].size.height)
        box = (b.l, b.t, b.r, b.b)
    return Block(kind, text, page, box)


def to_segments(doc: DoclingDocument, ocr_pictures: bool = False,
                describe: Callable[[Image.Image], str] | None = None,
                min_px: int = 100, hidden: list[str] | None = None,
                start_heading: str | None = None,
                removed: list[str] | None = None,
                latex_formulas: bool = False,
                turned: dict | None = None, colours: dict | None = None) -> list[Segment]:
    """Group items into sections that share a heading and a page.

    Headers and footers are in Docling's 'furniture' layer and are left out, so lines like
    'Confidential' do not repeat in every chunk. Each table becomes its own segment (one per
    page it is printed on), rebuilt from its cells with its caption, introduction and notes
    (tables.py), so the chunker can keep it whole or repeat its caption and header in every
    piece; caption and note text used by a table isn't repeated as text. A table that goes
    on over the next page is joined to its start first. `turned` and `colours` (from
    PDFium, by page) show tables printed sideways and cells holding only a colour.
    `start_heading` carries a heading over from the previous page batch. With
    `latex_formulas`, equations (written as LaTeX by the formula model) are kept as $$...$$.
    """
    hidden = hidden or []
    removed = removed if removed is not None else []
    clean = lambda text: _strip_hidden(text, hidden, removed)

    items = [item for item, _level in doc.iterate_items(traverse_pictures=True)]
    texts = [clean(item.text) if isinstance(item, TextItem) else "" for item in items]
    blocks = [_block(item, doc, text) for item, text in zip(items, texts)]
    index = {item.self_ref: i for i, item in enumerate(items)}
    linked = {i: [index[ref.cref] for ref in item.captions if ref.cref in index]
              for i, item in enumerate(items) if isinstance(item, TableItem)}
    context = attach(blocks, linked)

    segments: list[Segment] = []
    heading: str | None = start_heading
    heading_page: int | None = None
    heading_has_body = True
    page: int | None = None
    parts: list[str] = []

    def flush():
        nonlocal heading_has_body
        if parts:
            segments.append(Segment(text="\n\n".join(parts), section_heading=heading,
                                    page_number=page))
            parts.clear()
            heading_has_body = True

    def keep_empty_heading():
        # A heading with nothing under it is still content: on scans the layout model
        # sometimes labels a whole block of text as a heading
        if heading and not heading_has_body:
            segments.append(Segment(text=heading, section_heading=heading,
                                    page_number=heading_page))

    # The last table, held back until it is clear whether the next item goes on with it
    held: list = []  # [table, heading, page of its last part]

    def release():
        if held:
            table, table_heading, last_page = held
            for part_page, part in table_parts(finish(table)):
                segments.append(Segment(text=part.text(), modality=Modality.TABLE,
                                        section_heading=table_heading,
                                        page_number=part_page or last_page))
            held.clear()

    for i, item in enumerate(items):
        if i in context.used:  # a caption or note now part of its table
            continue
        item_page = _page_of(item)
        if item_page is not None and item_page != page:
            flush()
            page = item_page
        if not isinstance(item, TableItem):
            release()  # anything else between two tables: the second is a table of its own

        if isinstance(item, (TitleItem, SectionHeaderItem)):
            text = context.rest.get(i, texts[i])
            if not text.strip():
                continue
            flush()
            keep_empty_heading()
            heading, heading_page, heading_has_body = text, page, False
        elif isinstance(item, TableItem):
            flush()  # a table is its own segment, so its rows can be chunked with the header
            box = blocks[i].box
            quarter = sideways(turned[page], box) if turned and box and page in turned else None
            table = docling_table(item, doc, clean, quarter, (colours or {}).get(page))
            if table is None or not (table.rows or any(table.header)):
                continue
            heading_has_body = True
            found = context.tables.get(i)
            if found:
                table.caption, table.intro, table.notes = found.caption, found.intro, found.notes
            table.pages = [page] * len(table.rows)
            special = contents_list(table) or form_lines(table)
            if special:  # a table of contents or a form: plain lines
                release()
                top = "\n".join(p for p in (table.caption, table.intro) if p)
                text = "\n\n".join(p for p in (top, special, table.notes) if p)
                segments.append(Segment(text=text, modality=Modality.TABLE if top else Modality.TEXT,
                                        section_heading=heading, page_number=page))
                continue
            how = continuation(held[0], table) if held and (
                page is None or held[2] is None or 0 <= page - held[2] <= 1) else None
            if how:  # goes on from the table before it (next page)
                held[0], held[2] = join(held[0], table, how), page
            else:
                release()
                held.extend([table, heading, page])
        elif isinstance(item, PictureItem):
            image = item.get_image(doc) if (ocr_pictures or describe) else None
            if image is None or min(image.size) < min_px:  # unreadable (e.g. EMF) or a logo/icon
                continue
            if ocr_pictures:
                text = ocr_image(image)
                if text:
                    parts.append(f"[Text in image]\n{text}")
            description = describe(image) if describe else None
            if description:
                flush()  # text before the picture stays in reading order
                caption = item.caption_text(doc)  # e.g. "Figure 3: Sales by month"
                segments.append(Segment(
                    text=f"{caption}: {description}" if caption else description,
                    modality=Modality.PICTURE_DESCRIPTION,
                    section_heading=heading,
                    page_number=item_page or page))
                heading_has_body = True
        elif latex_formulas and item.label == DocItemLabel.FORMULA and item.text.strip():
            parts.append(f"$${compact_latex(item.text)}$$")  # written by the formula model
        elif isinstance(item, TextItem) and item.text.strip():  # includes list items
            text = context.rest.get(i, texts[i])
            if text.strip():
                parts.append(f"- {text}" if isinstance(item, ListItem) else text)
    release()
    flush()
    keep_empty_heading()
    return segments


def _is_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def _needs_gpu(path: Path, settings: Settings) -> bool:
    """PDFs always benefit; Word files only when pictures are described (otherwise ~1 s locally)."""
    return _is_pdf(path) or settings.pdf_picture_descriptions


def _describer(settings: Settings) -> Callable[[Image.Image], str] | None:
    if not settings.pdf_picture_descriptions:
        return None
    from ingest.vision import get_captioner
    captioner = get_captioner(settings)
    return lambda image: captioner.caption(image, settings.picture_prompt)


def _page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'"""
    spans, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i + 1 == len(pages) or pages[i + 1] != p + 1:
            spans.append(f"{start}-{p}" if p != start else f"{p}")
            start = None
    return ", ".join(spans)


def _convert_pdf(path: Path, settings: Settings, info: PdfInfo,
                 force_ocr: bool) -> tuple[list[DoclingDocument], list[str]]:
    """Convert in batches of pdf_page_batch pages, so memory stays bounded on long PDFs."""
    converter = _converter(settings.pdf_ocr, force_ocr, settings.pdf_picture_descriptions,
                           settings.device, formulas_on(settings), settings.docling_artifacts_path)
    batch = max(settings.pdf_page_batch, 1)
    if info.pages <= batch:  # also when PDFium couldn't count pages: let Docling try it whole
        ranges: list[tuple[int, int] | None] = [None]
    else:
        ranges = [(s, min(s + batch - 1, info.pages)) for s in range(1, info.pages + 1, batch)]
    docs, problems = [], []
    for r in ranges:
        result = converter.convert(path, raises_on_error=False, **({"page_range": r} if r else {}))
        where = f"pages {r[0]}-{r[1]}" if r else "the document"
        if result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            docs.append(result.document)
            if result.status is ConversionStatus.PARTIAL_SUCCESS:
                problems.append(f"{where} were only partly read")
        else:
            reason = result.errors[0].error_message if result.errors else result.status.value
            problems.append(f"{where} could not be read ({reason[:120]})")
        if r:
            log.info(f"pages {r[0]}-{r[1]} of {info.pages} converted", extra={"file": path.name})
        del result
        gc.collect()
    return docs, problems


def _plain_texts(docs: list[DoclingDocument]) -> list[str]:
    return [item.text for d in docs for item, _ in d.iterate_items(traverse_pictures=True)
            if isinstance(item, TextItem)]


def _last_heading(doc: DoclingDocument) -> str | None:
    last = None
    for item, _ in doc.iterate_items():
        if isinstance(item, (TitleItem, SectionHeaderItem)) and item.text.strip():
            last = item.text
    return last


def _without_watermark(path: Path, folder: Path, settings: Settings, info: PdfInfo,
                       warnings: list[str]) -> Path:
    """The PDF to read: a copy in `folder` without diagonal watermark text, if it has any."""
    if not settings.pdf_watermark_filter or info.open_error:
        return path
    clean = folder / path.name
    try:
        marks = remove_slanted_text(path, clean)
    except Exception as exc:  # PDFium couldn't rewrite it: read the PDF as it is
        log.warning(f"watermark check failed ({type(exc).__name__}: {exc}); read as it is",
                    extra={"file": path.name})
        return path
    if not marks:
        return path
    kinds = sorted(set(" ".join(m.split()) for m in marks if m.strip()))
    warnings.append(f"removed diagonal watermark text before reading ({len(marks)} time(s)"
                    + (f", e.g. {kinds[0][:60]!r}" if kinds else "") + ")")
    return clean


def _extract_pdf(path: Path, settings: Settings) -> Extraction:
    info = inspect_pdf(path, check_hidden=settings.pdf_hidden_text_filter)  # raises if encrypted
    warnings: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ingest-") as tmp:
        source = _without_watermark(path, Path(tmp), settings, info, warnings)
        docs, problems = _convert_pdf(source, settings, info, settings.pdf_force_full_page_ocr)
        texts = _plain_texts(docs)
        if (settings.pdf_garbage_retry and settings.pdf_ocr
                and not settings.pdf_force_full_page_ocr and looks_like_garbage(texts)):
            log.info("text layer looks like garbage; re-reading every page with OCR",
                     extra={"file": path.name})
            retry_docs, retry_problems = _convert_pdf(source, settings, info, force_ocr=True)
            before, after = garbage_share(texts), garbage_share(_plain_texts(retry_docs))
            if after[0] / max(after[1], 1) < before[0] / max(before[1], 1):
                docs, problems = retry_docs, retry_problems
                warnings.append(f"the PDF's own text layer was garbage ({before[0]} unreadable "
                                "words, e.g. from a bad old OCR), so every page was re-read with OCR")
            else:
                warnings.append(f"{before[0]} words look like garbage (broken text layer or fonts)")

    describe = _describer(settings)
    segments: list[Segment] = []
    removed: list[str] = []
    heading = None
    for doc in docs:
        segments += to_segments(doc, describe=describe, min_px=settings.picture_min_px,
                                hidden=info.hidden_text, start_heading=heading, removed=removed,
                                latex_formulas=formulas_on(settings), turned=info.turned,
                                colours=info.colours)
        heading = _last_heading(doc) or heading

    damage = info.open_error or ("the file is cut off (no end-of-file marker)" if info.cut_off else None)
    if not segments and damage:
        raise UnreadablePdf(f"corrupt or truncated PDF, no text could be read: {damage}")

    if removed:
        warnings.append(f"removed {len(removed)} invisible text fragment(s) (same colour as the "
                        f"background, or under 3 pt), e.g. {removed[0][:70]!r}")
    if info.pages and segments:
        with_text = {_page_of(item) for d in docs for item, _ in d.iterate_items(traverse_pictures=True)
                     if isinstance(item, (TextItem, TableItem)) and _page_of(item)}
        missing = [p for p in range(1, info.pages + 1) if p not in with_text]
        # The PDF itself has text on these pages, but none came out for them
        unread = [p for p in missing if info.chars_per_page[p - 1] > 0]
        empty = [p for p in missing if p not in unread]
        if empty:
            warnings.append(f"{len(empty)} of {info.pages} pages produced no text "
                            f"(pages {_page_ranges(empty)}): blank, images only, or damaged")
        if unread:
            one = len(unread) == 1
            warnings.append(f"{'page' if one else 'pages'} {_page_ranges(unread)} "
                            f"{'contains' if one else 'contain'} text that was not read as "
                            f"{'that page' if one else 'those pages'}: page numbers may be "
                            "wrong, or the page is damaged")
    if damage and segments:
        warnings.append(f"damaged PDF ({damage}); kept the text that could be read")
    warnings += problems
    return Extraction(segments, warnings)


def extract_document(path: Path, settings: Settings) -> Extraction:
    if settings.gpu_backend == "modal" and _needs_gpu(path, settings):
        from ingest import gpu_modal
        return gpu_modal.convert_document(path, settings)
    if _is_pdf(path):
        return _extract_pdf(path, settings)

    converter = _converter(settings.pdf_ocr, settings.pdf_force_full_page_ocr,
                           settings.pdf_picture_descriptions, settings.device,
                           artifacts=settings.docling_artifacts_path)
    result = converter.convert(path)
    return Extraction(to_segments(result.document, ocr_pictures=settings.docx_image_ocr,
                                  describe=_describer(settings), min_px=settings.picture_min_px))
