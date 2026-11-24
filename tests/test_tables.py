"""Tables rebuilt for search (tables.py), their chunks, and watermark removal, on small
made-up inputs shaped like the hard cases in eval/tables (no Docling, no models)."""
import ctypes
import math
import re
from pathlib import Path

from ingest.chunk import _split_table, chunk_segments
from ingest.models import DetectedFile, FileType, Modality, Segment
from ingest.pdf_checks import page_marks, remove_slanted_text, sideways
from ingest.tables import (Block, Cell, Table, _colour_cells, attach, build, contents_list,
                           continuation, finish, form_lines, join, parts, ticked, turn)


def grid(rows: list[list[str]], header_rows: int = 1, spans: dict | None = None) -> list[Cell]:
    """Cells for a table given as rows of text; spans: {(row, col): (rows, cols)}."""
    spans = spans or {}
    covered = {(r + i, c + j) for (r, c), (h, w) in spans.items()
               for i in range(h) for j in range(w) if (i, j) != (0, 0)}
    return [Cell(text, r, c, *spans.get((r, c), (1, 1)), header=r < header_rows)
            for r, row in enumerate(rows) for c, text in enumerate(row) if (r, c) not in covered]


# Table 1 of the stress test: a group row across the table, the branch on its first row only
BRANCHES = [
    ["Code", "Branch", "Qtr", "Revenue"],
    ["Region: Punjab", "", "", ""],
    ["PU-107", "Lahore Central", "Q1", "9,806"],
    ["", "", "Q2", "35,542"],
    ["", "", "Q3", "8,726"],
    ["PU-114", "Lahore Ring Rd", "Q1", "29,565"],
    ["", "", "Q2", "37,901"],
    ["", "Subtotal Punjab", "", "121,540"],
    ["Region: Sindh", "", "", ""],
    ["SI-156", "Hyderabad", "Q1", "38,962"],
    ["", "", "Q3", "3,663"],
    ["", "GRAND TOTAL", "", "163,165"],
]


def test_group_rows_become_a_column_and_rows_name_their_branch():
    spans = {(1, 0): (1, 4), (8, 0): (1, 4)}
    table = finish(build(grid(BRANCHES, spans=spans), len(BRANCHES), 4))
    md = table.markdown().splitlines()
    assert md[0] == "| Region | Code | Branch | Qtr | Revenue |"
    assert "| Punjab | PU-107 | Lahore Central | Q2 | 35,542 |" in md
    assert "| Sindh | SI-156 | Hyderabad | Q3 | 3,663 |" in md
    assert "| Punjab |  | Subtotal Punjab |  | 121,540 |" in md   # not given a branch code
    assert "|  |  | GRAND TOTAL |  | 163,165 |" in md              # in no region


def test_spans_stacked_headers_and_real_empty_cells():
    rows = [["Zone", "Region", "FY2026", ""],
            ["", "", "H1", "H2"],
            ["North", "Punjab", "246,949", "115,265"],
            ["", "Sindh", "218,789", ""]]
    spans = {(0, 0): (2, 1), (0, 1): (2, 1), (0, 2): (1, 2), (2, 0): (2, 1)}
    table = build(grid(rows, header_rows=2, spans=spans), 4, 4)
    assert table.header == ["Zone", "Region", "FY2026 - H1", "FY2026 - H2"]
    assert table.rows == [["North", "Punjab", "246,949", "115,265"],
                          ["North", "Sindh", "218,789", ""]]  # zone repeated; H2 stays empty
    wide = build(grid([["Service", "Commitment", "x"], ["Express", "Pickup in 2 hours", "10%"]],
                      spans={(1, 1): (1, 1)}) + [Cell("Pickup in 2 hours", 1, 1, 1, 2)], 2, 3)
    assert wide.rows == [["Express", "Pickup in 2 hours", ""]]  # across columns: written once


def test_contents_page_becomes_lines():
    rows = [["Executive summary ........ 2"], ["1. Branch financials ........ 3"],
            ["2. Shipment log ........ 7"]]
    text = contents_list(build(grid(rows), 3, 1))
    assert text.splitlines() == ["Executive summary: page 2", "1. Branch financials: page 3",
                                 "2. Shipment log: page 7"]
    assert contents_list(Table(["a", "b"], [["1", "2"]] * 3)) is None


def test_captions_by_position_notes_and_introduction():
    blocks = [
        Block("heading", "7. Customers and suppliers", 14, (80, 60, 400, 80)),
        Block("text", "Table 7a: Top 5 customers", 14, (80, 100, 300, 112)),   # over the left
        Block("text", "Table 7b: Top 5 suppliers", 14, (450, 100, 700, 112)),  # over the right
        Block("table", "", 14, (80, 118, 400, 250)),
        Block("table", "", 14, (450, 118, 780, 250)),
        Block("text", "Rows marked (EU) use comma decimals.", 15, (80, 80, 500, 92)),
        Block("text", "Table 4: Unit rates", 15, (80, 100, 300, 112)),
        Block("table", "", 15, (80, 118, 500, 300)),
        Block("text", "Legend: green = on track, red = off track.", 15, (80, 306, 500, 316)),
        Block("text", "An ordinary paragraph after it.", 15, (80, 330, 500, 340)),
        Block("table", "", 16, (80, 100, 500, 200)),
        Block("text", "Table 17: Branch managers (caption BELOW the table)", 16, (80, 206, 500, 218)),
    ]
    found = attach(blocks)
    assert found.tables[3].caption == "Table 7a: Top 5 customers"
    assert found.tables[4].caption == "Table 7b: Top 5 suppliers"
    assert found.tables[7].caption == "Table 4: Unit rates"
    assert found.tables[7].intro == "Rows marked (EU) use comma decimals."
    assert found.tables[7].notes == "Legend: green = on track, red = off track."
    assert found.tables[10].caption.startswith("Table 17: Branch managers")
    assert {1, 2, 6, 8, 11} <= found.used and 9 not in found.used and 5 not in found.used


def test_caption_inside_other_text_is_cut_out():
    blocks = [Block("text", "3. Regional results Table 3: Revenue by region Footnote: H1 = July",
                    11, (80, 90, 700, 110)),
              Block("table", "", 11, (80, 118, 700, 300))]
    found = attach(blocks)
    assert found.tables[1].caption == "Table 3: Revenue by region"
    assert found.rest[0] == "3. Regional results Footnote: H1 = July"


def words(text: str) -> int:
    return len(text.split())


def test_table_segment_pieces_repeat_caption_header_and_notes():
    rows = [[f"B{i}", str(100 + i)] for i in range(30)]
    table = Table(["Bay", "Free"], rows, caption="Table 16b: Lahore Central depot bays",
                  notes="Source: depot survey.")
    seg = Segment(table.text(), Modality.TABLE, "16. Depot detail", page_number=20)
    detected = DetectedFile(Path("r.pdf"), FileType.PDF, "x")
    whole = chunk_segments([seg], detected, "r.pdf", "t", 400, 100, table_size=1000, count=words)
    assert [c.content for c in whole] == [table.text()]                  # fits: kept whole
    pieces = chunk_segments([seg], detected, "r.pdf", "t", 400, 100, table_size=80, count=words)
    assert len(pieces) > 1 and all(c.modality is Modality.TABLE for c in pieces)
    for c in pieces:
        assert c.content.startswith("Table 16b: Lahore Central depot bays\n\n| Bay | Free |")
        assert c.content.endswith("\n\nSource: depot survey.")  # already labelled: kept as is
    assert sum(len(re.findall(r"\| B\d", c.content)) for c in pieces) == 30  # every row, once
    # text around a plain table (not a table segment) is not treated as its caption
    assert _split_table("Some words\n\n| a |\n|---|\n| 1 |", 5) is None


def _pdf_with_text(path: Path, angles: list[float]) -> None:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(600, 800)
    font = raw.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
    for k, angle in enumerate(angles):
        obj = raw.FPDFPageObj_CreateTextObj(pdf.raw, font, 14.0)
        text = ctypes.create_unicode_buffer(f"text at {angle:g} degrees")
        raw.FPDFText_SetText(obj, ctypes.cast(text, ctypes.POINTER(raw.FPDF_WCHAR)))
        a = math.radians(angle)
        raw.FPDFPageObj_Transform(obj, math.cos(a), math.sin(a), -math.sin(a), math.cos(a),
                                  100, 100 + 150 * k)
        raw.FPDFPage_InsertObject(page.raw, obj)
    raw.FPDFPage_GenerateContent(page.raw)
    page.close()
    pdf.save(str(path))
    pdf.close()


def test_diagonal_watermark_is_removed_and_turned_tables_kept(tmp_path):
    source, clean = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _pdf_with_text(source, [0, 90, 35, 270])
    assert remove_slanted_text(source, clean) == ["text at 35 degrees"]
    assert remove_slanted_text(clean, tmp_path / "again.pdf") == []
    assert not (tmp_path / "again.pdf").exists()  # nothing to remove: no copy written


# ---- tables over several pages ----

def paged(header, rows, page, caption="", groups=None):
    return Table(header, [list(r) for r in rows], caption, groups=groups or {},
                 pages=[page] * len(rows))


def test_next_page_with_the_same_header_carries_the_region_and_page_numbers():
    head = ["Code", "Branch", "Qtr", "Revenue"]
    first = paged(head, [["SI-142", "Karachi Port", "Q1", "39,842"], ["", "", "Q2", "26,172"],
                         ["", "", "Q3", "29,670"]], 3, "Table 1: Revenue by branch",
                  groups={0: "Region: Sindh"})
    second = paged(head, [["SI-163", "Sukkur", "Q1", "36,289"], ["", "", "Q2", "8,722"]], 4)
    assert continuation(first, second) == "rows"
    table = finish(join(first, second, "rows"))
    pages = parts(table)
    assert [p for p, _ in pages] == [3, 4]
    on_4 = pages[1][1]
    assert on_4.caption == "Table 1: Revenue by branch" and on_4.header[0] == "Region"
    assert on_4.rows[1] == ["Sindh", "SI-163", "Sukkur", "Q2", "8,722"]


def test_next_page_without_header_gets_the_first_pages_header():
    head = ["Shipment", "Date", "Origin", "Weight (kg)", "Notes"]
    rows = [[f"KF-26-40{i}", "2026-01-12", "Turbat", "8,325.1", ""] for i in range(10, 14)]
    first = paged(head, rows, 7, "Table 2: Shipment log")
    second = paged(["KF-26-4351", "2026-12-12", "Hyderabad", "8,448.7", "Customs hold"],
                   [["KF-26-4364", "2026-04-12", "Mardan", "8,124.8", ""]], 8)
    assert continuation(first, second) == "headerless"
    table = join(first, second, "headerless")
    assert table.header == head and table.rows[4][0] == "KF-26-4351" and table.pages[4] == 8


def test_table_split_by_columns_is_joined_on_its_row_numbers():
    first = paged(["#", "Depot", "Jul"], [["1", "Karachi Port", "84,882"], ["7", "Quetta", "39,757"]],
                  15, "Table 9: Diesel by depot")
    second = paged(["#", "Jan", "Mar"], [["1", "22,556", "62,514"], ["7", "19,714", "89,596"]], 16,
                   "Table 9 (continued): January-June")
    assert continuation(first, second) == "columns"
    table = join(first, second, "columns")
    assert table.rows[1] == ["7", "Quetta", "39,757", "19,714", "89,596"]
    assert "Table 9 (continued)" in table.notes


def test_another_table_or_another_shape_is_not_joined():
    head = ["Bay", "Capacity", "Free"]
    karachi = paged(head, [["B1", "271", "33"]], 20, "Table 16a: Karachi Port depot bays")
    lahore = paged(head, [["B1", "282", "145"]], 20, "Table 16b: Lahore Central depot bays")
    assert continuation(karachi, lahore) is None  # its own caption
    words = paged(["Name", "Town"], [["Ali", "Multan"], ["Saima", "Sukkur"], ["Nadia", "Mirpur"]], 1)
    assert continuation(words, paged(["Tariq", "Lahore"], [], 2)) is None  # words only: can't tell


# ---- special layouts ----

def test_sideways_table_is_turned_upright():
    # printed sideways, text running up the page: Docling sees the columns as rows, last first
    seen = [["Depot", "PEW", "KHI"], ["Leased", "37", "33"], ["Type", "Prime mover", "Reefer 20ft"]]
    cells, rows, cols = turn(grid(seen, header_rows=0), 3, 3, 90)
    table = build(cells, rows, cols)
    assert table.header == ["Type", "Leased", "Depot"]
    assert table.rows == [["Prime mover", "37", "PEW"], ["Reefer 20ft", "33", "KHI"]]
    # turned the other way (text running down): the header ends up as the right-hand column
    cells, rows, cols = turn(grid([["Pickup", "Type"], ["33", "Leased"]], header_rows=0), 2, 2, 270)
    table = build(cells, rows, cols)
    assert table.header == ["Type", "Leased"] and table.rows == [["Pickup", "33"]]


def test_cells_with_only_a_colour_say_the_colour():
    cells = [Cell("Project", 0, 0, header=True), Cell("Jan", 0, 1, header=True),
             Cell("Feb", 0, 2, header=True), Cell("ERP migration", 1, 0)]
    boxes = {0: (80, 100, 120, 110), 1: (210, 100, 230, 110), 2: (260, 100, 280, 110),
             3: (80, 120, 160, 130)}
    fills = [((205, 116, 255, 134), "green"), ((255, 116, 305, 134), "amber")]
    table = build(_colour_cells(cells, boxes, fills), 2, 3)
    assert table.rows == [["ERP migration", "green", "amber"]]


def test_forms_become_key_value_lines_and_ticks_are_spelt_out():
    form = Table(["Vendor name", "Atlas Tyres (Pvt.) Ltd.", "NTN", "4410287-6"],
                 [["Contact", "Ms. Hina Akhtar", "Phone", "+92 21 3507 0000"],
                  ["Approved", ticked("☑ Yes ☐ No"), "Date", "14-Jan-2026"]])
    assert form_lines(form).splitlines() == [
        "Vendor name: Atlas Tyres (Pvt.) Ltd.", "NTN: 4410287-6", "Contact: Ms. Hina Akhtar",
        "Phone: +92 21 3507 0000", "Approved: ☑ Yes ☐ No (ticked: Yes)", "Date: 14-Jan-2026"]
    kpis = Table(["KPI", "Value"], [["On-time %", "91.4"], ["Claims ratio", "0.37%"]])
    assert form_lines(kpis) is None  # first line holds column names: a normal table


def _pdf_with_shapes(path: Path) -> None:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(600, 800)
    font = raw.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
    for k in range(5):  # a sideways table: words turned 90 degrees around (300, 400)
        obj = raw.FPDFPageObj_CreateTextObj(pdf.raw, font, 10.0)
        text = ctypes.create_unicode_buffer(f"cell {k}")
        raw.FPDFText_SetText(obj, ctypes.cast(text, ctypes.POINTER(raw.FPDF_WCHAR)))
        raw.FPDFPageObj_Transform(obj, 0, 1, -1, 0, 280 + 12 * k, 380)
        raw.FPDFPage_InsertObject(page.raw, obj)
    for (r, g, b), x in (((76, 175, 80), 50), ((255, 193, 7), 120), ((220, 230, 241), 190)):
        rect = raw.FPDFPageObj_CreateNewRect(x, 700, 60, 20)  # green, amber, pale header blue
        raw.FPDFPageObj_SetFillColor(rect, r, g, b, 255)
        raw.FPDFPath_SetDrawMode(rect, raw.FPDF_FILLMODE_ALTERNATE, 0)
        raw.FPDFPage_InsertObject(page.raw, rect)
    raw.FPDFPage_GenerateContent(page.raw)
    page.close()
    pdf.save(str(path))
    pdf.close()


def test_pdfium_finds_sideways_text_and_status_colours(tmp_path):
    import pypdfium2 as pdfium
    _pdf_with_shapes(tmp_path / "shapes.pdf")
    pdf = pdfium.PdfDocument(str(tmp_path / "shapes.pdf"))
    try:
        turned, colours = page_marks(pdf)
    finally:
        pdf.close()
    assert sideways(turned[1], (200, 300, 400, 500)) == 90
    assert sorted(name for _, name in colours[1]) == ["amber", "green"]  # pale blue ignored
    green = next(box for box, name in colours[1] if name == "green")
    assert [round(v) for v in green] == [50, 80, 110, 100]  # measured from the top of the page
