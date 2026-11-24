"""Build a long, fictional annual report full of tables that are hard for TableFormer/Docling,
plus an answer key (questions with known answers and the table/page they come from).

Usage:  python make_table_torture.py OUTPUT_DIR
Writes  OUTPUT_DIR/kestrel_annual_report_2026.pdf  and  OUTPUT_DIR/answer_key.json
All companies, people and numbers are invented.
"""
import io
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image as RLImage,
                                KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

random.seed(2026)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "kestrel_annual_report_2026.pdf"

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerif", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"))

SS = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=SS["BodyText"], fontName="DejaVu", fontSize=9.5, leading=13)
CAP = ParagraphStyle("cap", parent=BODY, fontName="DejaVu-Bold", fontSize=10, spaceBefore=8, spaceAfter=4)
CAP_BELOW = ParagraphStyle("capb", parent=CAP, spaceBefore=3, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontName="DejaVu-Bold", fontSize=16, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontName="DejaVu-Bold", fontSize=12.5, spaceBefore=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)
CELL_R = ParagraphStyle("cellr", parent=CELL, alignment=TA_RIGHT)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#333333"))

FACTS: list[dict] = []        # questions with known answers
TABLES: dict[str, dict] = {}  # table id -> caption marker + challenge description


def fact(table, question, answer, kind="lookup", note=""):
    FACTS.append({"table": table, "question": question, "answer": answer, "kind": kind,
                  **({"note": note} if note else {})})


def caption(tid, title, challenge, below=False):
    """Register a table and return its caption paragraph. The marker 'Table N:' is used to find pages."""
    marker = f"Table {tid}:"
    TABLES[tid] = {"caption": f"{marker} {title}", "challenge": challenge}
    return Paragraph(f"{marker} {title}", CAP_BELOW if below else CAP)


def money(v):
    """Accounting style: negatives in parentheses, thousands separators."""
    return f"({abs(v):,})" if v < 0 else f"{v:,}"


def P(text, style=CELL):
    return Paragraph(text, style)


GRID = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 7.5)]
HEAD = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold")]


# ---------------------------------------------------------------- page furniture -------------
def furniture(canv, doc):
    w, h = canv._pagesize
    canv.saveState()
    # running header and footer (should be ignored by the parser)
    canv.setFont("DejaVu", 7.5)
    canv.setFillColor(colors.HexColor("#555555"))
    canv.drawString(1.5 * cm, h - 1.0 * cm, "Kestrel Freight (Pvt.) Ltd. — Annual Report FY2026 — fictional test document")
    canv.drawRightString(w - 1.5 * cm, 0.8 * cm, f"Page {doc.page}")
    canv.drawString(1.5 * cm, 0.8 * cm, "KF/AR/2026 rev. 3")
    # diagonal watermark over the content (text in the content stream, overlaps tables)
    canv.setFillColor(colors.Color(0.6, 0.6, 0.6, alpha=0.18))
    canv.setFont("DejaVu-Bold", 54)
    canv.translate(w / 2, h / 2)
    canv.rotate(35)
    canv.drawCentredString(0, 0, "DRAFT — CONFIDENTIAL")
    canv.restoreState()


def make_doc():
    doc = BaseDocTemplate(str(PDF), pagesize=A4, title="Kestrel Freight Annual Report FY2026",
                          author="Test generator", leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                          topMargin=1.6 * cm, bottomMargin=1.5 * cm)
    pw, ph = A4
    portrait = Frame(1.6 * cm, 1.5 * cm, pw - 3.2 * cm, ph - 3.1 * cm, id="p")
    lw, lh = landscape(A4)
    land = Frame(1.6 * cm, 1.5 * cm, lw - 3.2 * cm, lh - 3.1 * cm, id="l")
    colw = (pw - 3.2 * cm - 0.8 * cm) / 2
    two = [Frame(1.6 * cm, 1.5 * cm, colw, ph - 3.1 * cm, id="c1"),
           Frame(1.6 * cm + colw + 0.8 * cm, 1.5 * cm, colw, ph - 3.1 * cm, id="c2")]
    doc.addPageTemplates([
        PageTemplate("P", [portrait], onPage=furniture, pagesize=A4),
        PageTemplate("L", [land], onPage=furniture, pagesize=landscape(A4)),
        PageTemplate("TWO", two, onPage=furniture, pagesize=A4),
    ])
    return doc


# ---------------------------------------------------------------- helper flowables -----------
class Rotated(Flowable):
    """Draw a flowable rotated 90 degrees (a sideways table on a portrait page)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def wrap(self, aw, ah):
        w, h = self.inner.wrap(ah, aw)
        self.iw, self.ih = w, h
        return h, w

    def draw(self):
        self.canv.saveState()
        self.canv.rotate(90)
        self.inner.drawOn(self.canv, 0, -self.ih)
        self.canv.restoreState()


class VText(Flowable):
    """Vertical (rotated) header text inside a table cell."""

    def __init__(self, text, size=7):
        super().__init__()
        self.text, self.size = text, size

    def wrap(self, aw, ah):
        self.w = self.size + 2
        self.h = pdfmetrics.stringWidth(self.text, "DejaVu", self.size) + 4
        return self.w, self.h

    def draw(self):
        self.canv.saveState()
        self.canv.setFont("DejaVu", self.size)
        self.canv.translate(self.size, 2)
        self.canv.rotate(90)
        self.canv.drawString(0, 0, self.text)
        self.canv.restoreState()


# ---------------------------------------------------------------- data -----------------------
REGIONS = {
    "Punjab": ["Lahore Central", "Lahore Ring Rd", "Faisalabad", "Multan", "Rawalpindi"],
    "Sindh": ["Karachi Port", "Karachi SITE", "Hyderabad", "Sukkur", "Larkana"],
    "Khyber Pakhtunkhwa": ["Peshawar", "Mardan", "Abbottabad", "Swat", "D.I. Khan"],
    "Balochistan": ["Quetta", "Gwadar", "Turbat", "Khuzdar", "Zhob"],
    "Gilgit-Baltistan": ["Gilgit", "Skardu", "Hunza", "Ghizer", "Astore"],
    "Azad Kashmir": ["Muzaffarabad", "Mirpur", "Kotli", "Bagh", "Rawalakot"],
}
CODES = {}
_n = 100
for reg, brs in REGIONS.items():
    for b in brs:
        _n += 7
        CODES[b] = f"{reg[:2].upper()}-{_n}"


# ---------------------------------------------------------------- sections -------------------
TOC_IDS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16a", "17", "18", "Errata"]


def cover_and_toc(story, toc_pages):
    story += [Spacer(1, 3 * cm),
              Paragraph("Kestrel Freight (Pvt.) Ltd.", H1),
              Paragraph("Annual Report FY2026 — Operations and Finance Annex", H2),
              Spacer(1, 0.5 * cm),
              Paragraph("This document is entirely fictional. It was generated to stress-test table "
                        "extraction in a document ingestion pipeline. Figures are in PKR thousands "
                        "unless a table states otherwise.", BODY),
              Spacer(1, 1 * cm), Paragraph("Contents", H2)]
    # A table of contents with dot leaders: looks like a 2-column table but is not one
    toc = ["Executive summary", "1. Branch financials", "2. Shipment log", "3. Regional half-year results",
           "4. Rates and indices", "5. Service levels", "6. Fleet", "7. Customers and suppliers",
           "8. Headcount", "9. Fuel consumption", "10. Approvals (scanned)", "11. Budget variance",
           "12. Training", "13. Project status", "14. Vendor form", "15. Depot throughput",
           "16. Depot detail", "17. Branch managers", "18. Two-column commentary", "Errata"]
    for i, t in enumerate(toc):
        dots = "." * max(4, 95 - len(t))
        pg = toc_pages.get(TOC_IDS[i], "?")
        story.append(Paragraph(f"{t} {dots} {pg}", ParagraphStyle("toc", parent=BODY, fontSize=9)))
    story.append(PageBreak())


def exec_summary(story, totals):
    story.append(Paragraph("Executive summary", H1))
    story.append(Paragraph(
        "Group revenue grew in every region except Gilgit-Baltistan. The headline figures below are "
        "taken from Table 1 (branch financials). Note the errata on the final page: one figure in this "
        "summary was printed before the final audit adjustment.", BODY))
    wrong_total = totals["revenue"] + 1_250   # deliberately inconsistent with Table 1 (see errata)
    t = Table([["Measure", "FY2026", "Source"],
               ["Group revenue (PKR '000)", f"{wrong_total:,}", "Table 1, grand total"],
               ["Group cost (PKR '000)", f"{totals['cost']:,}", "Table 1, grand total"],
               ["Branches", "30", "Table 1"],
               ["Fleet size (vehicles)", "see Table 6", "—"]], colWidths=[6 * cm, 3.5 * cm, 5 * cm])
    t.setStyle(TableStyle(GRID + HEAD))
    story += [Spacer(1, 6), caption("0", "Headline figures (unaudited)",
                                    "Small summary table whose revenue figure CONFLICTS with Table 1 "
                                    "grand total; the errata page gives the correct value."), t]
    fact("0", "What group revenue does the executive summary report for FY2026?", f"{wrong_total:,}",
         "conflict", "Differs from Table 1 grand total; errata says Table 1 is correct.")
    fact("Errata", "What is the correct (audited) group revenue for FY2026?", f"{totals['revenue']:,}",
         "conflict", "Requires the errata page plus Table 1, not the executive summary.")
    story.append(PageBreak())


def table1_branch_financials(story):
    """Long multi-page table, header repeated, grouped rows with subtotals, accounting negatives."""
    story.append(Paragraph("1. Branch financials", H1))
    story.append(Paragraph(
        "Table 1 lists quarterly revenue and cost for all 30 branches, grouped by region with a "
        "subtotal after each region. Margins in parentheses are losses. Totals carry forward to "
        "Table 0 and Table 11.", BODY))
    head = ["Code", "Branch", "Qtr", "Revenue", "Cost", "Margin", "Margin %"]
    rows = [head]
    style = list(GRID + HEAD)
    tot_r = tot_c = 0
    for reg, branches in REGIONS.items():
        sub_r = sub_c = 0
        rows.append([f"Region: {reg}", "", "", "", "", "", ""])
        style += [("SPAN", (0, len(rows) - 1), (-1, len(rows) - 1)),
                  ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#F2F2F2")),
                  ("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "DejaVu-Bold")]
        for b in branches:
            for q in ("Q1", "Q2", "Q3", "Q4"):
                r = random.randint(2_000, 48_000)
                c = int(r * random.uniform(0.7, 1.12))
                m = r - c
                sub_r += r; sub_c += c
                rows.append([CODES[b] if q == "Q1" else "", "   " + b if q == "Q1" else "", q,
                             f"{r:,}", f"{c:,}", money(m), f"{m / r * 100:.1f}%"])
                if b == "Hyderabad" and q == "Q3":
                    fact("1", "What was Hyderabad branch revenue in Q3 FY2026 (PKR '000)?", f"{r:,}",
                         "lookup", "Branch name appears only on the Q1 row; Q3 row must inherit it.")
                if b == "Gwadar" and q == "Q4":
                    fact("1", "What was Gwadar's Q4 margin?", money(m), "lookup",
                         "Negative values are printed in parentheses." if m < 0 else "")
                if b == "Swat" and q == "Q2":
                    fact("1", "What is the branch code of the branch whose Q2 cost was "
                              f"{c:,}?", CODES[b], "reverse-lookup", "Code is on the Q1 row only.")
        tot_r += sub_r; tot_c += sub_c
        rows.append(["", f"Subtotal {reg}", "", f"{sub_r:,}", f"{sub_c:,}", money(sub_r - sub_c),
                     f"{(sub_r - sub_c) / sub_r * 100:.1f}%"])
        style += [("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "DejaVu-Bold"),
                  ("LINEABOVE", (0, len(rows) - 1), (-1, len(rows) - 1), 1, colors.black)]
        if reg == "Sindh":
            fact("1", "What is the Sindh regional subtotal revenue?", f"{sub_r:,}", "aggregate")
    rows.append(["", "GRAND TOTAL", "", f"{tot_r:,}", f"{tot_c:,}", money(tot_r - tot_c),
                 f"{(tot_r - tot_c) / tot_r * 100:.1f}%"])
    style += [("FONTNAME", (0, -1), (-1, -1), "DejaVu-Bold"),
              ("LINEABOVE", (0, -1), (-1, -1), 1.5, colors.black),
              ("ALIGN", (3, 1), (-1, -1), "RIGHT")]
    t = Table(rows, repeatRows=1, colWidths=[1.8 * cm, 3.8 * cm, 1 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2 * cm])
    t.setStyle(TableStyle(style))
    story += [caption("1", "Quarterly revenue and cost by branch, FY2026 (PKR '000)",
                      "~150 rows over several pages; header repeats on each page; region group rows "
                      "span all columns; branch name/code only on the first of four rows (implicit "
                      "row-span); subtotal rows; accounting-style negatives in parentheses; watermark "
                      "text drawn over the table."), t]
    fact("1", "What is the FY2026 grand total revenue in Table 1?", f"{tot_r:,}", "aggregate")
    story.append(PageBreak())
    return {"revenue": tot_r, "cost": tot_c}


def table2_shipment_log(story):
    """Long table without repeated header, symbols, wrapped notes."""
    story.append(Paragraph("2. Shipment log (sample)", H1))
    story.append(Paragraph("The header row of Table 2 is printed only once; continuation pages have no "
                           "header. Status symbols: ✓ delivered, ✗ lost, ↺ returned, → in transit.", BODY))
    brs = list(CODES)
    rows = [["Shipment", "Date", "Origin", "Destination", "Weight (kg)", "Status", "Notes"]]
    notes = ["", "", "", "Consignee refused delivery; goods returned to origin depot after 3 attempts.",
             "Fragile", "", "Customs hold at Torkham, released after inspection of seals.", "",
             "Temperature-controlled (2–8 °C).", ""]
    for i in range(1, 96):
        o, d = random.sample(brs, 2)
        st = random.choice("✓✓✓✓✗↺→")
        w = round(random.uniform(0.5, 9_800), 1)
        sid = f"KF-26-{4000 + i * 13}"
        note = random.choice(notes)
        rows.append([sid, f"2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}", o, d,
                     f"{w:,.1f}", st, P(note)])
        if i == 77:
            fact("2", f"What is the destination of shipment {sid}?", d, "lookup",
                 "On a continuation page that has no header row.")
            fact("2", f"What is the status symbol of shipment {sid} and what does it mean?", st,
                 "lookup", "Meaning is defined only in the paragraph above the table.")
    t = Table(rows, repeatRows=0, colWidths=[2.2 * cm, 2 * cm, 2.7 * cm, 2.7 * cm, 1.9 * cm, 1.2 * cm, 5 * cm])
    t.setStyle(TableStyle(GRID + HEAD + [("ALIGN", (4, 1), (5, -1), "CENTER")]))
    story += [caption("2", "Shipment log, sample of 95 consignments",
                      "Multi-page table whose header is NOT repeated on continuation pages; Unicode "
                      "status symbols; long wrapped text in the last column; random row heights."), t]


def table3_multilevel(story, values):
    """Landscape page, 3-level column header, row-spanned region groups."""
    story += [NextPageTemplate("L"), PageBreak()]
    story.append(Paragraph("3. Regional half-year results", H1))
    head1 = ["Zone", "Region", "FY2025", "", "", "", "FY2026", "", "", ""]
    head2 = ["", "", "H1", "", "H2", "", "H1", "", "H2", ""]
    head3 = ["", "", "Revenue", "Cost", "Revenue", "Cost", "Revenue", "Cost", "Revenue", "Cost"]
    zones = {"North": ["Punjab", "Khyber Pakhtunkhwa", "Gilgit-Baltistan", "Azad Kashmir"],
             "South": ["Sindh", "Balochistan"]}
    rows = [head1, head2, head3]
    style = list(GRID) + [
        ("SPAN", (2, 0), (5, 0)), ("SPAN", (6, 0), (9, 0)),
        ("SPAN", (2, 1), (3, 1)), ("SPAN", (4, 1), (5, 1)), ("SPAN", (6, 1), (7, 1)), ("SPAN", (8, 1), (9, 1)),
        ("SPAN", (0, 0), (0, 2)), ("SPAN", (1, 0), (1, 2)),
        ("BACKGROUND", (0, 0), (-1, 2), colors.HexColor("#DCE6F1")),
        ("FONTNAME", (0, 0), (-1, 2), "DejaVu-Bold"), ("ALIGN", (0, 0), (-1, 2), "CENTER"),
        ("ALIGN", (2, 3), (-1, -1), "RIGHT"), ("FONTSIZE", (0, 0), (-1, -1), 8.5)]
    for zone, regs in zones.items():
        start = len(rows)
        for reg in regs:
            nums = [random.randint(40_000, 260_000) for _ in range(8)]
            values[reg] = nums
            rows.append([zone if reg == regs[0] else "", reg] + [f"{n:,}" for n in nums])
        style.append(("SPAN", (0, start), (0, len(rows) - 1)))
    t = Table(rows, colWidths=[2.2 * cm, 4.2 * cm] + [2.5 * cm] * 8)
    t.setStyle(TableStyle(style))
    story += [caption("3", "Revenue and cost by region and half-year (PKR '000)",
                      "Three-level column header (year > half > measure) with column spans; zone "
                      "column uses vertical row spans; on a landscape page; referenced by Tables 8 and "
                      "the errata."), t]
    s = values["Sindh"]
    fact("3", "What was Sindh's revenue in H2 FY2026 according to Table 3 as printed?", f"{s[6]:,}",
         "multi-level-header", "Needs FY2026 > H2 > Revenue column path.")
    k = values["Khyber Pakhtunkhwa"]
    fact("3", "What was Khyber Pakhtunkhwa's cost in H1 FY2025?", f"{k[1]:,}", "multi-level-header")
    fact("3", "Which zone does Balochistan belong to?", "South", "row-span",
         "Zone label is printed once in a vertically merged cell.")
    story.append(Paragraph("Footnote: H1 = July–December, H2 = January–June (fiscal halves).", NOTE))
    story += [NextPageTemplate("P"), PageBreak()]


def table4_rates(story):
    """Borderless whitespace table, decimal alignment, mixed number formats."""
    story.append(Paragraph("4. Rates and indices", H1))
    story.append(Paragraph("Rows marked (EU) are copied from a European supplier sheet and use comma "
                           "decimals and dot thousands separators.", BODY))
    data = [["Item", "Unit", "FY2025", "FY2026", "Change"],
            ["Diesel, depot price", "PKR/litre", "281.40", "263.95", "−6.2%"],
            ["Petrol, retail", "PKR/litre", "289.10", "272.80", "−5.6%"],
            ["Tyre set, 10-wheeler (EU)", "EUR", "3.418,50", "3.602,75", "+5,4%"],
            ["Spare parts index (EU)", "index", "1.000,00", "1.087,30", "+8,7%"],
            ["USD/PKR average", "PKR", "283.62", "279.05", "−1.6%"],
            ["Toll, M-2 motorway", "PKR/trip", "2,950", "3,400", "+15.3%"],
            ["Insurance premium", "% of value", "0.85", "0.92", "+0.07 pp"],
            ["Driver daily allowance", "PKR", "1,800", "n/a", "—"]]
    t = Table(data, colWidths=[5.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                           ("ALIGN", (2, 0), (-1, -1), "DECIMAL"), ("RIGHTPADDING", (2, 0), (-1, -1), 18),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 1)]))
    story += [caption("4", "Unit rates and indices",
                      "No borders or shading at all (whitespace-only columns); decimal-aligned "
                      "numbers; European number format mixed with Pakistani/US format; unicode minus "
                      "signs; 'n/a' and dash for missing values. Decimal alignment makes the PDF store each "
                      "number as two separate text fragments (e.g. '3' and '.602,75')."), t]
    fact("4", "What was the FY2026 price of a 10-wheeler tyre set in EUR?", "3.602,75 (= 3602.75)",
         "number-format", "European format: dot = thousands, comma = decimals. Stored in the PDF as two "
         "fragments, '3' and '.602,75'.")
    fact("4", "What is the FY2026 driver daily allowance?", "n/a (not available)", "missing-value")
    story.append(Spacer(1, 12))


def table5_rich_cells(story):
    """Cells with paragraphs, bullet lists and a nested table."""
    story.append(Paragraph("5. Service levels", H1))
    nested = Table([["Zone", "Hours"], ["Urban", "24"], ["Rural", "72"], ["Mountain", "120"]],
                   colWidths=[1.8 * cm, 1.3 * cm])
    nested.setStyle(TableStyle(GRID + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
    data = [
        [P("<b>Service</b>"), P("<b>Commitment</b>"), P("<b>Penalty</b>")],
        [P("Express parcel"),
         P("• Pickup within 2 hours of booking<br/>• Delivery next business day in all Tier-1 cities<br/>"
           "• Proof of delivery uploaded within 30 minutes"),
         P("10% of freight charge per day late, capped at 50%")],
        [P("Standard freight"), [P("Maximum transit time by zone:"), Spacer(1, 3), nested],
         P("PKR 1,500 per day late per consignment")],
        [P("Cold chain"),
         P("Temperature kept at 2–8 °C for the whole journey. Excursions above 8 °C for more than "
           "15 minutes must be reported to the customer within one hour, with the logger file."),
         P("Full replacement value of spoiled goods")],
    ]
    t = Table(data, colWidths=[3.2 * cm, 9 * cm, 4.6 * cm])
    t.setStyle(TableStyle(GRID + HEAD + [("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [caption("5", "Service level commitments",
                      "Cells contain bullet lists, multi-sentence paragraphs and a table nested inside "
                      "a cell."), t]
    fact("5", "What is the maximum transit time for standard freight to mountain zones?", "120 hours",
         "nested-table", "Value is inside a table nested in a cell.")
    fact("5", "What is the penalty for late cold chain delivery?", "Full replacement value of spoiled goods",
         "lookup")
    story.append(PageBreak())


def table6_rotated(story):
    """Wide table printed sideways on a portrait page."""
    story.append(Paragraph("6. Fleet", H1))
    story.append(Paragraph("Table 6 is printed sideways (rotated 90°) on a portrait page.", BODY))
    types = ["Prime mover", "10-wheeler", "6-wheeler", "Mazda 3-ton", "Reefer 20ft", "Pickup", "Motorbike"]
    head = ["Type", "Make", "Count", "Avg age (y)", "Owned", "Leased", "Km/litre", "Utilisation",
            "Breakdowns", "Insured", "Tracker", "Depot"]
    rows = [head]
    for ty in types:
        n = random.randint(4, 160)
        owned = random.randint(0, n - 3)
        rows.append([ty, random.choice(["Hino", "Isuzu", "Volvo", "FAW", "Honda", "JAC"]), str(n),
                     f"{random.uniform(1, 14):.1f}", str(owned), str(n - owned),
                     f"{random.uniform(2.2, 38):.1f}", f"{random.randint(41, 97)}%",
                     str(random.randint(0, 40)), random.choice(["Yes", "Yes", "Partial"]),
                     random.choice(["GPS", "GPS+temp", "None"]), random.choice(["KHI", "LHE", "ISB", "PEW"])])
        if ty == "Reefer 20ft":
            fact("6", "How many Reefer 20ft vehicles are leased?", str(n - owned), "rotated-table")
    t = Table(rows, colWidths=[2.6 * cm, 1.6 * cm, 1.3 * cm, 1.8 * cm, 1.4 * cm, 1.4 * cm, 1.7 * cm,
                               1.9 * cm, 2 * cm, 1.6 * cm, 1.8 * cm, 1.4 * cm])
    t.setStyle(TableStyle(GRID + HEAD + [("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [caption("6", "Fleet inventory at 30 June 2026",
                      "Twelve-column table drawn rotated 90 degrees on a portrait page; text runs "
                      "bottom-to-top."), Rotated(t)]
    story.append(PageBreak())


def table7_side_by_side(story):
    story.append(Paragraph("7. Customers and suppliers", H1))
    cust = [["#", "Customer", "Revenue"]] + [[str(i), n, f"{random.randint(9_000, 90_000):,}"] for i, n in
                                             enumerate(["Indus Textiles", "Mehran Foods", "Saba Pharma",
                                                        "Ravi Steel", "Hunza Organics"], 1)]
    sup = [["#", "Supplier", "Spend"]] + [[str(i), n, f"{random.randint(5_000, 70_000):,}"] for i, n in
                                          enumerate(["PSO Fuels", "Hino Pak Parts", "Atlas Tyres",
                                                     "Jubilee Insurance", "TPL Trackers"], 1)]
    tc = Table(cust, colWidths=[0.7 * cm, 4 * cm, 2.4 * cm]); tc.setStyle(TableStyle(GRID + HEAD))
    ts = Table(sup, colWidths=[0.7 * cm, 4 * cm, 2.4 * cm]); ts.setStyle(TableStyle(GRID + HEAD))
    outer = Table([[Paragraph("Table 7a: Top 5 customers", CAP), Paragraph("Table 7b: Top 5 suppliers", CAP)],
                   [tc, ts]], colWidths=[8.4 * cm, 8.4 * cm])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    TABLES["7"] = {"caption": "Table 7a / 7b: Top 5 customers / Top 5 suppliers",
                   "challenge": "Two separate tables placed side by side with no gap row; the "
                                "detector may merge them into one 6-column table and mix customers "
                                "with suppliers."}
    story += [Paragraph("Table 7: Top customers and suppliers (two tables, side by side)", CAP), outer]
    fact("7", "Who is the number 3 supplier?", sup[3][1], "side-by-side")
    fact("7", "What revenue did Mehran Foods generate?", cust[2][2], "side-by-side")
    story.append(Spacer(1, 14))


def table8_footnotes(story, t3):
    story.append(Paragraph("8. Headcount", H1))
    data = [["Region", P("<b>Permanent<super>a</super></b>"), P("<b>Contract<super>b</super></b>"),
             P("<b>Drivers*</b>"), "Notes"],
            ["Punjab", "412", "88", "301", P("Includes Lahore HQ<super>c</super>")],
            ["Sindh", "389", "120", "276", P("Revenue per head: see Table 3, FY2026")],
            ["Khyber Pakhtunkhwa", "171", "64", "133", ""],
            ["Balochistan", "96", "51", "82", P("Gwadar opened mid-year<super>†</super>")],
            ["Gilgit-Baltistan", "38", "19", "29", ""],
            ["Azad Kashmir", "44", "12", "31", ""],
            ["Total", "1,150", "354", "852", ""]]
    t = Table(data, colWidths=[3.6 * cm, 2.6 * cm, 2.6 * cm, 2 * cm, 6 * cm])
    t.setStyle(TableStyle(GRID + HEAD + [("FONTNAME", (0, -1), (-1, -1), "DejaVu-Bold"),
                                          ("ALIGN", (1, 1), (3, -1), "RIGHT")]))
    story += [caption("8", "Headcount at year end",
                      "Footnote markers (a, b, c, *, †) inside header and body cells with footnotes "
                      "printed below the table; a cell that refers to another table."), t,
              Paragraph("a Employees on indefinite contracts. b Fixed-term and agency staff. "
                        "c Lahore HQ has 63 permanent staff. * Drivers are counted within both "
                        "Permanent and Contract and must not be added to them. † Gwadar headcount "
                        "covers 1 January to 30 June 2026 only.", NOTE)]
    fact("8", "Should drivers be added to permanent and contract staff to get total headcount?",
         "No — drivers are already included in Permanent and Contract (footnote *)", "footnote")
    fact("8", "How many permanent staff work at Lahore HQ?", "63", "footnote")
    sindh_rev = t3["Sindh"][6] + t3["Sindh"][4]   # FY2026 H1 + H2 revenue
    fact("8+3", "What is Sindh FY2026 revenue per permanent employee (PKR '000)? Use Table 3 FY2026 H1+H2 "
                "revenue and Table 8 permanent headcount.",
         f"{sindh_rev / 389:,.1f}", "cross-table",
         "Requires two tables; uses the H2 figure as printed (errata corrects it — see errata question).")
    story.append(PageBreak())


def table9_split_horizontal(story):
    """Too wide: months split across two pages; second half repeats only row numbers."""
    story.append(Paragraph("9. Fuel consumption", H1))
    depots = ["Karachi Port", "Karachi SITE", "Lahore Central", "Faisalabad", "Multan", "Peshawar",
              "Quetta", "Gilgit", "Muzaffarabad", "Hyderabad"]
    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    vals = {d: [random.randint(8_000, 95_000) for _ in months] for d in depots}
    rows1 = [["#", "Depot"] + months[:6]] + [[str(i), d] + [f"{v:,}" for v in vals[d][:6]]
                                               for i, d in enumerate(depots, 1)]
    t1 = Table(rows1, colWidths=[0.8 * cm, 3.6 * cm] + [2.05 * cm] * 6)
    t1.setStyle(TableStyle(GRID + HEAD + [("ALIGN", (2, 1), (-1, -1), "RIGHT")]))
    story += [caption("9", "Diesel consumption by depot, litres (July–December)",
                      "A table too wide for the page, split horizontally: July–December on one page, "
                      "January–June on the next page with only row numbers repeated (no depot names). "
                      "Rows must be re-joined by '#'."), t1, PageBreak()]
    rows2 = [["#"] + months[6:]] + [[str(i)] + [f"{v:,}" for v in vals[d][6:]] for i, d in enumerate(depots, 1)]
    t2 = Table(rows2, colWidths=[0.8 * cm] + [2.4 * cm] * 6)
    t2.setStyle(TableStyle(GRID + HEAD + [("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    story += [Paragraph("Table 9 (continued): January–June. Row numbers match the first part.", CAP), t2]
    fact("9", "How many litres of diesel did the Quetta depot use in March?", f"{vals['Quetta'][8]:,}",
         "split-table", "Quetta is row 7; March is on the continuation page where depot names are absent.")
    story.append(Spacer(1, 14))


def table10_scanned(story):
    """A table that exists only as a skewed, noisy image."""
    story.append(Paragraph("10. Approvals (scanned)", H1))
    rows = [["Ref", "Approved by", "Amount (PKR)", "Date", "Signature"],
            ["AP-311", "S. Qureshi", "4,750,000", "12-Feb-2026", "signed"],
            ["AP-312", "M. Baloch", "980,500", "19-Feb-2026", "signed"],
            ["AP-317", "R. Khattak", "12,300,000", "03-Mar-2026", "pending"],
            ["AP-322", "A. Siddiqui", "615,250", "28-Mar-2026", "signed"]]
    W, Hh = 1500, 520
    img = Image.new("L", (W, Hh), 250)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 30)
    xs = [40, 240, 560, 860, 1140, 1460]
    for r, row in enumerate(rows):
        y = 40 + r * 85
        for c, cell in enumerate(row):
            d.text((xs[c] + 14, y + 22), cell, fill=20, font=f)
    for r in range(len(rows) + 1):
        d.line([(xs[0], 40 + r * 85), (xs[-1], 40 + r * 85)], fill=40, width=3)
    for x in xs:
        d.line([(x, 40), (x, 40 + len(rows) * 85)], fill=40, width=3)
    img = img.rotate(1.6, resample=Image.BICUBIC, fillcolor=250, expand=True)
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    px = img.load()
    for _ in range(25_000):
        x, y = random.randrange(img.width), random.randrange(img.height)
        px[x, y] = random.choice((60, 120, 200))
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=55); buf.seek(0)
    story += [caption("10", "Approval register, scanned from the signed paper copy",
                      "Table exists only as a low-quality, slightly rotated JPEG image: needs OCR plus "
                      "table-structure recognition on noisy pixels."),
              RLImage(buf, width=16.5 * cm, height=16.5 * cm * img.height / img.width)]
    fact("10", "Who approved AP-317 and for what amount?", "R. Khattak, PKR 12,300,000", "scanned-table")
    story.append(PageBreak())


def table11_variance(story, totals):
    story.append(Paragraph("11. Budget variance", H1))
    budget = totals["revenue"] - 37_500
    data = [["Line item", "Budget", "Actual", "Variance", "Var %", "Comment"],
            ["Revenue (from Table 1)", f"{budget:,}", f"{totals['revenue']:,}",
             money(totals['revenue'] - budget), f"{(totals['revenue'] - budget) / budget * 100:.1f}%", ""],
            ["Fuel", "1,940,000", "2,012,450", "(72,450)", "−3.7%", P("Diesel price fell, volume rose")],
            ["Tolls", "212,000", "", "", "", P("Not yet reconciled")],
            ["Insurance", "180,500", "180,500", "—", "0.0%", ""],
            ["Rent", "nil", "nil", "—", "n/a", ""],
            ["Depreciation", "—", "455,120", "", "", P("Not budgeted; new policy")],
            ["IT systems", "95,000", "131,700", "(36,700)", "−38.6%", P("Tracker upgrade (see Table 6)")]]
    t = Table(data, colWidths=[4 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 1.6 * cm, 4.2 * cm])
    t.setStyle(TableStyle(GRID + HEAD + [("ALIGN", (1, 1), (4, -1), "RIGHT")]))
    story += [caption("11", "Budget versus actual, FY2026 (PKR '000)",
                      "Empty cells, 'nil', em dashes, 'n/a', accounting negatives, a percentage "
                      "column with unicode minus; one row pulls its value from Table 1."), t]
    fact("11", "What is the actual Tolls expense in FY2026?", "Not reported (cell empty; not yet reconciled)",
         "missing-value", "A good pipeline should NOT invent a number here.")
    fact("11", "What was the IT systems variance?", "(36,700), i.e. 36,700 over budget", "lookup")
    story.append(Spacer(1, 10))


def table12_training(story):
    story.append(Paragraph("12. Training completion", H1))
    courses = ["Defensive driving", "Hazmat handling", "Cold chain basics", "First aid",
               "Anti-bribery policy", "Customs documentation", "Forklift safety"]
    people = ["A. Rehman", "B. Khan", "C. Fatima", "D. Ali", "E. Shah", "F. Noor", "G. Iqbal", "H. Javed"]
    rows = [[P("<b>Employee</b>")] + [VText(c) for c in courses]]
    for p in people:
        rows.append([p] + [random.choice(["✓", "✓", "✓", "✗", "68%", "—"]) for _ in courses])
    t = Table(rows, colWidths=[3.2 * cm] + [1.3 * cm] * len(courses))
    style = [("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 8),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black)]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EEF3F8")))
    t.setStyle(TableStyle(style))
    story += [caption("12", "Mandatory training completion by employee",
                      "Column headers are vertical (rotated) text; no grid lines, only zebra shading; "
                      "cells hold symbols and partial percentages."), t]
    fact("12", "Has C. Fatima completed Hazmat handling?", rows[3][2], "vertical-header",
         "Answer is the symbol/value in the C. Fatima row, Hazmat handling column.")
    story.append(PageBreak())


def table13_colour_only(story):
    story.append(Paragraph("13. Project status", H1))
    projects = ["Gwadar depot build", "Tracker upgrade", "ERP migration", "Cold-chain fleet",
                "Driver app", "Karachi warehouse"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    palette = {"G": colors.HexColor("#4CAF50"), "A": colors.HexColor("#FFC107"), "R": colors.HexColor("#E53935")}
    rows = [["Project"] + months]
    style = list(GRID + HEAD)
    status = {}
    for i, p in enumerate(projects, 1):
        rows.append([p] + [""] * len(months))
        status[p] = [random.choice("GGGAAR") for _ in months]
        for j, s in enumerate(status[p], 1):
            style.append(("BACKGROUND", (j, i), (j, i), palette[s]))
    t = Table(rows, colWidths=[4.5 * cm] + [1.8 * cm] * 6)
    t.setStyle(TableStyle(style))
    story += [caption("13", "Monthly RAG status by project",
                      "Status is encoded ONLY by cell colour (no text in cells); meaning is in a "
                      "legend below. Text extraction sees an empty grid."), t,
              Paragraph("Legend: green = on track, amber = at risk, red = off track.", NOTE)]
    names = {"G": "green (on track)", "A": "amber (at risk)", "R": "red (off track)"}
    fact("13", "What was the status of the ERP migration in April?", names[status["ERP migration"][3]],
         "colour-only", "Only recoverable from cell colour plus the legend.")
    story.append(Spacer(1, 12))


def table14_form(story):
    story.append(Paragraph("14. Vendor registration form", H1))
    data = [["Vendor name", "Atlas Tyres (Pvt.) Ltd.", "NTN", "4410287-6"],
            ["Address", P("Plot 17, Sector 25, Korangi Industrial Area, Karachi"), "STRN", "17-00-4410-287-11"],
            ["Contact", "Ms. Hina Akhtar", "Phone", "+92 21 3507 0000"],
            ["Bank", "Meezan Bank, Korangi branch", "IBAN", "PK00 MEZN 0000 0000 0000 0000"],
            ["Approved", "☑ Yes   ☐ No", "Date", "14-Jan-2026"]]
    t = Table(data, colWidths=[3 * cm, 6.5 * cm, 2 * cm, 5.3 * cm])
    t.setStyle(TableStyle(GRID + [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
                                  ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F2F2F2")),
                                  ("FONTNAME", (0, 0), (0, -1), "DejaVu-Bold"),
                                  ("FONTNAME", (2, 0), (2, -1), "DejaVu-Bold")]))
    story += [caption("14", "Vendor registration (form)",
                      "A key-value FORM laid out as a grid: two key/value pairs per row. Treating it as "
                      "a normal table with a header row gives wrong column semantics; checkboxes as "
                      "unicode symbols."), t]
    fact("14", "What is Atlas Tyres' NTN?", "4410287-6", "form")
    fact("14", "Was the vendor approved?", "Yes (☑ ticked)", "form")


def table15_dense(story):
    story += [NextPageTemplate("L"), PageBreak()]
    story.append(Paragraph("15. Depot throughput (daily, June 2026)", H1))
    cols = ["Day", "KHI-P", "KHI-S", "HYD", "SUK", "LHE-C", "LHE-R", "FSD", "MUL", "RWP", "PEW", "QTA",
            "GWD", "Total"]
    rows = [cols]
    for d in range(1, 31):
        v = [random.randint(80, 2_400) for _ in cols[1:-1]]
        rows.append([f"{d:02d}-Jun"] + [f"{x:,}" for x in v] + [f"{sum(v):,}"])
        if d == 23:
            fact("15", "How many consignments did the Multan (MUL) depot handle on 23 June?", f"{v[7]:,}",
                 "dense-table")
    t = Table(rows, colWidths=[1.6 * cm] + [1.55 * cm] * 12 + [1.8 * cm], rowHeights=0.42 * cm)
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 5.5),
                           ("GRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
                           ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    story += [caption("15", "Consignments handled per depot per day",
                      "14 columns × 31 rows in 5.5 pt font with faint hairline grid; depot names are "
                      "abbreviations explained nowhere in the table."), t]
    story += [NextPageTemplate("P"), PageBreak()]


def table16_back_to_back(story):
    story.append(Paragraph("16. Depot detail", H1))
    def depot_table(seed):
        random.seed(seed)
        rows = [["Bay", "Capacity (pallets)", "Used", "Free"]]
        for b in range(1, 7):
            cap = random.randint(40, 300); used = random.randint(0, cap)
            rows.append([f"B{b}", str(cap), str(used), str(cap - used)])
        t = Table(rows, colWidths=[2 * cm, 4 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle(GRID + HEAD))
        return t, rows
    ta, ra = depot_table(11)
    tb, rb = depot_table(12)
    random.seed(99)
    story += [caption("16a", "Karachi Port depot bays",
                      "Two tables with identical columns placed directly after each other, separated "
                      "only by a one-line caption; they may be merged into one table and lose which "
                      "depot each row belongs to."), ta,
              caption("16b", "Lahore Central depot bays", "(second half of the 16a/16b pair)"), tb]
    fact("16b", "How many pallet spaces are free in bay B4 of the Lahore Central depot?", rb[4][3],
         "back-to-back", "Karachi's table has a B4 row too; mixing them gives a wrong answer.")
    story.append(Spacer(1, 12))


def table17_urdu(story):
    story.append(Paragraph("17. Branch managers", H1))
    import arabic_reshaper
    from bidi.algorithm import get_display
    ur = lambda s: Paragraph(get_display(arabic_reshaper.reshape(s)),
                             ParagraphStyle("ur", parent=CELL, fontName="FreeSerif", fontSize=10,
                                            alignment=TA_RIGHT))
    data = [["Code", "Manager (English)", "Manager (Urdu)", "Branch", "Mobile"],
            [CODES["Multan"], "Tariq Mehmood", ur("طارق محمود"), "Multan", "0300-1234567"],
            [CODES["Sukkur"], "Saima Jatoi", ur("صائمہ جتوئی"), "Sukkur", "0301-7654321"],
            [CODES["Skardu"], "Ali Hussain", ur("علی حسین"), "Skardu", "0345-1112223"],
            [CODES["Mirpur"], "Nadia Chaudhry", ur("نادیہ چوہدری"), "Mirpur", "0333-9998887"]]
    t = Table(data, colWidths=[2 * cm, 3.6 * cm, 3.8 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle(GRID + HEAD))
    story += [t, caption("17", "Branch managers (caption printed BELOW the table)",
                         "Mixed English and Urdu (right-to-left) columns; caption placed below the "
                         "table instead of above.", below=True)]
    fact("17", "Who manages the Skardu branch?", "Ali Hussain", "rtl-mixed")


def section18_two_column(story, t3):
    story += [NextPageTemplate("TWO"), PageBreak()]
    story.append(Paragraph("18. Commentary", H1))
    txt = ("Regional managers reported that demand in the North zone was driven by textile exports "
           "and cement. As Table 3 shows, both halves of FY2026 exceeded FY2025 in most regions. ")
    for _ in range(3):
        story.append(Paragraph(txt * 3, BODY))
    small = Table([["KPI", "Value"], ["On-time %", "91.4"], ["Claims ratio", "0.37%"],
                   ["NPS", "+42"]], colWidths=[3.2 * cm, 2.6 * cm])
    small.setStyle(TableStyle(GRID + HEAD))
    story += [caption("18", "Customer KPIs (inside a two-column text layout)",
                      "Small table embedded in the first column of a two-column page; reading order "
                      "must not mix the table with the neighbouring column's text."), small]
    for _ in range(5):
        story.append(Paragraph(txt * 3, BODY))
    fact("18", "What was the claims ratio?", "0.37%", "two-column-layout")
    story += [NextPageTemplate("P"), PageBreak()]


def errata(story, totals, t3):
    story.append(Paragraph("Errata", H1))
    corrected = t3["Sindh"][6] + 4_321
    story.append(Paragraph(
        f"1. Executive summary: the group revenue figure was printed before the final audit adjustment. "
        f"The correct figure is the Table 1 grand total, PKR {totals['revenue']:,} thousand.", BODY))
    story.append(Paragraph(
        f"2. Table 3: Sindh revenue for FY2026 H2 should read {corrected:,} (printed as "
        f"{t3['Sindh'][6]:,}). All other Table 3 figures are unchanged.", BODY))
    TABLES["Errata"] = {"caption": "Errata", "challenge": "Prose corrections that override table values "
                                                         "printed earlier in the document."}
    fact("Errata+3", "What is the corrected Sindh revenue for FY2026 H2?", f"{corrected:,}", "conflict",
         "Table 3 prints a different value; the errata page overrides it.")


def main(toc_pages=None):
    random.seed(2026)
    FACTS.clear(); TABLES.clear()
    doc = make_doc()
    story = []
    cover_and_toc(story, toc_pages or {})
    # Table 1 must be generated before the summary to know totals; build into a temporary list
    t1_story: list = []
    totals = table1_branch_financials(t1_story)
    exec_summary(story, totals)
    story += t1_story
    table2_shipment_log(story)
    t3: dict = {}
    table3_multilevel(story, t3)
    table4_rates(story)
    table5_rich_cells(story)
    table6_rotated(story)
    table7_side_by_side(story)
    table8_footnotes(story, t3)
    table9_split_horizontal(story)
    table10_scanned(story)
    table11_variance(story, totals)
    table12_training(story)
    table13_colour_only(story)
    table14_form(story)
    table15_dense(story)
    table16_back_to_back(story)
    table17_urdu(story)
    section18_two_column(story, t3)
    errata(story, totals, t3)
    doc.build(story)

    # find on which pages each table starts/ends, from the extracted text
    reader = PdfReader(str(PDF))
    page_text = [p.extract_text() or "" for p in reader.pages]
    starts = {}
    for tid, info in TABLES.items():
        marker = info["caption"].split(":")[0] + ":" if tid not in ("Errata", "7") else (
            "Errata" if tid == "Errata" else "Table 7:")
        for i, txt in enumerate(page_text, 1):
            if (tid == "Errata" and i > 3 and txt.lstrip().splitlines()[1:2] and "Errata" in txt
                    and "1. Executive summary" in txt) or (tid != "Errata" and re.search(re.escape(marker), txt)):
                starts[tid] = i
                break
    order = sorted(starts.items(), key=lambda kv: kv[1])
    for k, (tid, p) in enumerate(order):
        nxt = order[k + 1][1] if k + 1 < len(order) else len(page_text) + 1
        TABLES[tid]["pages"] = list(range(p, max(p, nxt - 1) + 1)) if tid in ("1", "2") else ([p, p + 1] if tid == "9" else [p])
    for f in FACTS:
        ids = re.split(r"\+", f["table"])
        f["pages"] = sorted({pg for i in ids for pg in TABLES.get(i, {}).get("pages", [])})
    key = {"document": PDF.name, "pages": len(page_text), "fictional": True,
           "tables": TABLES, "questions": FACTS}
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{PDF}  pages={len(page_text)}  tables={len(TABLES)}  questions={len(FACTS)}")
    return {tid: info["pages"][0] for tid, info in TABLES.items()}


if __name__ == "__main__":
    first = main()          # pass 1 finds where each section lands
    second = main(first)    # pass 2 prints the real page numbers in the contents list
    assert first == second, "page layout changed between passes"
