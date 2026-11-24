"""Tables from the user's documents, copied into files exactly (the copy_table tool): no
model retypes a figure, and every total is worked out here, not by the model.

Seen live: agents retyping a 120-row table lost a whole region (8 branches) and a figure
or two on the way. A stored table comes in pieces (per page, or a few rows each, every piece
repeating the caption and header); they are joined back in reading order. Then, as asked:
keep some columns, pivot a long table into a wide one (one row per branch, a column per
quarter), add a row total, a subtotal per group and a grand total, and compare the totals
the document itself prints with its rows.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

_CAPTION = re.compile(r"^\s*(table|tab\.|exhibit|schedule)\s*([\w.-]*\d[\w.-]*)", re.I)
_TOTAL = re.compile(r"\b(sub-?\s?)?totals?\b", re.I)
MAX_ROWS = 5000


@dataclass
class DocTable:
    name: str                 # "Table 1", or "table on p. 4" when it has no caption
    caption: str              # the caption line as printed ("" if none)
    header: list[str]
    rows: list[list[str]]
    pages: list[int] = field(default_factory=list)

    def describe(self) -> str:
        pages = sorted(set(self.pages))
        where = (f"p. {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}") \
            if pages else "no page"
        totals = sum(1 for r in self.rows if _is_total(r))
        count = f"{len(self.rows) - totals} rows" + (f" + {totals} printed total rows"
                                                     if totals else "")
        title = self.caption if self.caption.lower().startswith(self.name.lower()) else \
            f"{self.name}: {self.caption or '(no caption)'}"
        return f"{title} | {where} | {count} | columns: {' | '.join(self.header)}"


def number(text) -> int | float | None:
    """A cell's number: 1,234.5, (1,234) or -1,234 for negatives, **bold** allowed; None for
    text, percentages and empty cells."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return text
    plain = str(text or "").strip().strip("*").strip().replace("−", "-")
    negative = plain.startswith("(") and plain.endswith(")")
    plain = plain.strip("()").replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", plain):
        return None
    value = float(plain)
    value = -value if negative else value
    return int(value) if "." not in plain else value


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _total_label(cell: str) -> bool:
    """A cell naming a total row: "Total", "Subtotal Sindh", "GRAND TOTAL", "Region total";
    not a note that mentions one ("Table 1, grand total")."""
    text = str(cell).strip(" *")
    return number(text) is None and bool(
        re.match(r"(grand\s+|sub-?\s?)?totals?\b", text, re.I)
        or (_TOTAL.search(text) and len(text.split()) <= 3 and "," not in text))


def _is_total(row: list[str]) -> bool:
    return any(_total_label(c) for c in row)


def tables_in(points: list) -> list[DocTable]:
    """The document's tables, from its stored chunks (Qdrant points with payloads)."""
    pieces = sorted((p for p in points if (p.payload or {}).get("modality") == "table"),
                    key=lambda p: p.payload.get("chunk_index") or 0)
    tables: list[DocTable] = []
    for p in pieces:
        lines = p.payload.get("content", "").split("\n")
        first = lines[0].strip() if lines else ""
        caption = first if first and not first.startswith("|") else ""
        rows = [_cells(line) for line in lines if line.strip().startswith("|")]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
        if not rows:
            continue
        header, body = rows[0], [r for r in rows[1:] if r != rows[0]]
        page = p.payload.get("page_number")
        same = next((t for t in reversed(tables) if t.caption == caption and t.header == header
                     and (caption or (t.pages and page and page - t.pages[-1] <= 1))), None)
        if same is None:
            m = _CAPTION.match(caption)
            name = f"{m.group(1).title()} {m.group(2)}" if m else \
                f"table on p. {page}" if page else f"table {len(tables) + 1}"
            if any(t.name == name for t in tables):
                name = f"{name} ({sum(t.name.startswith(name) for t in tables) + 1})"
            same = DocTable(name, caption, header, [])
            tables.append(same)
        same.rows += body
        if page:
            same.pages.append(page)
    return tables


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find(tables: list[DocTable], wanted: str) -> DocTable:
    """The table called `wanted`: its name ("Table 1", "1") or words from its caption."""
    key = _norm(wanted)
    exact = [t for t in tables if _norm(t.name) in (key, "table" + key)]
    words = [t for t in tables if wanted.strip().lower() in t.caption.lower()]
    found = exact or words
    if len(found) != 1:
        names = "\n".join(t.describe() for t in tables) or "(none)"
        what = "several tables match" if found else "no table matches"
        raise ValueError(f"{what} {wanted!r}. The tables in this document:\n{names}")
    return found[0]


@dataclass
class Shaped:
    header: list[str]
    rows: list[list]                 # numbers as numbers
    kinds: list[str]                 # "data", "subtotal" or "total", per row
    sum_columns: list[int]           # columns added up in totals
    row_total: int | None = None     # the column holding row totals
    groups: list[tuple[int, int, int]] = field(default_factory=list)  # (subtotal row, first, last)
    checks: list[str] = field(default_factory=list)  # the document's printed totals
    notes: list[str] = field(default_factory=list)
    style: dict = field(default_factory=dict)


def _column(header: list[str], name: str) -> int:
    key = _norm(name)
    for i, h in enumerate(header):
        if _norm(h) == key:
            return i
    for i, h in enumerate(header):  # "Revenue" for "Revenue (PKR '000)"
        if key and _norm(h).startswith(key):
            return i
    raise ValueError(f"no column {name!r}; the columns are: {' | '.join(header)}")


def printed_totals(table: DocTable) -> list[str]:
    """Each total row the document prints, compared with the sum of its rows."""
    data = [r for r in table.rows if not _is_total(r)]
    out = []
    for row in (r for r in table.rows if _is_total(r)):
        label = next(c for c in row if _total_label(c))
        group = _TOTAL.sub("", re.sub(r"\bgrand\b", "", label, flags=re.I)).strip(" :-*")
        members = [r for r in data if group and group in r] if group else data
        if group and not members:
            continue
        for c, cell in enumerate(row):
            printed = number(cell)
            if printed is None or c >= len(table.header):
                continue
            values = [number(r[c]) for r in members if c < len(r)]
            if not any(v is not None for v in values):
                continue
            total = sum(v for v in values if v is not None)
            if abs(total - printed) > 0.5 + abs(printed) * 1e-9:
                out.append(f"DIFFERS: {label} {table.header[c]}: printed {cell}, its rows add "
                           f"up to {total:,}")
            else:
                out.append(f"ok: {label} {table.header[c]} {cell}")
    return out


def shape(table: DocTable, columns=(), rows_by=(), across: str = "", values: str = "",
          row_total: str = "", sum_columns=(), subtotals_by: str = "",
          grand_total: bool = False) -> Shaped:
    header = table.header
    adding = bool(row_total or subtotals_by or grand_total)
    # the document's own total rows would be counted twice in totals added here
    source = [r for r in table.rows if not (adding or across) or not _is_total(r)]
    source = [r + [""] * (len(header) - len(r)) for r in source]
    if across:
        if not rows_by or not values:
            raise ValueError("to pivot, give rows_by (the columns naming a row, e.g. "
                             "[\"Region\", \"Branch\"]), across and values")
        keys = [_column(header, n) for n in rows_by]
        pivot, value = _column(header, across), _column(header, values)
        order: dict[tuple, dict] = {}
        heads: list[str] = []
        doubled = 0
        for r in source:
            if not r[pivot].strip():
                continue
            key = tuple(r[k] for k in keys)
            slot = order.setdefault(key, {})
            if r[pivot] not in heads:
                heads.append(r[pivot])
            v = number(r[value])
            if r[pivot] in slot:
                doubled += 1
                slot[r[pivot]] = (number(slot[r[pivot]]) or 0) + (v or 0)
            else:
                slot[r[pivot]] = v if v is not None else r[value]
        out_header = [header[k] for k in keys] + heads
        rows = [list(key) + [slot.get(h, "") for h in heads] for key, slot in order.items()]
        sums = list(range(len(keys), len(out_header)))
        notes = [f"{doubled} rows shared a {header[pivot]}: their values were added"] \
            if doubled else []
    else:
        keep = [_column(header, n) for n in columns] if columns else list(range(len(header)))
        out_header = [header[k] for k in keep]
        rows = [[number(r[k]) if number(r[k]) is not None else r[k] for k in keep]
                for r in source]
        sums = []
        notes = []
    if sum_columns:
        sums = [_column(out_header, n) for n in sum_columns]
    if not sums and adding:
        sums = [c for c in range(len(out_header)) if any(isinstance(r[c], (int, float))
                                                        for r in rows)
                and all(isinstance(r[c], (int, float)) or r[c] == "" for r in rows)]
        if not sums:
            raise ValueError("no number column to add up: say which in sum_columns")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"the table has {len(rows)} rows; at most {MAX_ROWS} are copied")
    total_col = None
    if row_total:
        out_header = out_header + [row_total]
        total_col = len(out_header) - 1
        for r in rows:
            r.append(sum(v for c in sums if isinstance(v := r[c], (int, float))))
    added = sums + ([total_col] if total_col is not None else [])
    kinds = ["data"] * len(rows)
    groups: list[tuple[int, int, int]] = []
    if subtotals_by:
        g = _column(out_header, subtotals_by)
        grouped, grouped_kinds, start = [], [], 0
        for i, r in enumerate(rows):
            grouped.append(r)
            grouped_kinds.append("data")
            if i + 1 == len(rows) or rows[i + 1][g] != r[g]:
                block = rows[start:i + 1]
                sub = ["" for _ in out_header]
                sub[g] = f"Subtotal {r[g]}".strip()
                for c in added:
                    sub[c] = sum(v for b in block if isinstance(v := b[c], (int, float)))
                first = len(grouped) - len(block)
                groups.append((len(grouped), first, len(grouped) - 1))
                grouped.append(sub)
                grouped_kinds.append("subtotal")
                start = i + 1
        rows, kinds = grouped, grouped_kinds
    if grand_total:
        total = ["" for _ in out_header]
        total[0] = "Grand total"
        for c in added:
            total[c] = sum(v for r, k in zip(rows, kinds) if k == "data"
                           and isinstance(v := r[c], (int, float)))
        rows.append(total)
        kinds.append("total")
    numbers = [c for r in table.rows for c in r if number(c) is not None]
    style = {"commas": any("," in c for c in numbers),
             "brackets": any(c.strip("* ").startswith("(") for c in numbers)}
    return Shaped(out_header, rows, kinds, sums, total_col, groups, printed_totals(table),
                  notes, style)


def show(value, style: dict) -> str:
    """A number in the document's own style (thousands commas, (123) for negatives)."""
    if not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and not value.is_integer():
        text = f"{abs(value):,.2f}" if style.get("commas") else f"{abs(value):.2f}"
    else:
        text = f"{abs(int(value)):,}" if style.get("commas") else str(abs(int(value)))
    if value < 0:
        return f"({text})" if style.get("brackets") else f"-{text}"
    return text


def write(path: Path, shaped: Shaped, source_note: str, sheet: str = "", heading: str = "",
          note: str = "") -> str:
    """Write the table into `path` (.xlsx: its own sheet, totals as Excel formulas; .docx:
    added at the end; .md: added at the end; .csv: the whole file)."""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _xlsx(path, shaped, source_note, sheet or "Table", note)
    if suffix == ".docx":
        return _docx(path, shaped, source_note, heading, note)
    lines = [shaped.header] + [[show(v, shaped.style) for v in r] for r in shaped.rows]
    if suffix == ".csv":
        import csv
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([shaped.header] + shaped.rows)
        return "the whole file"
    text = "\n".join(["| " + " | ".join(lines[0]) + " |", "|" + "---|" * len(lines[0])]
                     + ["| " + " | ".join(r) + " |" for r in lines[1:]])
    before = path.read_text(encoding="utf-8") + "\n\n" if path.exists() else ""
    parts = [f"## {heading}" if heading else "", text, f"*{source_note}*", note]
    path.write_text(before + "\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")
    return "added at the end"


def _xlsx(path: Path, shaped: Shaped, source_note: str, sheet: str, note: str) -> str:
    import openpyxl
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    book = openpyxl.load_workbook(path) if path.exists() else openpyxl.Workbook()
    if not path.exists():
        book.remove(book.active)
    title = re.sub(r"[\[\]:*?/\\]", "_", sheet.strip() or "Table")[:31]
    if title in book.sheetnames:
        del book[title]
    ws = book.create_sheet(title)
    ws.append(shaped.header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    letter = lambda c: get_column_letter(c + 1)  # noqa: E731
    data_rows = [i + 2 for i, k in enumerate(shaped.kinds) if k == "data"]
    group_of = {sub: (first, last) for sub, first, last in shaped.groups}
    for i, (row, kind) in enumerate(zip(shaped.rows, shaped.kinds)):
        excel_row = i + 2
        out = list(row)
        if shaped.row_total is not None and kind == "data":
            cells = [f"{letter(c)}{excel_row}" for c in shaped.sum_columns]
            contiguous = shaped.sum_columns == list(range(shaped.sum_columns[0],
                                                          shaped.sum_columns[-1] + 1))
            out[shaped.row_total] = (f"=SUM({cells[0]}:{cells[-1]})" if contiguous
                                     else "=" + "+".join(cells))
        added = shaped.sum_columns + ([shaped.row_total] if shaped.row_total is not None
                                      else [])
        if kind == "subtotal":
            first, last = group_of[i]
            for c in added:
                out[c] = f"=SUM({letter(c)}{first + 2}:{letter(c)}{last + 2})"
        elif kind == "total":
            subs = [s + 2 for s, _, _ in shaped.groups]
            for c in added:
                out[c] = ("=" + "+".join(f"{letter(c)}{s}" for s in subs)) if subs else \
                    f"=SUM({letter(c)}{min(data_rows)}:{letter(c)}{max(data_rows)})"
        ws.append(out)
        if kind != "data":
            for cell in ws[excel_row]:
                cell.font = Font(bold=True)
    ws.append([])
    ws.append([source_note + (f" {note}" if note else "")])
    book.save(path)
    return f"sheet {title!r}, totals as Excel formulas"


def _docx(path: Path, shaped: Shaped, source_note: str, heading: str, note: str) -> str:
    import docx
    document = docx.Document(path) if path.exists() else docx.Document()
    if heading:
        document.add_heading(heading, level=2)
    table = document.add_table(rows=len(shaped.rows) + 1, cols=len(shaped.header))
    table.style = "Table Grid"
    for r, row in enumerate([shaped.header] + shaped.rows):
        bold = r == 0 or shaped.kinds[r - 1] != "data"
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = show(value, shaped.style)
            for run in cell.paragraphs[0].runs:
                run.bold = bold
    document.add_paragraph().add_run(source_note).italic = True
    if note:
        document.add_paragraph(note)
    document.save(path)
    return "added at the end of the document"
