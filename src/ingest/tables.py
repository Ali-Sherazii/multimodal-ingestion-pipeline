"""Tables rebuilt for search from Docling's cell grid (FR-08, FR-09).

Docling's own Markdown copies a merged cell into every column and row it covers, keeps group
rows ("Region: Punjab" across the whole table) and leaves cells that a merged cell above
stands for empty, so a row chunked on its own can lose what it is about ("| | | Q3 | 3,663 |"
- which branch?). Here:

- text that covers several columns is written once; text that covers several rows is
  repeated in each, so every row stands on its own;
- group rows become a column ("Region: Punjab" -> a Region column saying Punjab);
- empty leading cells under a value (a branch named only on its first row) are filled
  from the row above;
- stacked header rows become one ("FY2025 - H1 - Revenue"), as Docling does;
- each table keeps its caption, the sentence introducing it and its footnotes or legend,
  found by position on the page when Docling didn't link them;
- a table continued on the next page is joined to its start: with the same header, without
  a header (its first row is data), or split by columns (the same row numbers, more
  columns); each page's rows keep their page number;
- a table printed sideways is turned upright, cells whose only content is a colour say
  the colour ("amber"), a form laid out as a grid becomes "Key: value" lines, and a ticked
  box says which option is ticked.

The table becomes one segment of its own (per page it covers): caption and introduction,
the Markdown table, then notes. The chunker keeps it whole if it fits, or splits it by rows,
repeating the caption, header and notes in every piece.
"""
import re
from dataclasses import dataclass, field
from statistics import median

HEADER_JOIN = " - "


@dataclass
class Cell:
    text: str
    row: int                 # where the cell starts
    col: int
    rows: int = 1            # how many rows and columns it covers
    cols: int = 1
    header: bool = False


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    caption: str = ""
    intro: str = ""          # the sentence(s) just before it that introduce it
    notes: str = ""          # footnotes, legend or source printed under it
    groups: dict[int, str] = field(default_factory=dict)  # group rows, until finish()
    pages: list[int | None] = field(default_factory=list)  # each row's page

    def markdown(self) -> str:
        line = lambda cells: "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"
        return "\n".join([line(self.header), line(["---"] * len(self.header))]
                         + [line(r) for r in self.rows])

    def text(self) -> str:
        """The table as one segment: caption and introduction, the table, then notes."""
        top = "\n".join(p for p in (self.caption, self.intro) if p)
        notes = self.notes
        if notes and not re.match(r"(notes?|footnotes?|legend|sources?|key)\s*:", notes, re.I):
            notes = f"Notes: {notes}"
        return "\n\n".join(p for p in (top, self.markdown(), notes) if p)


# ---- rebuilding the table ----

def build(cells: list[Cell], num_rows: int, num_cols: int) -> Table | None:
    if not num_rows or not num_cols:
        return None
    grid: list[list[Cell | None]] = [[None] * num_cols for _ in range(num_rows)]
    for cell in cells:
        for r in range(cell.row, min(cell.row + cell.rows, num_rows)):
            for c in range(cell.col, min(cell.col + cell.cols, num_cols)):
                grid[r][c] = cell

    heads = 0  # the leading rows holding header cells (Docling's rule); else the first row
    while heads < num_rows and any(c is not None and c.header for c in grid[heads]):
        heads += 1
    heads = heads or 1
    header = []
    for c in range(num_cols):
        parts: list[str] = []
        for r in range(heads):
            text = grid[r][c].text if grid[r][c] else ""
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        header.append(HEADER_JOIN.join(parts))

    rows: list[list[str]] = []
    groups: dict[int, str] = {}  # body row index -> group label ("Region: Punjab")
    for r in range(heads, num_rows):
        row = []
        for c, cell in enumerate(grid[r]):
            # covering several columns: written once; several rows: repeated in each row
            row.append("" if cell is None or cell.col < c else cell.text)
        filled = [t for t in row if t]
        if not filled:
            continue
        spans_all = [c for c in grid[r] if c is not None and c.text and c.cols >= num_cols]
        if num_cols > 1 and (spans_all or (len(filled) > 1 and len(set(filled)) == 1)):
            groups[len(rows)] = filled[0]
            continue
        rows.append(row)
    if groups and not rows:  # nothing but group rows: they are the content
        rows = [[label] + [""] * (num_cols - 1) for label in groups.values()]
        groups = {}
    return Table(header, rows, groups=groups)


def finish(table: Table) -> Table:
    """Fill names down and turn group rows into a column: done once the whole table is
    known (after joining the pages it is printed on)."""
    _fill_down(table.rows)
    if table.groups:
        _group_column(table, table.groups)
        table.groups = {}
    return table


def _fill_down(rows: list[list[str]]) -> None:
    """Empty leading cells under a value stand for it (a branch named on its first row
    only). Filled only in columns that are often empty, and only when the row's first value
    is in a column that is nearly always filled: a subtotal row ("| | Subtotal Punjab |")
    writes its own label where a branch would go, so it is left as it is."""
    if len(rows) < 3:
        return
    width = len(rows[0])
    empty = [sum(1 for r in rows if not r[c]) for c in range(width)]
    sparse = [empty[c] > len(rows) / 3 and empty[c] <= len(rows) - 2 for c in range(width)]
    above: list[str] | None = None
    for row in rows:
        first = next((c for c, t in enumerate(row) if t), None)
        if (first and above is not None and all(sparse[c] for c in range(first))
                and not sparse[first] and all(above[c] for c in range(first))):
            row[:first] = above[:first]
        above = row


_TOTAL = re.compile(r"^\s*(grand\s+)?total\b", re.I)
_LABEL = re.compile(r"^([^:]{1,30}):\s*(\S.*)$")


def _group_column(table: Table, groups: dict[int, str]) -> None:
    """Group rows ("Region: Punjab") become a first column, so each row says its group."""
    labels = list(groups.values())
    parsed = [_LABEL.match(label) for label in labels]
    names = {m.group(1).strip() for m in parsed if m}
    if len(names) == 1 and all(parsed):
        name = names.pop()
        value = lambda label: _LABEL.match(label).group(2).strip()
    else:
        name, value = "Group", lambda label: label
    current = ""
    for i, row in enumerate(table.rows):
        if i in groups:
            current = value(groups[i])
        row.insert(0, "" if any(_TOTAL.match(t) for t in row) else current)
    table.header.insert(0, name)


_LEADER = re.compile(r"\s*\.{4,}\s*(\d+)\s*$")


def contents_list(table: Table) -> str | None:
    """A table of contents (titles with dot leaders and page numbers), which Docling reads as
    a table, written as plain lines instead: "1. Branch financials: page 3"."""
    lines = [" ".join(t for t in row if t) for row in [table.header] + table.rows]
    found = [m for m in (_LEADER.search(line) for line in lines) if m]
    if len(lines) < 3 or len(found) < 0.6 * len(lines):
        return None
    out = []
    for line in lines:
        m = _LEADER.search(line)
        out.append(f"{line[:m.start()].strip()}: page {m.group(1)}" if m else line)
    return "\n".join(out)


_TICKED, _UNTICKED = "☑☒✔✅", "☐"


def ticked(text: str) -> str:
    """'☑ Yes ☐ No' -> '☑ Yes ☐ No (ticked: Yes)': which box is ticked, in words."""
    if _UNTICKED not in text or not any(t in text for t in _TICKED):
        return text
    chosen = [m.strip() for m in re.findall(rf"[{_TICKED}]\s*([^{_TICKED}{_UNTICKED}]+)", text)]
    return f"{text} (ticked: {', '.join(c for c in chosen if c)})" if chosen else text


def turn(cells: list[Cell], num_rows: int, num_cols: int, quarter: int) -> tuple[list[Cell], int, int]:
    """A table printed sideways, read as Docling sees it on the page, turned upright. Text
    running up the page (90 degrees): the page's rows are the table's columns, last first;
    running down (270): the page's columns are the table's rows, last first."""
    out = []
    for c in cells:
        if quarter == 90:
            out.append(Cell(c.text, c.col, num_rows - (c.row + c.rows), c.cols, c.rows))
        else:
            out.append(Cell(c.text, num_cols - (c.col + c.cols), c.row, c.cols, c.rows))
    return out, num_cols, num_rows  # header flags were for the sideways layout: first row now


def _colour_cells(cells: list[Cell], boxes: dict[int, tuple], fills: list[tuple]) -> list[Cell]:
    """Empty cells filled with a strong colour (status shown only by colour: green, amber,
    red) get the colour's name. Docling keeps no cell for an empty one, so where it sits is
    taken from its row's and its column's other cells."""
    taken = {(r, c) for cell in cells for r in range(cell.row, cell.row + cell.rows)
             for c in range(cell.col, cell.col + cell.cols)}
    xs: dict[int, list[float]] = {}
    ys: dict[int, list[float]] = {}
    for i, cell in enumerate(cells):
        box = boxes.get(i)
        if box and cell.cols == 1:
            xs.setdefault(cell.col, []).append((box[0] + box[2]) / 2)
        if box and cell.rows == 1:
            ys.setdefault(cell.row, []).append((box[1] + box[3]) / 2)
    out = []
    for r in ys:
        for c in xs:
            if (r, c) in taken:
                continue
            x, y = median(xs[c]), median(ys[r])
            name = next((n for (l, t, rr, b), n in fills if l <= x <= rr and t <= y <= b), None)
            if name:
                out.append(Cell(name, r, c))
    return cells + out


def docling_table(item, doc, clean=lambda text: text, quarter: int | None = None,
                  fills: list[tuple] | None = None) -> Table | None:
    """The Table for a Docling TableItem; `clean` is applied to every cell's text.
    `quarter`: the table is printed sideways (90 or 270). `fills`: strongly coloured shapes
    on its page, ((left, top, right, bottom), colour name), for cells with only a colour."""
    data = item.data
    page = item.prov[0].page_no if item.prov else None
    height = doc.pages[page].size.height if page in doc.pages else None
    cells, boxes = [], {}
    for c in data.table_cells:
        # a cell holding other content (a list, a nested table) serialises it
        text = c._get_text(doc=doc) if type(c).__name__ == "RichTableCell" else c.text
        cells.append(Cell(ticked(clean(" ".join((text or "").split()))),
                          c.start_row_offset_idx, c.start_col_offset_idx,
                          max(1, c.end_row_offset_idx - c.start_row_offset_idx),
                          max(1, c.end_col_offset_idx - c.start_col_offset_idx),
                          c.column_header))
        if c.bbox is not None and height:
            b = c.bbox.to_top_left_origin(height)
            boxes[len(cells) - 1] = (b.l, b.t, b.r, b.b)
    if fills:
        cells = _colour_cells(cells, boxes, fills)
    rows, cols = data.num_rows, data.num_cols
    if quarter in (90, 270):
        cells, rows, cols = turn(cells, rows, cols, quarter)
    return build(cells, rows, cols)


def form_lines(table: Table) -> str | None:
    """A form laid out as a grid, label then value ("Vendor name | Atlas Tyres | NTN |
    4410287-6"), as "Key: value" lines: read as a table, its first line would be taken
    for column names. A form: an even number of columns, every other cell a distinct short
    label without digits, and values (not column names) on the first line."""
    rows = [table.header] + table.rows
    width = len(table.header)
    if width % 2 or len(rows) < 3:
        return None
    labels = [row[c] for row in rows for c in range(0, width, 2) if row[c] or row[c + 1]]
    if (len(set(labels)) != len(labels)
            or not all(label and not re.search(r"\d", label) and len(label.split()) <= 4
                       for label in labels)):
        return None
    first_values = [table.header[c] for c in range(1, width, 2)]
    if not any(re.search(r"\d", v) or len(v.split()) > 4 for v in first_values):
        return None  # first line holds column names: a normal table
    return "\n".join(f"{row[c]}: {row[c + 1]}" for row in rows for c in range(0, width, 2)
                     if row[c] or row[c + 1])


# ---- tables printed over several pages ----

_ID = re.compile(r"\bTable\s+(\d+[A-Za-z]?)", re.I)


def _kind(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if re.fullmatch(r"[\d\s.,%+()\-−]+", t):
        return "number"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4}", t):
        return "date"
    if len(t.split()) == 1 and re.search(r"\d", t) and re.search(r"[A-Za-z]", t):
        return "code"
    return "text"


def _like_rows(cells: list[str], rows: list[list[str]]) -> float | None:
    """In the columns that hold numbers, dates or codes, the share where this line of cells
    does too: 1.0 for a row of data, 0 for a header (words). Columns of words can't tell
    them apart; None if there are only such columns."""
    typed = []
    for c in range(len(cells)):
        kinds = [k for k in (_kind(row[c]) for row in rows) if k != "empty"]
        if kinds:
            top = max(set(kinds), key=kinds.count)
            if top != "text" and kinds.count(top) >= 0.6 * len(kinds):
                typed.append((c, top))
    if not typed:
        return None
    return sum(_kind(cells[c]) == kind for c, kind in typed) / len(typed)


def continuation(first: Table, second: Table) -> str | None:
    """How `second` (the next table, nothing else in between) carries on `first`: "rows" (the
    same header again), "headerless" (its header line is really a row of `first`'s kind) or
    "columns" (the same first column with the same values, then new columns). None if it
    is another table, e.g. with its own caption."""
    if second.caption and _ID.search(second.caption) and (
            not first.caption or not _ID.search(first.caption)
            or _ID.search(second.caption).group(1) != _ID.search(first.caption).group(1)):
        return None
    norm = lambda cells: [" ".join(c.lower().split()) for c in cells]
    if norm(first.header) == norm(second.header):
        return "rows"
    if len(first.header) == len(second.header) and len(first.rows) >= 3:
        data, header = _like_rows(second.header, first.rows), _like_rows(first.header, first.rows)
        if data is not None and data >= 0.8 and header < 0.5:
            return "headerless"
    if (norm(first.header[:1]) == norm(second.header[:1]) and len(first.rows) == len(second.rows)
            and [r[0] for r in first.rows] == [r[0] for r in second.rows]
            and not set(norm(first.header[1:])) & set(norm(second.header[1:]))):
        return "columns"
    return None


def join(first: Table, second: Table, how: str) -> Table:
    notes = " ".join(n for n in (first.notes, second.notes) if n)
    if how == "columns":
        # e.g. "Table 9 (continued): January-June. Row numbers match the first part."
        if second.caption and second.caption != first.caption:
            notes = " ".join(n for n in (second.caption, notes) if n)
        return Table(first.header + second.header[1:],
                     [a + b[1:] for a, b in zip(first.rows, second.rows)],
                     first.caption, first.intro, notes, dict(first.groups), list(first.pages))
    moved = [second.header] if how == "headerless" else []
    shift = len(first.rows) + len(moved)
    groups = {**first.groups, **{i + shift: g for i, g in second.groups.items()}}
    page = second.pages[0] if second.pages else (first.pages[-1] if first.pages else None)
    pages = first.pages + [page] * len(moved) + second.pages
    return Table(first.header, first.rows + moved + second.rows, first.caption or second.caption,
                 first.intro or second.intro, notes, groups, pages)


def parts(table: Table) -> list[tuple[int | None, Table]]:
    """The finished table per page it is printed on, each with its caption, header and notes."""
    out: list[tuple[int | None, Table]] = []
    pages = table.pages or [None] * len(table.rows)
    for page, row in zip(pages, table.rows):
        if not out or out[-1][0] != page:
            out.append((page, Table(table.header, [], table.caption, table.intro, table.notes)))
        out[-1][1].rows.append(row)
    return out or [(pages[0] if pages else None, table)]


# ---- caption, introduction and notes ----

Box = tuple[float, float, float, float]  # left, top, right, bottom; top of the page is 0

CAPTION = re.compile(r"\bTable\s+\d+[A-Za-z]?\b(?:\s*\([^)]*\))?\s*[:.]\s*\S[^\n]*")
_NOTES_START = re.compile(r"\s(?=(?:footnotes?|legend|notes?|sources?)\s*:)", re.I)
NOTE = re.compile(r"^\s*((notes?|footnotes?|legend|sources?|key)\s*:|[*†‡§¹²³]|[a-z]\s+[A-Z]|\(\w\)\s)",
                  re.I)
ABOVE, BELOW, NOTE_GAP = 60.0, 40.0, 60.0   # points
MAX_INTRO, MAX_NOTES = 400, 800              # characters


@dataclass
class Block:
    """One item of the document in reading order, as far as tables' context needs it."""
    kind: str                # table, caption, heading, text, footnote, picture, other
    text: str = ""
    page: int | None = None
    box: Box | None = None


@dataclass
class Context:
    caption: str = ""
    intro: str = ""
    notes: str = ""
    caption_block: int | None = None


@dataclass
class Attached:
    tables: dict[int, Context] = field(default_factory=dict)  # by the table's block index
    used: set[int] = field(default_factory=set)               # blocks now part of a table
    rest: dict[int, str] = field(default_factory=dict)        # what is left of a block's text


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2]


def _caption_in(text: str) -> tuple[str, str] | None:
    """(caption, what is left of the text) if the text holds a "Table N: ..." caption."""
    m = CAPTION.search(text)
    if not m:
        return None
    caption, after = m.group(0), ""
    cut = _NOTES_START.search(caption)
    if cut:
        caption, after = caption[:cut.start()], caption[cut.start():]
    rest = f"{text[:m.start()]} {after} {text[m.end():]}"
    return caption.strip(), " ".join(rest.split())


def attach(blocks: list[Block], linked: dict[int, list[int]] | None = None) -> Attached:
    """Each table's caption, introduction and notes. `linked`: the caption blocks Docling
    itself linked to a table. Otherwise the caption is the nearest "Table N: ..." text above
    the table and over it (so side-by-side tables each get their own), or just below it if
    there is none above and it isn't right above another table."""
    linked = linked or {}
    out = Attached()
    tables = [i for i, b in enumerate(blocks) if b.kind == "table"]
    found = {i: _caption_in(b.text) for i, b in enumerate(blocks)
             if b.kind in ("caption", "text", "heading", "footnote") and b.text}
    found = {i: v for i, v in found.items() if v}

    def take(t: int, c: int, caption: str, rest: str) -> None:
        out.tables[t] = Context(caption=caption, caption_block=c)
        if rest:
            out.rest[c] = rest
        else:
            out.used.add(c)
        found.pop(c, None)

    for t in tables:
        if linked.get(t):
            caption = " ".join(blocks[c].text for c in linked[t] if blocks[c].text)
            out.tables[t] = Context(caption=caption, caption_block=linked[t][0])
            out.used.update(linked[t])
            for c in linked[t]:
                found.pop(c, None)
    for t in (t for t in tables if t not in out.tables):
        box, page = blocks[t].box, blocks[t].page
        if box is None:  # no positions (Word): the "Table N:" line just before it
            if t > 0 and (t - 1) in found and blocks[t - 1].text.lstrip().startswith("Table"):
                take(t, t - 1, *found[t - 1])
            continue
        near = [(box[1] - blocks[c].box[3], c) for c in found
                if blocks[c].page == page and blocks[c].box and _overlap(blocks[c].box, box)
                and -5 <= box[1] - blocks[c].box[3] <= ABOVE]
        if near:
            take(t, min(near)[1], *found[min(near)[1]])
    for t in (t for t in tables if t not in out.tables):
        box, page = blocks[t].box, blocks[t].page
        if box is None:
            continue
        below = [(blocks[c].box[1] - box[3], c) for c in found
                 if blocks[c].page == page and blocks[c].box and _overlap(blocks[c].box, box)
                 and -5 <= blocks[c].box[1] - box[3] <= BELOW
                 and not any(o != t and blocks[o].page == page and blocks[o].box
                             and _overlap(blocks[c].box, blocks[o].box)
                             and -5 <= blocks[o].box[1] - blocks[c].box[3] <= ABOVE
                             for o in tables)]
        if below:
            take(t, min(below)[1], *found[min(below)[1]])

    for t in tables:
        context = out.tables.setdefault(t, Context())
        context.intro = _intro(blocks, t, context, out)
        context.notes = _notes(blocks, t, out)
    return out


def _intro(blocks: list[Block], t: int, context: Context, out: Attached) -> str:
    """A short paragraph right before the table (or its caption) introduces it. It stays in
    the text as well."""
    start = context.caption_block if (context.caption_block is not None
                                      and context.caption_block < t) else t
    i = start - 1
    while i >= 0 and i in out.used:
        i -= 1
    if i < 0:
        return ""
    b = blocks[i]
    text = out.rest.get(i, b.text)
    if (b.kind != "text" or b.page != blocks[t].page or not text or len(text) > MAX_INTRO
            or CAPTION.match(text)):
        return ""
    if b.box and blocks[start].box and blocks[start].box[1] - b.box[3] > ABOVE:
        return ""
    return text


def _notes(blocks: list[Block], t: int, out: Attached) -> str:
    """Footnotes, a legend or a source right under the table belong to it."""
    notes, bottom, size = [], blocks[t].box[3] if blocks[t].box else None, 0
    for i in range(t + 1, len(blocks)):
        if i in out.used:
            continue
        b = blocks[i]
        text = out.rest.get(i, b.text)
        if b.page != blocks[t].page or b.kind not in ("text", "footnote", "caption") or not text:
            break
        if b.kind != "footnote" and not NOTE.match(text):
            break
        if bottom is not None and b.box and b.box[1] - bottom > NOTE_GAP:
            break
        if size + len(text) > MAX_NOTES:
            break
        notes.append(text)
        out.used.add(i)
        size += len(text)
        bottom = b.box[3] if b.box else bottom
    return " ".join(notes)
