"""MCP server: the documents and file tools, over the Model Context Protocol.

The agents (agents/) use these tools through an MCP client, the same way any MCP client
(Claude Desktop, an IDE) can use them through `ingest mcp` (stdio):

  search_documents      the best passages for a question, with citations (what the user may read)
  read_document         a stored file's text, chunk by chunk
  create_word_document  a .docx from Markdown-style text (headings, lists, tables, bold)
  create_spreadsheet    an .xlsx sheet from CSV text (added to the workbook if it exists)
  create_text_file      .md, .txt, .csv or .json
  read_file             what a created file contains, as text (to check it)
  list_files            the files created so far

Files are written only inside one output folder (per user), never elsewhere.
"""
import csv
import io
import json
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from ingest import doc_tables
from ingest.users import LOCAL, User

TEXT_TYPES = {".md", ".txt", ".csv", ".json"}
MAX_READ = 20_000


def safe_file(folder: Path, filename: str, suffixes: set[str]) -> Path:
    """`filename` inside `folder` (no folders, no tricks), with an allowed suffix."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(filename.strip()).name).strip(" .")
    if not name:
        raise ValueError("Give a file name, e.g. summary.docx")
    if Path(name).suffix.lower() not in suffixes:
        raise ValueError(f"The file name must end in one of: {', '.join(sorted(suffixes))}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _inline(paragraph, text: str) -> None:
    """Text with **bold** parts into a python-docx paragraph."""
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            paragraph.add_run(part).bold = bool(i % 2)


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # the |---|---| line
        rows.append(cells)
    return rows


def _cell_number(text: str):
    value = _number(text.strip().strip("*").strip())
    return value if isinstance(value, (int, float)) else None


def fill_formulas(rows: list[list[str]]) -> tuple[int, list[str]]:
    """Table cells written as formulas get their exact value, so no total is ever added up
    by the model: =SUM(LEFT) the numbers to the left in its row; =SUM(ABOVE) the numbers
    above in its column back to the previous =SUM(ABOVE) (a subtotal) or the header;
    =TOTAL(ABOVE) every number above that isn't a subtotal (a grand total); or arithmetic,
    e.g. =40406+29003+11734+14890. Results are written in the table's own number style."""
    numbers = [c for row in rows[1:] for c in row if _cell_number(c) is not None]
    commas = any("," in c for c in numbers)
    brackets = any(c.strip("* ").startswith("(") for c in numbers)

    def style(value) -> str:
        if isinstance(value, float) and not value.is_integer():
            text = f"{abs(value):,.2f}" if commas else f"{abs(value):.2f}"
        else:
            text = f"{abs(int(value)):,}" if commas else str(abs(int(value)))
        if value < 0:
            return f"({text})" if brackets else f"-{text}"
        return text

    subtotals: set[tuple[int, int]] = set()
    # Rows labelled total/subtotal are subtotals even when typed in as numbers (seen live:
    # a grand total counted them and came out double)
    total_rows = {r for r, row in enumerate(rows) if r and any(
        re.search(r"\b(sub-?)?totals?\b", c, re.I) and _cell_number(c) is None for c in row)}
    done, problems = 0, []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            raw = cell.strip()
            bold = raw.startswith("**") and raw.endswith("**") and len(raw) > 4
            text = raw.strip("*").strip()
            if r == 0 or not text.startswith("="):
                continue
            formula = text[1:].strip()
            key = formula.upper().replace(" ", "")
            try:
                if key == "SUM(LEFT)":
                    value = sum(v for x in row[:c] if (v := _cell_number(x)) is not None)
                elif key in ("SUM(ABOVE)", "TOTAL(ABOVE)"):
                    value = 0
                    for k in range(r - 1, 0, -1):
                        if c >= len(rows[k]):
                            continue
                        if (k, c) in subtotals or k in total_rows:  # a subtotal: where a
                            if key == "SUM(ABOVE)":  # subtotal starts, and never counted
                                break                # twice in a total
                            continue
                        value += _cell_number(rows[k][c]) or 0
                    subtotals.add((r, c))
                else:  # arithmetic; thousands commas (1,234) are dropped first
                    value = calculate_one(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", formula))
            except Exception as exc:
                problems.append(f"row {r + 1}, column {c + 1}: {text} ({exc})")
                continue
            row[c] = f"**{style(value)}**" if bold else style(value)
            done += 1
    return done, problems


def write_docx(path: Path, title: str, markdown: str) -> dict:
    import docx
    document = docx.Document()
    if title.strip():
        document.add_heading(title.strip(), level=0)
    lines, i = markdown.splitlines(), 0
    counts = {"headings": 0, "paragraphs": 0, "tables": 0, "formulas": 0, "problems": []}
    first = next((n for n, line in enumerate(lines) if line.strip()), None)
    same = lambda a, b: re.sub(r"\W+", "", a.lower()) == re.sub(r"\W+", "", b.lower())
    if first is not None and same(lines[first].lstrip("# "), title):
        i = first + 1  # the title written again as the first heading: once is enough
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = _table_rows(block)
            if rows:
                done, problems = fill_formulas(rows)
                counts["formulas"] += done
                counts["problems"] += problems
                width = max(len(r) for r in rows)
                table = document.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c in range(width):
                        cell = table.cell(r, c)
                        cell.text = row[c] if c < len(row) else ""
                        if r == 0:
                            for run in cell.paragraphs[0].runs:
                                run.bold = True
                counts["tables"] += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)", line)
        number = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        if heading:
            document.add_heading(heading.group(2).strip(), level=len(heading.group(1)))
            counts["headings"] += 1
        elif bullet:
            _inline(document.add_paragraph(style="List Bullet"), bullet.group(1))
            counts["paragraphs"] += 1
        elif number:
            _inline(document.add_paragraph(style="List Number"), number.group(1))
            counts["paragraphs"] += 1
        else:
            _inline(document.add_paragraph(), line.strip())
            counts["paragraphs"] += 1
        i += 1
    document.save(path)
    return counts


def read_any(path: Path) -> str:
    """A created file's content as plain text (Word: headings marked with #, tables as rows)."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx
        document = docx.Document(path)
        out = []
        for block in document.element.body.iterchildren():
            tag = block.tag.rsplit("}", 1)[-1]
            if tag == "p":
                paragraph = next(p for p in document.paragraphs if p._p is block)
                style = paragraph.style.name if paragraph.style is not None else ""
                level = re.search(r"Heading (\d)", style)
                prefix = "#" * int(level.group(1)) + " " if level else \
                    "# " if style == "Title" else "- " if "List" in style else ""
                if paragraph.text.strip():
                    out.append(prefix + paragraph.text)
            elif tag == "tbl":
                table = next(t for t in document.tables if t._tbl is block)
                for row in table.rows:
                    out.append("| " + " | ".join(c.text for c in row.cells) + " |")
        return "\n".join(out)
    if suffix == ".xlsx":
        out = []
        for title, rows in sheet_values(path).items():
            out.append(f"## Sheet: {title}")
            out += [" | ".join("" if v is None else str(v) for v in row) for row in rows]
        return "\n".join(out)
    return path.read_text(encoding="utf-8", errors="replace")


_EXCEL_REF = re.compile(r"(?<![A-Za-z_\d.])(?:'([^']+)'!|([A-Za-z_][\w.]*)!)?\$?([A-Z]{1,3})\$?"
                        r"(\d+)(?::\$?([A-Z]{1,3})\$?(\d+))?(?![\w(])")


def _numbers(values) -> list:
    out = []
    for v in values:
        if isinstance(v, list):
            out += _numbers(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(v)
    return out


_EXCEL_FUNCTIONS = {
    "SUM": lambda *a: sum(_numbers(a)), "MIN": lambda *a: min(_numbers(a)),
    "MAX": lambda *a: max(_numbers(a)), "COUNT": lambda *a: len(_numbers(a)),
    "AVERAGE": lambda *a: sum(_numbers(a)) / len(_numbers(a)),
    "ROUND": lambda x, n=0: round(x, int(n)), "ABS": abs}


def sheet_values(path: Path) -> dict[str, list[list]]:
    """Each sheet's cells with formulas worked out. openpyxl (create_spreadsheet) saves a
    formula without its result, which Excel works out on opening; read as saved, every total
    would look empty (seen live: a verifier failed a correct file for it). Formulas use the
    safe calculator; one it can't work out (IF, text...) is shown as written."""
    import openpyxl
    from openpyxl.utils import column_index_from_string, get_column_letter
    grids = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)]
             for ws in openpyxl.load_workbook(path).worksheets}
    cached = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)]
              for ws in openpyxl.load_workbook(path, data_only=True).worksheets}
    done: dict = {}

    def at(grid, row, col):
        return grid[row][col] if row < len(grid) and col < len(grid[row]) else None

    def cell(sheet: str, col: str, row: int, seen: frozenset):
        key = (sheet, col, row)
        if key not in done:
            r, c = row - 1, column_index_from_string(col) - 1
            raw = at(grids.get(sheet, []), r, c)
            if isinstance(raw, str) and raw.startswith("="):
                stored = at(cached.get(sheet, []), r, c)   # Excel saved its result
                if stored is None and key in seen:
                    raise ValueError("circular reference")
                raw = stored if stored is not None else formula(sheet, raw, seen | {key})
            done[key] = raw
        return done[key]

    def formula(sheet: str, text: str, seen: frozenset):
        def ref(m):
            where, (c1, r1, c2, r2) = m.group(1) or m.group(2) or sheet, m.group(3, 4, 5, 6)
            if c2 is None:
                v = cell(where, c1, int(r1), seen)
                v = 0 if v is None else v
                if not isinstance(v, (int, float)):
                    raise ValueError("not a number")
                return repr(v)
            i1, i2 = sorted((column_index_from_string(c1), column_index_from_string(c2)))
            j1, j2 = sorted((int(r1), int(r2)))
            if (i2 - i1 + 1) * (j2 - j1 + 1) > 100_000:
                raise ValueError("range too big")
            values = [cell(where, get_column_letter(i), j, seen)
                      for j in range(j1, j2 + 1) for i in range(i1, i2 + 1)]
            return repr(_numbers(values))
        body = text[1:] if "'" in text else text[1:].upper()
        return calculate_one(_EXCEL_REF.sub(ref, body).replace("^", "**"), _EXCEL_FUNCTIONS)

    out = {}
    for title, grid in grids.items():
        rows = []
        for r, row in enumerate(grid):
            shown = []
            for c, raw in enumerate(row):
                if isinstance(raw, str) and raw.startswith("="):
                    try:
                        raw = cell(title, get_column_letter(c + 1), r + 1, frozenset())
                    except Exception:  # shown as written; the reader can still check it
                        pass
                shown.append(raw)
            rows.append(shown)
        out[title] = rows
    return out


def _number(value: str):
    """A CSV cell as a number when it is one (1,234.5 and (12) accounting negatives too)."""
    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    plain = text.strip("()").replace(",", "")
    try:
        number = float(plain)
    except ValueError:
        return value
    number = -number if negative else number
    return int(number) if number.is_integer() and "." not in plain else number


_CALC_FUNCTIONS = {"sum": sum, "round": round, "abs": abs, "min": min, "max": max}


def calculate_one(expression: str, functions: dict | None = None) -> float | int:
    """Exact arithmetic on numbers, + - * / ** ( ) and sum/round/abs/min/max (or the
    `functions` given): parsed, never executed as code, so nothing else can run."""
    functions = _CALC_FUNCTIONS if functions is None else functions
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
           ast.UAdd: operator.pos}

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("power too large")
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in functions and not node.keywords:
            return functions[node.func.id](*[value(a) for a in node.args])
        raise ValueError(f"not allowed: {ast.unparse(node)}")
    if len(expression) > 5000:
        raise ValueError("expression too long")
    result = value(ast.parse(expression.strip(), mode="eval").body)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return round(result, 10) if isinstance(result, float) else result


def build_server(out_dir: Path, pipeline=None, user: User = LOCAL) -> MCPServer:
    """The MCP server for one user: their documents (if `pipeline`) and their output folder."""
    server = MCPServer(name="ingest", instructions="Search the user's documents and make files "
                       "(Word, Excel, Markdown, CSV) from what they contain.")
    out_dir = Path(out_dir)

    if pipeline is not None:
        from ingest.retrieve import Retriever
        retriever = Retriever(pipeline)

        @server.tool()
        def search_documents(query: str, limit: int = 5) -> str:
            """The passages of the user's documents that best answer `query`, each with its
            citation (file, page or time). Use specific queries; call again to search more."""
            found = retriever.search(query, max(1, min(limit, 10)), groups=user.reads)
            if not found:
                return "No passage matches."
            return "\n\n".join(f"[{i}] {p.citation}\n{p.text}" for i, p in enumerate(found, 1))

        def document_points(source: str) -> tuple[str, list] | None:
            """The stored chunks of `source` (a file name or path), if this user may read it."""
            files = pipeline.store.files(user.reads)
            match = [p for p in files if p == source or p.replace("\\", "/").endswith("/" + source)
                     or p.replace("\\", "/").rsplit("/", 1)[-1] == source]
            if not match:
                return None
            return match[0], list(pipeline.store.iter_points(
                with_vectors=False, source_path=match[0], groups=user.reads))

        @server.tool()
        def read_document(source: str, first_page: int = 0, last_page: int = 0,
                          first_chunk: int = 0, chunks: int = 20) -> str:
            """A stored document's text in reading order. `source`: a file name from
            search_documents. To read a whole table or section spread over several passages,
            give the pages search_documents showed: first_page and last_page (e.g. 3 and 6).
            Without pages: `chunks` chunks from `first_chunk` (0 = the start)."""
            found = document_points(source)
            if found is None:
                return f"No readable document called {source!r}."
            points = found[1]
            points.sort(key=lambda p: p.payload.get("chunk_index") or 0)
            if first_page or last_page:
                low, high = first_page or last_page, last_page or first_page
                part = [p for p in points if low <= (p.payload.get("page_number") or 0) <= high]
                if not part:
                    return f"No text on pages {low}-{high} ({len(points)} chunks in all)."
                where = f"pages {low}-{high}: {len(part)} of {len(points)} chunks"
            else:
                part = points[first_chunk:first_chunk + max(1, min(chunks, 50))]
                where = f"{len(points)} chunks in all; showing {first_chunk}-" \
                        f"{first_chunk + len(part) - 1}"
            text, used = [], 0
            for p in part:
                page = p.payload.get("page_number")
                piece = (f"[p. {page}] " if page else "") + p.payload["content"]
                if used + len(piece) > 40_000:
                    text.append("…(cut: read fewer pages at a time)")
                    break
                text.append(piece)
                used += len(piece)
            return f"{where}.\n\n" + "\n\n".join(text)

        @server.tool()
        def list_tables(source: str) -> str:
            """The tables in a document (`source`: a file name from search_documents): each
            one's name, caption, pages, number of rows and columns. Use the name and the
            column names with copy_table."""
            found = document_points(source)
            if found is None:
                return f"No readable document called {source!r}."
            tables = doc_tables.tables_in(found[1])
            return "\n".join(t.describe() for t in tables) or "No tables in this document."

        @server.tool()
        def copy_table(source: str, table: str, filename: str,
                       columns: list[str] | str | None = None,
                       rows_by: list[str] | str | None = None, across: str = "",
                       values: str = "", row_total: str = "",
                       sum_columns: list[str] | str | None = None,
                       subtotals_by: str | list[str] | None = "", grand_total: bool = False,
                       sheet: str = "", heading: str = "", note: str = "") -> str:
            """Copy a table from a document into a file exactly: every row and figure as the
            document has them, so nothing is retyped. Totals are worked out by this tool.
            - source, table: the document and the table's name from list_tables ("Table 1").
            - filename: .xlsx (written to `sheet`, replaced if it exists; totals as Excel
              formulas), .docx (added at the end: make the file first with
              create_word_document for its title and text; `heading` goes above the table),
              .md (added at the end) or .csv.
            - columns: keep only these columns (default: all).
            - Pivot a long table (a row per branch and quarter) into a wide one: rows_by
              ["Region", "Branch"], across "Qtr", values "Revenue" gives a row per branch and
              a column per quarter.
            - row_total: add a column with this name holding each row's total (of the pivoted
              columns, or of sum_columns).
            - subtotals_by: a column (e.g. "Region"): a subtotal row after each group.
              grand_total: a total row at the end.
            The document's own printed total rows are left out when totals are added (they'd
            be counted twice) and are checked against its rows; the reply says if they match."""
            found = document_points(source)
            if found is None:
                return f"No readable document called {source!r}."
            path, points = found

            def names(value) -> list[str]:  # seen live: "Region,Code,Branch" for a list
                if isinstance(value, str):
                    return [n.strip() for n in value.split(",") if n.strip()]
                return [str(n) for n in value or []]
            group = names(subtotals_by)  # and ["Region"] for one name
            try:
                wanted = doc_tables.find(doc_tables.tables_in(points), table)
                shaped = doc_tables.shape(wanted, names(columns), names(rows_by), across,
                                          values, row_total, names(sum_columns),
                                          group[0] if group else "", grand_total)
                target = safe_file(out_dir, filename, {".xlsx", ".docx", ".md", ".csv"})
                out_dir.mkdir(parents=True, exist_ok=True)
                pages = sorted(set(wanted.pages))
                name = path.replace("\\", "/").rsplit("/", 1)[-1]
                where = (f"p. {pages[0]}" if len(pages) == 1 else
                         f"pages {pages[0]}-{pages[-1]}") if pages else ""
                source_note = (f"Source: {name}, {wanted.name}" + (f" ({where})" if where else "")
                               + (". Totals worked out from its rows." if shaped.kinds.count("data")
                                  != len(shaped.kinds) or shaped.row_total is not None else "."))
                placed = doc_tables.write(target, shaped, source_note, sheet or wanted.name,
                                          heading, note)
            except ValueError as exc:
                return f"Error: {exc}"
            data = shaped.kinds.count("data")
            extra = len(shaped.kinds) - data
            checks = shaped.checks
            differs = [c for c in checks if c.startswith("DIFFERS")]
            summary = (f"Copied {wanted.name} ({name}{', ' + where if where else ''}) to "
                       f"{target.name}, {placed}: {data} rows"
                       + (f" + {extra} total rows" if extra else "")
                       + f"; columns: {' | '.join(shaped.header)}.")
            if shaped.kinds and shaped.kinds[-1] == "total":
                last = shaped.rows[-1]
                summary += " Grand total: " + ", ".join(
                    f"{shaped.header[c]} {doc_tables.show(last[c], shaped.style)}"
                    for c in range(len(last)) if isinstance(last[c], (int, float)))
            if checks:
                ok = len(checks) - len(differs)
                summary += (f"\nThe document's printed totals: {ok} of {len(checks)} match its "
                            "rows." + ("".join(f"\n- {d}" for d in differs)))
            return summary + "".join(f"\nNote: {n}" for n in shaped.notes)

    @server.tool()
    def calculate(expressions: list[str]) -> str:
        """Exact arithmetic, so no figure is ever worked out in your head: totals, differences,
        percentages, averages. Each expression uses numbers (no thousands commas), + - * / ( )
        and sum([...]), round(x, n), abs, min, max; e.g. "8180+34540+9348+39775" or
        "(3663-3125)/3125*100". Give many at once; one result per line."""
        out = []
        for expression in expressions[:200]:
            try:
                out.append(f"{expression} = {calculate_one(expression)}")
            except Exception as exc:
                out.append(f"{expression} -> error: {exc}")
        return "\n".join(out) or "No expressions given."

    @server.tool()
    def create_word_document(filename: str, title: str, markdown: str) -> str:
        """Make a Word file (.docx). `markdown`: # headings, paragraphs, - bullets, 1. numbered
        items, **bold**, and tables as | a | b | rows (first row = header). Write every total
        in a table as a formula and the exact value is filled in: =SUM(LEFT) (the numbers to
        its left in the row), =SUM(ABOVE) (the numbers above in its column, back to the
        previous =SUM(ABOVE): a subtotal), =TOTAL(ABOVE) (every number above that isn't a
        subtotal: a grand total), or arithmetic such as =40406+29003+11734+14890."""
        path = safe_file(out_dir, filename, {".docx"})
        counts = write_docx(path, title, markdown)
        note = f", {counts['formulas']} totals filled in" if counts["formulas"] else ""
        if counts["problems"]:
            note += ". Formulas that failed (fix them): " + "; ".join(counts["problems"][:10])
        return f"Saved {path.name}: {counts['headings']} headings, {counts['paragraphs']} " \
               f"paragraphs, {counts['tables']} tables{note}."

    @server.tool()
    def create_spreadsheet(filename: str, sheet: str, csv_text: str) -> str:
        """Make (or add a sheet to) an Excel file (.xlsx) from CSV text; the first row is the
        header. Numbers become numbers (1,234.5 and (12) for negatives too). Write totals as
        Excel formulas (e.g. =SUM(D2:G2)): Excel works them out exactly."""
        import openpyxl
        from openpyxl.styles import Font
        path = safe_file(out_dir, filename, {".xlsx"})
        book = openpyxl.load_workbook(path) if path.exists() else openpyxl.Workbook()
        if not path.exists():
            book.remove(book.active)
        title = re.sub(r"[\[\]:*?/\\]", "_", sheet.strip() or "Sheet")[:31]
        if title in book.sheetnames:
            del book[title]
        ws = book.create_sheet(title)
        rows = list(csv.reader(io.StringIO(csv_text.strip())))
        for r, row in enumerate(rows):
            ws.append([cell if r == 0 else _number(cell) for cell in row])
        for cell in ws[1] if rows else []:
            cell.font = Font(bold=True)
        book.save(path)
        return f"Saved {path.name}, sheet {title!r}: {len(rows)} rows."

    @server.tool()
    def create_text_file(filename: str, content: str) -> str:
        """Make a text file: .md, .txt, .csv or .json (JSON is checked for validity)."""
        path = safe_file(out_dir, filename, TEXT_TYPES)
        if path.suffix.lower() == ".json":
            json.loads(content)
        path.write_text(content, encoding="utf-8")
        return f"Saved {path.name}: {len(content)} characters."

    @server.tool()
    def read_file(filename: str) -> str:
        """What a created file contains, as text (Word headings marked #, tables as rows;
        Excel sheet by sheet), to check it."""
        path = safe_file(out_dir, filename, TEXT_TYPES | {".docx", ".xlsx"})
        if not path.exists():
            return f"There is no file {path.name}."
        text = read_any(path)
        return text if len(text) <= MAX_READ else text[:MAX_READ] + "\n…(cut)"

    @server.tool()
    def list_files() -> str:
        """The files created so far, with their sizes."""
        if not out_dir.exists():
            return "No files yet."
        files = sorted(p for p in out_dir.iterdir() if p.is_file())
        return "\n".join(f"{p.name} ({p.stat().st_size} bytes)" for p in files) or "No files yet."

    return server


def serve(pipeline) -> None:
    """`ingest mcp`: the server over stdio, as the local administrator, files in outputs/."""
    build_server(Path(pipeline.settings.outputs_dir).resolve(), pipeline, LOCAL).run("stdio")
