"""copy_table: a stored table joined from its pieces, pivoted and totalled exactly, written
to Excel / Word, and checked against the totals the document prints."""
import asyncio
from types import SimpleNamespace

import pytest

from ingest import doc_tables
from ingest.mcp_server import build_server, read_any, sheet_values, write_docx
from ingest.users import User

CAPTION = "Table 1: Quarterly revenue by branch, FY2026 (PKR '000)"
INTRO = "Table 1 lists revenue for every branch, with a subtotal after each region."
HEADER = "| Region | Branch | Qtr | Revenue | Margin % |\n| --- | --- | --- | --- | --- |"


def piece(index, page, rows):
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return SimpleNamespace(payload={"modality": "table", "chunk_index": index,
                                    "page_number": page,
                                    "content": f"{CAPTION}\n{INTRO}\n\n{HEADER}\n{body}"})


ROWS = [  # as stored: three pieces over two pages, subtotals and a grand total printed
    piece(7, 3, [["Punjab", "Lahore", "Q1", "9,806", "16.6%"],
                 ["Punjab", "Lahore", "Q2", "35,542", "2.8%"],
                 ["Punjab", "Multan", "Q1", "2,166", "4.2%"]]),
    piece(8, 3, [["Punjab", "Multan", "Q2", "(455)", "-4.4%"],
                 ["Punjab", "", "Subtotal Punjab", "47,059", ""]]),
    piece(9, 4, [["Sindh", "Sukkur", "Q1", "36,289", "-7.2%"],
                 ["Sindh", "Sukkur", "Q2", "1,000", "1.0%"],
                 ["Sindh", "", "Subtotal Sindh", "37,289", ""],
                 ["", "", "GRAND TOTAL", "84,348", ""]]),
    SimpleNamespace(payload={"modality": "text", "chunk_index": 10, "page_number": 4,
                             "content": "Revenue grew."}),
    SimpleNamespace(payload={"modality": "table", "chunk_index": 11, "page_number": 5,
                             "content": "Table 16a: Depot bays\n\n| Bay | Slots |\n|---|---|\n"
                                        "| B1 | 271 |"}),
]


def test_a_table_is_joined_from_its_pieces_and_found_by_name():
    tables = doc_tables.tables_in(ROWS)
    assert [t.name for t in tables] == ["Table 1", "Table 16a"]
    one = tables[0]
    assert len(one.rows) == 9 and one.pages == [3, 3, 4]
    assert "pages 3-4 | 6 rows + 3 printed total rows" in one.describe()
    assert doc_tables.find(tables, "Table 1") is one          # not Table 16a
    assert doc_tables.find(tables, "1") is one
    assert doc_tables.find(tables, "depot bays").name == "Table 16a"
    with pytest.raises(ValueError, match="no table matches"):
        doc_tables.find(tables, "Table 9")
    assert one.describe().startswith(CAPTION + " |")          # its name not said twice
    # a note mentioning a total doesn't make a row a total row; labels do
    assert not doc_tables._is_total(["Group revenue", "3,063,502", "Table 1, grand total"])
    assert all(doc_tables._is_total([label, "5"]) for label in
               ("Total", "GRAND TOTAL", "Subtotal Sindh", "**Region total**", "Total revenue"))


def test_pivot_with_row_totals_subtotals_and_a_grand_total_is_exact():
    table = doc_tables.find(doc_tables.tables_in(ROWS), "Table 1")
    shaped = doc_tables.shape(table, rows_by=["Region", "Branch"], across="Qtr",
                              values="Revenue", row_total="Full year", subtotals_by="Region",
                              grand_total=True)
    assert shaped.header == ["Region", "Branch", "Q1", "Q2", "Full year"]
    assert shaped.rows[0] == ["Punjab", "Lahore", 9806, 35542, 45348]
    assert shaped.rows[1] == ["Punjab", "Multan", 2166, -455, 1711]     # (455) is negative
    assert shaped.rows[2] == ["Subtotal Punjab", "", 11972, 35087, 47059]
    assert shaped.rows[-1] == ["Grand total", "", 48261, 36087, 84348]  # subtotals not doubled
    assert shaped.kinds == ["data", "data", "subtotal", "data", "subtotal", "total"]
    # the printed totals agree with the rows
    assert all(c.startswith("ok") for c in shaped.checks) and len(shaped.checks) == 3
    assert doc_tables.show(-455, shaped.style) == "(455)"


def test_a_printed_total_that_disagrees_with_its_rows_is_reported():
    wrong = [piece(7, 3, [["Punjab", "Lahore", "Q1", "100", ""],
                          ["Punjab", "Lahore", "Q2", "200", ""],
                          ["", "", "GRAND TOTAL", "301", ""]])]
    table = doc_tables.tables_in(wrong)[0]
    assert doc_tables.printed_totals(table) == [
        "DIFFERS: GRAND TOTAL Revenue: printed 301, its rows add up to 300"]


def test_unknown_columns_say_which_there_are():
    table = doc_tables.tables_in(ROWS)[0]
    with pytest.raises(ValueError, match="Region | Branch | Qtr"):
        doc_tables.shape(table, rows_by=["Region"], across="Quarter", values="Revenue")


def test_excel_gets_formulas_whose_values_match(tmp_path):
    table = doc_tables.tables_in(ROWS)[0]
    shaped = doc_tables.shape(table, rows_by=["Region", "Branch"], across="Qtr",
                              values="Revenue", row_total="Full year", subtotals_by="Region",
                              grand_total=True)
    path = tmp_path / "revenue.xlsx"
    doc_tables.write(path, shaped, "Source: report.pdf, Table 1.", sheet="Table 1")
    doc_tables.write(path, shaped, "Source: report.pdf, Table 1.", sheet="Table 1")  # replaced
    import openpyxl
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Table 1"]
    assert book["Table 1"]["E2"].value == "=SUM(C2:D2)"
    rows = sheet_values(path)["Table 1"]
    assert rows[1:7] == [[None if v == "" else v for v in r] for r in shaped.rows]  # same values
    assert rows[-1][0] == "Source: report.pdf, Table 1."


def test_word_gets_the_table_after_its_text(tmp_path):
    path = tmp_path / "summary.docx"
    write_docx(path, "Revenue", "Revenue grew in both regions.")
    table = doc_tables.tables_in(ROWS)[0]
    shaped = doc_tables.shape(table, rows_by=["Region", "Branch"], across="Qtr",
                              values="Revenue", row_total="Full year", grand_total=True)
    doc_tables.write(path, shaped, "Source: report.pdf, Table 1.", heading="By branch")
    text = read_any(path)
    assert text.index("Revenue grew") < text.index("## By branch") < text.index("| Region |")
    assert "| Punjab | Multan | 2,166 | (455) | 1,711 |" in text
    assert "| Grand total |  | 48,261 | 36,087 | 84,348 |" in text


def test_copy_table_through_mcp_reaches_only_the_users_documents(make_pipeline, tmp_path):
    from mcp import Client
    report = tmp_path / "report.md"
    rows = [line for p in ROWS[:3] for line in p.payload["content"].splitlines()[5:]]
    report.write_text(f"{CAPTION}\n\n{HEADER}\n" + "\n".join(rows), encoding="utf-8")
    pipeline = make_pipeline(chunk_size=5000)
    pipeline.ingest_file(report, groups=["user:alice"], owner="alice")
    store = pipeline.store
    for point in store.iter_points(with_vectors=False):  # pretend the PDF reader marked it
        store.client.set_payload(store.collection, {"modality": "table", "page_number": 3},
                                 points=[point.id])

    async def call(user, tool, **args):
        async with Client(build_server(tmp_path / "out", pipeline, user)) as client:
            return (await client.call_tool(tool, {"source": "report.md", **args})
                    ).content[0].text
    listed = asyncio.run(call(User("alice"), "list_tables"))
    assert listed.startswith("Table 1:") and "6 rows + 3 printed total rows" in listed
    done = asyncio.run(call(User("alice"), "copy_table", table="Table 1",
                            filename="revenue.xlsx", rows_by=["Region", "Branch"],
                            across="Qtr", values="Revenue", row_total="Full year",
                            subtotals_by="Region", grand_total=True))
    assert "Copied Table 1 (report.md, p. 3) to revenue.xlsx" in done
    assert "3 rows + 3 total rows" in done and "Full year 84,348" in done
    assert "printed totals: 3 of 3 match" in done
    assert sheet_values(tmp_path / "out" / "revenue.xlsx")["Table 1"][1][4] == 45348
    # seen live: a list given as "a,b" and one name given as a list
    loose = asyncio.run(call(User("alice"), "copy_table", table="1", filename="loose.csv",
                             rows_by="Region, Branch", across="Qtr", values="Revenue",
                             subtotals_by=["Region"], grand_total=True))
    assert "3 rows + 3 total rows" in loose
    assert "No readable document" in asyncio.run(call(User("bob"), "list_tables"))
