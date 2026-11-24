"""The agent harness and the MCP server, with a scripted model (no Gemini, no Modal)."""
import asyncio

import pytest

from ingest.agents import run_task
from ingest.agents.harness import Agent
from ingest.llm import FakeLLM, LLMUnavailable, Step, ToolCall
from ingest.mcp_server import build_server, read_any, safe_file
from ingest.users import User


def plan(*steps) -> dict:
    return {"steps": [{"name": n, "kind": k, "goal": g} for n, k, g in steps],
            "requirements": ["Word file: a table of leave days"],
            "deliverable": "A Word file about leave."}


def calls(*pairs) -> Step:
    return Step("", [ToolCall(name, args) for name, args in pairs])


TABLE = "# Annual leave\n\nStaff get **20 days** paid (policy.md).\n\n| Item | Days |\n" \
        "|---|---|\n| Annual leave | 20 |\n\n- approved by finance"


ENGINES = pytest.mark.parametrize("engine", ["langgraph", "custom"])


@ENGINES
def test_agents_research_write_and_verify(make_pipeline, docs, tmp_path, engine):
    pipeline = make_pipeline(agent_engine=engine)
    pipeline.run([docs])
    llm = FakeLLM(
        plan(("find leave", "research", "Find the annual leave days."),
             ("write file", "write", "Write leave.docx with a table.")),
        calls(("search_documents", {"query": "annual leave days"})),
        calls(("finish", {"summary": "Found it", "findings": ["20 days paid leave (policy.md)"],
                          "files": [], "handover": "20 days"})),
        calls(("create_word_document", {"filename": "leave.docx", "title": "Leave",
                                        "markdown": TABLE})),
        calls(("finish", {"summary": "Wrote leave.docx", "findings": [], "files": ["leave.docx"],
                          "handover": ""})),
        calls(("read_file", {"filename": "leave.docx"})),
        calls(("verdict", {"passed": True, "problems": []})))
    seen = []
    result = run_task("A Word file on annual leave", pipeline, out_dir=tmp_path / "out",
                      echo=seen.append, llm=llm)
    assert result.ok and [p.name for p in result.files] == ["leave.docx"]
    assert "passed" in result.summary
    text = read_any(tmp_path / "out" / "leave.docx")
    assert "| Annual leave | 20 |" in text and "# Annual leave" in text
    # the research agent's report reached the writer; the search really ran through MCP
    assert "20 days paid leave (policy.md)" in llm.prompts[3][0]["text"]
    assert any("search_documents" in line and "20 days" in line for line in seen)
    # the planner's checklist reached the writer and the verifier (not the researcher)
    checklist = "1. Word file: a table of leave days"
    assert checklist not in llm.prompts[1][0]["text"]
    assert checklist in llm.prompts[3][0]["text"] and checklist in llm.prompts[5][0]["text"]


@ENGINES
def test_a_failed_check_gets_fixed_and_checked_again(make_pipeline, docs, tmp_path, engine):
    pipeline = make_pipeline(agent_verify_rounds=1, agent_engine=engine)
    pipeline.run([docs])
    llm = FakeLLM(
        plan(("write", "write", "Write leave.md")),
        calls(("create_text_file", {"filename": "leave.md", "content": "Leave: 25 days"})),
        calls(("finish", {"summary": "done", "findings": [], "files": ["leave.md"],
                          "handover": ""})),
        calls(("verdict", {"passed": False, "problems": ["says 25 days; policy.md says 20"]})),
        calls(("create_text_file", {"filename": "leave.md", "content": "Leave: 20 days"})),
        calls(("finish", {"summary": "fixed", "findings": [], "files": ["leave.md"],
                          "handover": ""})),
        calls(("verdict", {"passed": True, "problems": []})))
    result = run_task("leave.md with the leave days", pipeline, out_dir=tmp_path / "out",
                      echo=lambda line: None, llm=llm)
    assert result.ok and (tmp_path / "out" / "leave.md").read_text() == "Leave: 20 days"
    assert [r.passed for r in result.reports if r.passed is not None] == [False, True]


def test_a_stopped_task_carries_on_from_its_last_finished_step(make_pipeline, docs, tmp_path):
    """LangGraph saves the state after every node: a task stopped by the model (quota, the
    internet) is resumed without planning or researching again."""
    pipeline = make_pipeline()
    pipeline.run([docs])
    first = FakeLLM(
        plan(("find leave", "research", "Find the annual leave days."),
             ("write file", "write", "Write leave.md.")),
        calls(("search_documents", {"query": "annual leave days"})),
        calls(("finish", {"summary": "Found it", "findings": ["20 days paid leave (policy.md)"],
                          "files": [], "handover": "20 days"})),
        LLMUnavailable("quota used up"))
    stopped = run_task("leave.md with the leave days", pipeline, out_dir=tmp_path / "out",
                       echo=lambda line: None, llm=first)
    assert not stopped.ok and "quota used up" in stopped.summary
    assert [r.agent for r in stopped.reports] == ["agent 1"]
    run_id = stopped.summary.rsplit("--resume ", 1)[-1].strip()
    second = FakeLLM(
        calls(("create_text_file", {"filename": "leave.md", "content": "Leave: 20 days"})),
        calls(("finish", {"summary": "done", "findings": [], "files": ["leave.md"],
                          "handover": ""})),
        calls(("verdict", {"passed": True, "problems": []})))
    result = run_task("", pipeline, out_dir=tmp_path / "out", echo=lambda line: None,
                      llm=second, resume=run_id)
    assert result.ok and [p.name for p in result.files] == ["leave.md"]
    # no new plan, no new research: the writer got agent 1's saved report straight away
    assert len(second.prompts) == 3 and "20 days paid leave" in second.prompts[0][0]["text"]
    assert [r.agent for r in result.reports] == ["agent 1", "agent 2", "verifier (leave.md)"]
    # another user can't resume it
    from ingest.users import User
    other = run_task("", pipeline, User("mallory"), out_dir=tmp_path / "out",
                     echo=lambda line: None, llm=FakeLLM(), resume=run_id)
    assert not other.ok and "No stopped task" in other.summary


def test_the_agent_graph_has_its_nodes_and_loops(settings):
    from ingest.agents.graph import diagram
    text = diagram(settings)
    for node in ("planner", "worker", "find_files", "verifier", "fixer", "next_file"):
        assert node in text
    assert "next step" in text and "failed" in text


def test_agent_memory_is_deleted_after_reporting():
    llm = FakeLLM(Step("thinking aloud"), calls(("finish", {"summary": "ok", "findings": [],
                                                            "files": [], "handover": ""})))
    agent = Agent("a", "system", "goal", [], lambda n, a: "", llm, 5, lambda line: None)
    report = agent.run()
    assert report.summary == "ok" and agent.memory == []


def test_users_only_reach_their_own_documents_through_mcp(make_pipeline, docs, tmp_path):
    from mcp import Client
    pipeline = make_pipeline()
    pipeline.ingest_file(docs / "policy.md", groups=["user:alice"], owner="alice")
    pipeline.ingest_file(docs / "notes.txt", groups=["user:bob"], owner="bob")

    async def ask(user):
        async with Client(build_server(tmp_path / user.name, pipeline, user)) as client:
            result = await client.call_tool("search_documents", {"query": "Lahore growth"})
            return result.content[0].text
    alice = asyncio.run(ask(User("alice")))
    assert "policy.md" in alice and "notes.txt" not in alice


def test_an_agent_out_of_turns_still_hands_over_what_it_found():
    """Seen live: agents used up their turns and handed nothing to the next one."""
    search = calls(("search_documents", {"query": "x"}))
    llm = FakeLLM(search, search, calls(("finish", {
        "summary": "Partly done", "findings": ["Q1 revenue 3,663 (p. 6)"], "files": [],
        "handover": "Q2-Q4 still missing"})))
    tools = [{"name": "search_documents", "description": "", "parameters": {}}]
    report = Agent("a", "system", "goal", tools, lambda n, a: "found", llm, 2,
                   lambda line: None).run()
    assert not report.ok and report.findings == ["Q1 revenue 3,663 (p. 6)"]
    last = llm.prompts[-1]
    assert "out of turns" in str(last[-1])


def test_read_document_reads_a_page_range_whole(make_pipeline, tmp_path):
    """A table spread over pages 3-6 comes back whole and in order (search shows only a few
    of its pieces)."""
    from mcp import Client
    report = tmp_path / "report.md"
    report.write_text("\n\n".join(f"Paragraph {i}: " + "figures and words " * 15
                                  for i in range(1, 9)), encoding="utf-8")
    pipeline = make_pipeline(chunk_size=300, chunk_overlap=0)
    pipeline.ingest_file(report)
    store = pipeline.store
    for point in store.iter_points(with_vectors=False):  # pretend each chunk is a page
        store.client.set_payload(store.collection, {"page_number": point.payload["chunk_index"] + 1},
                                 points=[point.id])

    async def read(**args):
        async with Client(build_server(tmp_path / "out", pipeline, User("me", admin=True))) as c:
            return (await c.call_tool("read_document", {"source": "report.md", **args})
                    ).content[0].text
    text = asyncio.run(read(first_page=3, last_page=4))
    assert text.startswith("pages 3-4: 2 of") and "[p. 3]" in text and "[p. 4]" in text
    assert "[p. 2]" not in text and "[p. 5]" not in text
    assert text.index("[p. 3]") < text.index("[p. 4]")
    assert "No text on pages 90-91" in asyncio.run(read(first_page=90, last_page=91))


def test_calculate_is_exact_and_runs_nothing_but_arithmetic(tmp_path):
    """Seen live: an agent added up costs in its head and got 4 of 30 totals wrong."""
    from mcp import Client
    from ingest.mcp_server import calculate_one
    assert calculate_one("33060+22316+21899+37673") == 114948
    assert calculate_one("(3663-3125)/3125*100") == pytest.approx(17.216)
    assert calculate_one("round(sum([1.5, 2.25, -0.75]) / 3, 2)") == 1.0
    for evil in ("__import__('os').system('echo hi')", "open('x')", "(1).__class__",
                 "9**9**9", "x + 1"):
        with pytest.raises(ValueError):
            calculate_one(evil)

    async def run():
        async with Client(build_server(tmp_path)) as client:
            return (await client.call_tool("calculate", {"expressions": [
                "8180+34540+9348+39775", "1/0", "max(3, 7)"]})).content[0].text
    lines = asyncio.run(run()).splitlines()
    assert lines[0] == "8180+34540+9348+39775 = 91843"
    assert "error" in lines[1] and lines[2] == "max(3, 7) = 7"


def test_word_table_totals_are_filled_in_exactly():
    """Seen live: a writer put 3 wrong branch totals into a Word table (Astore off by
    10,000). Totals written as formulas are worked out by the tool instead."""
    from ingest.mcp_server import fill_formulas
    rows = [["Branch", "Q1", "Q2", "Full year"],
            ["Gilgit", "40,406", "29,003", "=SUM(LEFT)"],
            ["Astore", "6,555", "(4,238)", "=SUM(LEFT)"],
            ["**Subtotal GB**", "=SUM(ABOVE)", "=SUM(ABOVE)", "**=SUM(ABOVE)**"],
            ["Muzaffarabad", "5,158", "43,998", "=SUM(LEFT)"],
            ["Subtotal AJK", "=SUM(ABOVE)", "=SUM(ABOVE)", "=SUM(ABOVE)"],
            ["Grand total", "=TOTAL(ABOVE)", "=TOTAL(ABOVE)", "=TOTAL(ABOVE)"],
            ["Check", "=46,961+5,158", "=round(2/3, 2)", "=bad("]]
    done, problems = fill_formulas(rows)
    assert rows[1][3] == "69,409" and rows[2][3] == "2,317"           # (4,238) is negative
    assert rows[3][1:] == ["46,961", "24,765", "**71,726**"]
    assert rows[5][1:] == ["5,158", "43,998", "49,156"]                 # only its own region
    assert rows[6][1:] == ["52,119", "68,763", "120,882"]               # subtotals not counted twice
    assert rows[7][1] == "52,119" and rows[7][2] == "0.67"
    assert done == 14 and len(problems) == 1 and "=bad(" in problems[0]

    # Subtotals typed in as numbers still aren't counted twice in the grand total
    typed = [["Branch", "Revenue"], ["Gilgit", "96,033"], ["Astore", "61,855"],
             ["**Subtotal GB**", "**157,888**"], ["Muzaffarabad", "106,093"],
             ["Subtotal AJK", "=SUM(ABOVE)"], ["**Grand Total**", "**=TOTAL(ABOVE)**"]]
    fill_formulas(typed)
    assert typed[5][1] == "106,093" and typed[6][1] == "**263,981**"


def test_spreadsheet_formulas_are_read_back_worked_out(tmp_path):
    """Seen live: openpyxl saves a formula without its result, so a verifier reading the
    file saw every total empty and failed a correct spreadsheet."""
    import openpyxl
    from ingest.mcp_server import sheet_values
    path = tmp_path / "revenue.xlsx"
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = "Revenue"
    for row in (["Branch", "Q1", "Q2", "Total"], ["Gilgit", 40406, 29003, "=SUM(B2:C2)"],
                ["Astore", 6555, -4238, "=b3+c3"], ["All", "=SUM(B2:B3)", "=SUM(C2:C3)",
                                                    "=SUM(D2:D3)"],
                ["Share", "=ROUND(B2/B4*100, 1)", "=IF(C4>0, 1, 0)", "=1/0"]):
        ws.append(row)
    book.create_sheet("Check").append(["='Revenue'!D4 - Revenue!D2"])
    book.save(path)
    values = sheet_values(path)
    rows = values["Revenue"]
    assert rows[1][3] == 69409 and rows[2][3] == 2317          # lower case refs too
    assert rows[3][1:] == [46961, 24765, 71726]                 # totals of totals
    assert rows[4][1] == 86.0 and rows[4][2] == "=IF(C4>0, 1, 0)" and rows[4][3] == "=1/0"
    assert values["Check"][0][0] == 2317                        # another sheet's cells
    assert "Gilgit | 40406 | 29003 | 69409" in read_any(path)


def test_a_fixer_writes_only_its_file_and_starts_it_afresh(tmp_path):
    """Seen live: fixers made "_v2" copies and added sheets beside the wrong ones."""
    from ingest.agents.harness import only_reading, only_writing
    target = tmp_path / "revenue.xlsx"
    target.write_bytes(b"old")
    seen = []

    def call(name, args):
        seen.append((name, target.exists()))
        return "done"
    fixing = only_writing(call, target)
    assert "fix only revenue.xlsx" in fixing("create_word_document", {"filename": "rev_v2.docx"})
    assert fixing("read_file", {"filename": "revenue.xlsx"}) == "done"
    assert seen == [("read_file", True)]               # reading the old file is fine
    fixing("copy_table", {"filename": "revenue.xlsx"})
    assert seen[-1] == ("copy_table", False)            # its first save starts afresh
    fixing.restore()                                    # it saved nothing: the old file is back
    assert target.read_bytes() == b"old"
    checking = only_reading(lambda name, args: "text", target)
    assert "you check only revenue.xlsx" in checking("read_file", {"filename": "other.docx"})
    assert checking("read_file", {"filename": "Revenue.xlsx"}) == "text"


def test_word_title_is_not_repeated(tmp_path):
    from ingest.mcp_server import write_docx
    path = tmp_path / "a.docx"
    write_docx(path, "Revenue", "# Revenue\n\nText.\n\n## Part")
    text = read_any(path)
    assert text.count("Revenue") == 1 and "## Part" in text


def test_files_stay_inside_the_output_folder(tmp_path):
    assert safe_file(tmp_path, "../../evil.docx", {".docx"}) == tmp_path / "evil.docx"
    with pytest.raises(ValueError):
        safe_file(tmp_path, "report.exe", {".docx"})
