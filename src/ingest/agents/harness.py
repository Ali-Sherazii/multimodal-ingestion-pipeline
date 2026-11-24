"""Planner, agents, verifiers and the tools they reach through MCP."""
import asyncio
import concurrent.futures
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ingest.llm import LLMUnavailable, make_agent_llm
from ingest.logs import log
from ingest.users import LOCAL, User

PLANNER = """You plan a task for a team of agents that work on the user's own documents.
Split it into 1 to {steps} steps, done in order by separate agents. Each agent sees only its
own goal and the reports of the agents before it, so write each goal in full.
Kinds of step:
- "research": find the facts needed in the documents (searching, reading).
- "write": make the requested file(s) from the facts (Word, Excel, Markdown, CSV, JSON).
- "code": calculate or analyse with Python (totals, growth rates, charts' numbers).
Usually one research step, then one write step; add a code step only for real calculations.
Put every requirement of the task (file type and name, sections, tables, figures) into the
goal of the step that makes the file. `requirements`: a checklist of everything the finished
files must contain, one checkable item each, naming the file it applies to, e.g. "Word file:
a table of each branch's quarterly revenue", "Word file: each branch's full-year total",
"Spreadsheet: each branch's full-year total" (spell out what "the same" or "totals" means
for every file). `deliverable`: what the user gets at the end. `steps`, `requirements` and
`deliverable` are three separate parts of the plan: don't write the checklist or the
deliverable inside a step's goal."""

WORKER = """You are one agent in a team working on the user's documents. Reach your goal by
calling your tools. Use only facts found with search_documents / read_document or given in
the handover; never invent names or figures; keep figures and units exactly as written, and
name their source (file, page) where it helps the reader. Search with several specific
queries if the first finds too little. search_documents shows the best few passages; a table
or section spread over several of them (pages 3, 4, 5...) is read whole, in order, with
read_document(source, first_page, last_page). To put a table from the documents into a file,
use copy_table, never retype its rows: it copies every row and figure exactly, can pivot it
(e.g. a column per quarter) and adds row totals, subtotals and a grand total itself
(list_tables shows a document's tables and their columns). Never work out a figure in your head: in a
file, write each total as a formula (Word tables: =SUM(LEFT), =SUM(ABOVE), =TOTAL(ABOVE);
spreadsheets: =SUM(D2:G2)) so the exact value is filled in; anywhere else, get it from
calculate (many expressions in one call). run_python can't open documents or files: use it
only for bigger computations, with the figures pasted into the code. You have few turns:
once you have the facts, make the file, meeting every item of the requirements checklist.
Make files only if your goal asks for them, with the exact file name asked for. When done,
call finish: summary (what you did), findings (the facts
found, each with its source), files (names of files you made), handover (what the next agent
needs: for a table, its rows). Call the tools; don't just describe calls."""

VERIFIER = """You check a file another agent made. Read it with read_file. Go through the
requirements checklist item by item: every item that applies to this file must be met (a
total asked for must be in the file, for every row it is asked for); one unmet item fails
the file. Count the rows: when the file copies a table, read the source table whole (from
the page it starts on) and check that every one of its rows is in the file; one missing row
fails the file. Check that its figures match the user's documents: confirm
figures with search_documents or read_document (whole tables: read_document with pages).
Recompute every total, subtotal, difference or percentage in the file with calculate from
the figures it is built from: a single one that differs fails the file. Fail it for wrong
or invented figures, missing requested parts or an empty/garbled file; ignore style and
wording. Then call verdict(passed, problems): problems lists each issue precisely (what is
wrong, where, and the right value)."""

FINISH = {"name": "finish", "description": "Report your work and stop.", "parameters": {
    "type": "object", "required": ["summary", "findings", "files", "handover"],
    "properties": {
        "summary": {"type": "string", "description": "what you did, in 1-3 sentences"},
        "findings": {"type": "array", "items": {"type": "string"},
                     "description": "facts found, each with its source"},
        "files": {"type": "array", "items": {"type": "string"},
                  "description": "names of the files you made"},
        "handover": {"type": "string", "description": "what the next agent needs to know"}}}}

VERDICT = {"name": "verdict", "description": "Give your verdict on the file and stop.",
           "parameters": {"type": "object", "required": ["passed", "problems"], "properties": {
               "passed": {"type": "boolean"},
               "problems": {"type": "array", "items": {"type": "string"},
                            "description": "each problem found (empty if it passed)"}}}}

RUN_PYTHON = {"name": "run_python", "description": (
    "Run Python 3 code in an isolated sandbox (no network, no access to the documents or "
    "files: paste in the figures you need) and get what it prints. numpy and pandas are "
    "installed. Print the results."), "parameters": {
    "type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}}}

KIND_TOOLS = {
    "research": {"search_documents", "read_document", "list_tables", "calculate", "run_python"},
    "code": {"search_documents", "read_document", "list_tables", "copy_table", "calculate",
             "run_python", "create_text_file", "create_spreadsheet", "read_file"},
    "write": {"search_documents", "read_document", "list_tables", "copy_table", "calculate",
              "create_word_document", "create_spreadsheet", "create_text_file", "read_file",
              "list_files"},
    "verify": {"read_file", "search_documents", "read_document", "list_tables", "calculate"},
}


class PlanStep(BaseModel):
    name: str = Field(min_length=1)
    kind: Literal["research", "write", "code"]
    goal: str = Field(min_length=1)


class Plan(BaseModel):
    # Never empty: vLLM holds the Modal models' replies to this form exactly, and Qwen,
    # allowed to, filled it in as {"steps": [], "requirements": [], "deliverable": ""}
    steps: list[PlanStep] = Field(min_length=1)
    # the checklist writers meet and verifiers tick off (Gemma left it out when optional)
    requirements: list[str] = Field(min_length=1)
    deliverable: str = Field(min_length=1)


@dataclass
class Report:
    """What an agent hands back to the parent; its own memory is gone after this."""
    agent: str
    goal: str
    summary: str
    findings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    handover: str = ""
    ok: bool = True
    turns: int = 0
    passed: bool | None = None          # verifiers: the verdict
    problems: list[str] = field(default_factory=list)

    def snapshot(self) -> str:
        """The report as the next agent reads it."""
        lines = [f"{self.agent}: {self.summary}"]
        lines += [f"  - {f}" for f in self.findings]
        if self.files:
            lines.append(f"  files made: {', '.join(self.files)}")
        if self.handover:
            lines.append(f"  for the next agent: {self.handover}")
        return "\n".join(lines)


@dataclass
class TaskResult:
    ok: bool
    summary: str
    files: list[Path] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    calls: int = 0


class McpTools:
    """Tools of MCP servers (our own, in this process, and any configured stdio servers),
    callable from ordinary code: the client sessions live in one asyncio task on a
    background thread, opened and closed there."""

    def __init__(self, servers: list[tuple[str, object]]):
        self.tools: dict[str, tuple[object, str, dict]] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="mcp-client")
        self._thread.start()
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._stop: asyncio.Event | None = None
        self._main = asyncio.run_coroutine_threadsafe(self._serve(servers), self._loop)
        self._ready.result(timeout=120)

    async def _serve(self, servers) -> None:
        from contextlib import AsyncExitStack

        from mcp import Client
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for prefix, server in servers:
                    client = await stack.enter_async_context(Client(server))
                    for tool in (await client.list_tools()).tools:
                        name = f"{prefix}__{tool.name}" if prefix else tool.name
                        schema = {"name": name, "description": tool.description or "",
                                  "parameters": tool.input_schema if hasattr(tool, "input_schema")
                                  else tool.inputSchema}
                        self.tools[name] = (client, tool.name, schema)
                self._ready.set_result(True)
                await self._stop.wait()
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                raise

    def call(self, name: str, args: dict) -> str:
        client, real, _ = self.tools[name]
        result = asyncio.run_coroutine_threadsafe(client.call_tool(real, args),
                                                  self._loop).result(timeout=300)
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        failed = getattr(result, "is_error", None) or getattr(result, "isError", False)
        return f"Error: {text}" if failed else text

    def schemas(self, names: set[str] | None = None) -> list[dict]:
        return [s for n, (_, _, s) in self.tools.items()
                if names is None or n in names or "__" in n]

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._main.result(timeout=30)
        except Exception:  # closing a stdio server can fail; the process ends anyway
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


class Sandbox:
    """run_python: code runs in a Modal sandbox (its own container, no network), so nothing
    an agent writes can touch this machine, the documents or the database."""
    PACKAGES = ("numpy==2.3.4", "pandas==2.3.3")

    def run(self, code: str) -> str:
        try:
            import modal
            app = modal.App.lookup("ingest-sandbox", create_if_missing=True)
            image = modal.Image.debian_slim(python_version="3.12").pip_install(*self.PACKAGES)
            box = modal.Sandbox.create("python", "-c", code, app=app, image=image, timeout=120,
                                       block_network=True, cpu=1.0, memory=1024)
            box.wait()
            out, err = box.stdout.read(), box.stderr.read()
        except Exception as exc:
            return f"Error: the sandbox failed ({type(exc).__name__}: {exc})"
        text = out + (f"\nErrors:\n{err}" if err.strip() else "")
        return (text[:8000] + "\n…(cut)") if len(text) > 8000 else (text or "(printed nothing)")


def _short(value, size: int = 140) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= size else text[:size] + "…"


class Agent:
    """One agent: a goal, its tools, its own memory (the message history), and a report."""

    def __init__(self, name: str, system: str, goal: str, tools: list[dict],
                 call: Callable[[str, dict], str], llm, max_turns: int,
                 echo: Callable[[str], None], ending: dict = FINISH):
        self.name, self.system, self.goal = name, system, goal
        self.tools = [*tools, ending]
        self.ending = ending["name"]
        self.call, self.llm, self.max_turns, self.echo = call, llm, max_turns, echo
        self.memory: list = []

    def run(self) -> Report:
        self.memory = [self.llm.user_message(self.goal)]
        nudged = False
        for turn in range(1, self.max_turns + 1):
            step = self.llm.chat_step(self.memory, self.system, self.tools)
            self.memory.append(step.raw)
            if not step.calls:
                if step.finish not in ("", "STOP"):
                    self.echo(f"  [{self.name}] reply ended without a tool call: {step.finish}")
                if nudged:  # it answered in words twice: take those as its report
                    return self._report(turn, {"summary": step.text or "(no report)"})
                nudged = True
                hint = (" Your last reply was cut off or malformed: split big content over "
                        "several calls (e.g. one sheet or section at a time)."
                        if step.finish in ("MAX_TOKENS", "MALFORMED_FUNCTION_CALL") else "")
                self.memory.append(self.llm.user_message(
                    f"Use your tools; when done, call {self.ending}.{hint}"))
                continue
            results = []
            for call in step.calls:
                if call.name == self.ending:
                    return self._report(turn, call.args)
                try:
                    output = self.call(call.name, call.args)
                except Exception as exc:  # a tool failing is information for the agent
                    output = f"Error: {type(exc).__name__}: {exc}"
                self.echo(f"  [{self.name}] {call.name}({_short(call.args, 90)}) -> "
                          f"{_short(output)}")
                results.append((call, {"result": output}))
            self.memory.append(self.llm.tool_results(results))
        self.echo(f"  [{self.name}] ran out of turns")
        # One last turn, to hand over what it has rather than nothing
        self.memory.append(self.llm.user_message(
            f"You are out of turns. Call {self.ending} now with what you have found so far "
            "(say what is missing)."))
        ending = [t for t in self.tools if t["name"] == self.ending]
        try:
            step = self.llm.chat_step(self.memory, self.system, ending)
        except LLMUnavailable:
            step = None
        for call in (step.calls if step else []):
            if call.name == self.ending:
                return self._report(self.max_turns, call.args, ok=False)
        return self._report(self.max_turns, {"summary": "Stopped: ran out of turns before "
                                             "finishing."}, ok=False)

    def _report(self, turns: int, args: dict, ok: bool = True) -> Report:
        report = Report(self.name, self.goal, str(args.get("summary") or ""),
                        [str(x) for x in args.get("findings") or []],
                        [str(x) for x in args.get("files") or []],
                        str(args.get("handover") or ""), ok, turns,
                        passed=args.get("passed"), problems=[str(x) for x in
                                                             args.get("problems") or []])
        if self.ending == "verdict":
            report.summary = "passed" if report.passed else "failed: " + "; ".join(report.problems)
        self.forget()
        return report

    def forget(self) -> None:
        """The agent's memory is deleted once it has reported; only the report remains."""
        self.memory.clear()


WRITERS = {"create_word_document", "create_spreadsheet", "create_text_file", "copy_table"}


def only_reading(call: Callable[[str, dict], str], path: Path) -> Callable[[str, dict], str]:
    """A verifier's tools: it reads only the file it checks (seen live: a verifier failed a
    correct Word file for problems it found in the spreadsheet)."""
    def run(name: str, args: dict) -> str:
        if name == "read_file" and \
                Path(str(args.get("filename", ""))).name.lower() != path.name.lower():
            return f"Error: you check only {path.name}; read that file."
        return call(name, args)
    return run


class only_writing:
    """A fixer's tools: it may write only the file it fixes, and its first save starts that
    file afresh (seen live: fixers made "_v2" copies, and added sheets next to the wrong
    ones). The old file comes back if the fixer saves nothing."""

    def __init__(self, call: Callable[[str, dict], str], path: Path):
        self.call, self.path, self.old, self.fresh = call, path, None, False

    def __call__(self, name: str, args: dict) -> str:
        if name in WRITERS:
            if Path(str(args.get("filename", ""))).name.lower() != self.path.name.lower():
                return f"Error: fix only {self.path.name}: save your work under that name."
            if not self.fresh:
                self.fresh = True
                if self.path.exists():
                    self.old = self.path.read_bytes()
                    self.path.unlink()
        return self.call(name, args)

    def restore(self) -> None:
        if self.old is not None and not self.path.exists():
            self.path.write_bytes(self.old)


def open_tools(settings, folder: Path, pipeline, user: User):
    """The agents' tools: our MCP server for this user (and any configured stdio servers) and
    run_python in the sandbox. Returns the MCP client (close it when done), a function that
    runs a tool, and one that lists the tools a kind of agent gets."""
    from mcp import StdioServerParameters

    from ingest.mcp_server import build_server
    servers = [("", build_server(folder, pipeline, user))]
    for extra in settings.agent_mcp_servers:
        servers.append((extra["name"], StdioServerParameters(
            command=extra["command"], args=list(extra.get("args") or []))))
    tools = McpTools(servers)
    sandbox = Sandbox() if settings.agent_sandbox else None

    def call(name: str, args: dict) -> str:
        if name == "run_python":
            if sandbox is None:
                return "Error: the sandbox is turned off (agent_sandbox)"
            return sandbox.run(str(args.get("code", "")))
        return tools.call(name, args)

    def toolset(kind: str) -> list[dict]:
        names = KIND_TOOLS[kind]
        found = tools.schemas(names)
        if "run_python" in names and sandbox is not None:
            found.append(RUN_PYTHON)
        return found
    return tools, call, toolset


def make_plan(task: str, llm, settings, say: Callable[[str], None]) -> tuple[Plan, str]:
    """The planner: the steps (at most agent_max_steps) and the requirements checklist, as
    the writers, verifiers and fixers read it."""
    plan = llm.generate(f"The task:\n{task}", system=PLANNER.format(
        steps=settings.agent_max_steps), schema=Plan, temperature=0.0, max_tokens=1500)
    plan = plan.model_copy(update={"steps": plan.steps[:settings.agent_max_steps]})
    say(f"[planner] {len(plan.steps)} step(s): "
        + "; ".join(f"{s.kind}: {s.name}" for s in plan.steps))
    checklist = "\n".join(f"{n}. {item}" for n, item in enumerate(plan.requirements, 1))
    if checklist:
        say("[planner] requirements:\n" + checklist)
        checklist = f"\n\nRequirements checklist (the files must meet every item):\n{checklist}"
    return plan, checklist


def worker_goal(task: str, step: PlanStep, checklist: str, reports: list[Report]) -> str:
    handover = "\n\n".join(r.snapshot() for r in reports)
    return (f"The user's task: {task}\n\nYour step ({step.kind}): {step.goal}"
            + (checklist if step.kind != "research" else "")
            + (f"\n\nReports from the agents before you:\n{handover}" if handover else ""))


def verifier_goal(task: str, name: str, checklist: str) -> str:
    return (f"The user's task: {task}\n\nCheck the file {name} (only this file: checklist "
            "items about other files don't apply to it)." + checklist)


def fixer_goal(task: str, name: str, problems: list[str], checklist: str,
               reports: list[Report]) -> str:
    return (f"The user's task: {task}\n\nThe file {name} failed its check. Make it again, "
            "fixed, under the same name (your first save starts the file afresh; you may make "
            "no other file). The problems:\n" + "\n".join(f"- {p}" for p in problems)
            + checklist + "\n\nWhat the earlier agents found:\n"
            + "\n\n".join(r.snapshot() for r in reports if r.passed is None))


def files_made(folder: Path, before: dict[str, float]) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file()
                  and before.get(p.name) != p.stat().st_mtime)


def task_result(deliverable: str, folder: Path, before: dict[str, float],
                reports: list[Report], ok: bool, lines: list[str], calls: int,
                say: Callable[[str], None]) -> TaskResult:
    """The summary the user gets: each file made and whether it passed its last check."""
    made = files_made(folder, before)
    checked = {r.agent.split("(", 1)[-1].rstrip(")"): r.passed for r in reports
               if r.passed is not None}
    parts = [deliverable or "Done."]
    for path in made:
        parts.append(f"{path.name}: " + ("checked against the documents, passed"
                                         if checked.get(path.name) else
                                         "did NOT pass its check: see the log"))
    if not made:
        parts.append("No file was made.")
        found = [f for r in reports for f in r.findings]
        parts += [f"- {f}" for f in found[:20]]
    say(f"[done] {len(made)} file(s), {calls} model calls")
    return TaskResult(ok and all(checked.get(p.name) for p in made), "\n".join(parts), made,
                      reports, lines, calls)


def run_task(task: str, pipeline, user: User = LOCAL, out_dir: Path = Path("outputs"),
             echo: Callable[[str], None] = print, llm=None,
             resume: str | None = None) -> TaskResult:
    """Run a task with the agents. agent_engine "langgraph" (the default) runs it as a
    LangGraph graph (agents/graph.py: each finished step is saved, so a stopped task can be
    resumed); "custom" runs the same agents in the plain loop below."""
    if pipeline.settings.agent_engine == "langgraph":
        from ingest.agents.graph import run_graph
        return run_graph(task, pipeline, user, out_dir, echo, llm, resume)
    if resume:
        return TaskResult(False, "Resuming a task needs agent_engine: langgraph")
    settings = pipeline.settings
    lines: list[str] = []

    def say(line: str) -> None:
        lines.append(line)
        log.info(line)
        echo(line)
    try:
        llm = llm or make_agent_llm(settings)
    except LLMUnavailable as exc:
        return TaskResult(False, f"The agents need the language model: {exc}", log=lines)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    before = {p.name: p.stat().st_mtime for p in folder.iterdir() if p.is_file()}
    tools, call, toolset = open_tools(settings, folder, pipeline, user)
    calls_before = llm.usage.calls

    reports: list[Report] = []
    try:
        plan, checklist = make_plan(task, llm, settings, say)
        for i, step in enumerate(plan.steps, 1):
            say(f"[agent {i}: {step.name}] started")
            report = Agent(f"agent {i}", WORKER, worker_goal(task, step, checklist, reports),
                           toolset(step.kind), call, llm, settings.agent_max_turns, say).run()
            reports.append(report)
            say(f"[agent {i}: {step.name}] reported: {_short(report.summary, 200)}")

        ok = all(r.ok for r in reports)
        for path in files_made(folder, before):
            for attempt in range(settings.agent_verify_rounds + 1):
                verdict = Agent(f"verifier ({path.name})", VERIFIER,
                                verifier_goal(task, path.name, checklist),
                                toolset("verify"), only_reading(call, path), llm,
                                settings.agent_max_turns, say, ending=VERDICT).run()
                reports.append(verdict)
                say(f"[verifier] {path.name}: {verdict.summary}")
                if verdict.passed or attempt == settings.agent_verify_rounds:
                    break
                fixing = only_writing(call, path)
                fixer = Agent(f"fixer ({path.name})", WORKER,
                              fixer_goal(task, path.name, verdict.problems, checklist, reports),
                              toolset("write"), fixing, llm, settings.agent_max_turns,
                              say).run()
                fixing.restore()
                reports.append(fixer)
                say(f"[fixer] {path.name}: {_short(fixer.summary, 200)}")
    except LLMUnavailable as exc:
        say(f"[stopped] {exc}")
        return TaskResult(False, f"Stopped: {exc}", reports=reports, log=lines,
                          calls=llm.usage.calls - calls_before)
    finally:
        tools.close()
    return task_result(plan.deliverable, folder, before, reports, ok, lines,
                       llm.usage.calls - calls_before, say)
