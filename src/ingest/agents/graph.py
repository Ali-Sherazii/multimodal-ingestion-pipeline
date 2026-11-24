"""The agent harness as a LangGraph graph (agent_engine: langgraph, the default).

    START -> planner -> worker (again for each plan step) -> find_files -> verifier
    verifier: passed -> next_file | failed -> fixer -> verifier (at most agent_verify_rounds)
    next_file -> verifier (the next file) | END (all checked)

Each node is one agent (planner, worker, verifier, fixer) or a bookkeeping step. The graph's
state holds only what travels between agents: the plan, the checklist and every agent's
report. An agent's own memory (its messages and tool results) never enters it, and is
deleted once the agent has reported (Agent.forget). The agents, prompts, MCP tools and the
verifier's and fixer's limits are the same as in harness.py; LangGraph runs the order.

After every node the state is saved (SQLite, beside app_db), so a task stopped half-way (the
model's quota, the internet, Ctrl+C) carries on from the last finished node:
`ingest agent --resume ID`. `ingest agent --graph` prints this graph as a Mermaid diagram.
"""
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ingest.agents.harness import (VERDICT, VERIFIER, WORKER, Agent, Plan, Report, TaskResult,
                                   _short, files_made, fixer_goal, make_plan, only_reading,
                                   only_writing, open_tools, task_result, verifier_goal,
                                   worker_goal)
from ingest.llm import LLMUnavailable, make_agent_llm
from ingest.logs import log
from ingest.users import LOCAL, User


class TaskState(TypedDict, total=False):
    """What the graph saves after each node: plain data only, so it can be resumed."""
    task: str
    before: dict          # the output folder's files (name: modified time) before the task
    plan: dict
    checklist: str
    step: int             # the next plan step to run
    reports: list         # every agent's report, in order
    ok: bool              # every worker finished its step
    files: list           # names of the files made, to check
    current: int          # the file being checked
    attempt: int          # its check-and-fix round
    passed: bool          # the last verdict


def build_graph(settings, folder: Path, call, toolset, llm, say: Callable[[str], None],
                checkpointer=None):
    """The graph. `call` runs a tool, `toolset(kind)` lists an agent kind's tools (both from
    harness.open_tools); nothing here runs until the graph is invoked."""
    rounds = settings.agent_verify_rounds
    turns = settings.agent_max_turns

    def reports_of(state) -> list[Report]:
        return [Report(**r) for r in state.get("reports", [])]

    def planner(state: TaskState) -> dict:
        plan, checklist = make_plan(state["task"], llm, settings, say)
        return {"plan": plan.model_dump(), "checklist": checklist, "step": 0, "reports": [],
                "ok": True}

    def worker(state: TaskState) -> dict:
        """An agent made for this step: it works, reports, and its memory is deleted."""
        step = Plan.model_validate(state["plan"]).steps[state["step"]]
        n = state["step"] + 1
        say(f"[agent {n}: {step.name}] started")
        report = Agent(f"agent {n}", WORKER,
                       worker_goal(state["task"], step, state["checklist"], reports_of(state)),
                       toolset(step.kind), call, llm, turns, say).run()
        say(f"[agent {n}: {step.name}] reported: {_short(report.summary, 200)}")
        return {"step": n, "reports": state["reports"] + [asdict(report)],
                "ok": state["ok"] and report.ok}

    def more_steps(state: TaskState) -> str:
        return "next step" if state["step"] < len(state["plan"]["steps"]) else "all done"

    def find_files(state: TaskState) -> dict:
        return {"files": [p.name for p in files_made(folder, state["before"])],
                "current": 0, "attempt": 0}

    def any_files(state: TaskState) -> str:
        return "check" if state["files"] else "none made"

    def verifier(state: TaskState) -> dict:
        """A fresh read-only agent checks one file against the checklist and the documents."""
        path = folder / state["files"][state["current"]]
        verdict = Agent(f"verifier ({path.name})", VERIFIER,
                        verifier_goal(state["task"], path.name, state["checklist"]),
                        toolset("verify"), only_reading(call, path), llm, turns, say,
                        ending=VERDICT).run()
        say(f"[verifier] {path.name}: {verdict.summary}")
        return {"reports": state["reports"] + [asdict(verdict)], "passed": bool(verdict.passed)}

    def judged(state: TaskState) -> str:
        """Failed with fix rounds left: to the fixer. Passed (or out of rounds): done with it."""
        if not state["passed"] and state["attempt"] < rounds:
            return "failed"
        return "passed / out of fix rounds"

    def fixer(state: TaskState) -> dict:
        """A fresh agent remakes the failed file; it may write only that file."""
        path = folder / state["files"][state["current"]]
        problems = state["reports"][-1]["problems"]
        fixing = only_writing(call, path)
        report = Agent(f"fixer ({path.name})", WORKER,
                       fixer_goal(state["task"], path.name, problems, state["checklist"],
                                  reports_of(state)),
                       toolset("write"), fixing, llm, turns, say).run()
        fixing.restore()
        say(f"[fixer] {path.name}: {_short(report.summary, 200)}")
        return {"reports": state["reports"] + [asdict(report)], "attempt": state["attempt"] + 1}

    def next_file(state: TaskState) -> dict:
        return {"current": state["current"] + 1, "attempt": 0}

    def more_files(state: TaskState) -> str:
        return "next file" if state["current"] < len(state["files"]) else "all checked"

    graph = StateGraph(TaskState)
    for name, node in (("planner", planner), ("worker", worker), ("find_files", find_files),
                       ("verifier", verifier), ("fixer", fixer), ("next_file", next_file)):
        graph.add_node(name, node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "worker")
    graph.add_conditional_edges("worker", more_steps,
                                {"next step": "worker", "all done": "find_files"})
    graph.add_conditional_edges("find_files", any_files, {"check": "verifier", "none made": END})
    graph.add_conditional_edges("verifier", judged, {"failed": "fixer",
                                                     "passed / out of fix rounds": "next_file"})
    graph.add_edge("fixer", "verifier")
    graph.add_conditional_edges("next_file", more_files,
                                {"next file": "verifier", "all checked": END})
    return graph.compile(checkpointer=checkpointer)


def diagram(settings) -> str:
    """The graph as Mermaid text (paste into a Markdown file, or any Mermaid viewer)."""
    return build_graph(settings, Path("."), None, None, None, lambda line: None
                       ).get_graph().draw_mermaid()


def checkpoint_file(settings) -> Path:
    return Path(settings.app_db).parent / "agent_runs.db"


def run_graph(task: str, pipeline, user: User = LOCAL, out_dir: Path = Path("outputs"),
              echo: Callable[[str], None] = print, llm=None,
              resume: str | None = None) -> TaskResult:
    """Run a task through the graph, or carry on with a stopped one (`resume` = its id)."""
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
    saved_at = checkpoint_file(settings)
    saved_at.parent.mkdir(parents=True, exist_ok=True)
    run_id = resume or uuid.uuid4().hex[:8]
    # A user's runs are theirs: another user's id doesn't resume
    config = {"configurable": {"thread_id": f"{user.name}:{run_id}"}, "recursion_limit": 500}
    connection = sqlite3.connect(str(saved_at), check_same_thread=False)
    tools, call, toolset = open_tools(settings, folder, pipeline, user)
    calls_before = llm.usage.calls
    graph = build_graph(settings, folder, call, toolset, llm, say, SqliteSaver(connection))
    try:
        if resume:
            saved = graph.get_state(config)
            if not saved.values:
                return TaskResult(False, f"No stopped task {run_id} for {user.name}", log=lines)
            task = saved.values["task"]
            if saved.next:
                say(f"[graph] task {run_id}: carrying on at {', '.join(saved.next)}")
                state = graph.invoke(None, config)
            else:
                say(f"[graph] task {run_id} had already finished")
                state = saved.values
        else:
            say(f"[graph] task {run_id} (LangGraph; if it stops, carry on with: "
                f"ingest agent --resume {run_id})")
            before = {p.name: p.stat().st_mtime for p in folder.iterdir() if p.is_file()}
            state = graph.invoke({"task": task, "before": before}, config)
    except Exception as exc:  # the model, the internet...: what is done so far stays saved
        reason = str(exc) if isinstance(exc, LLMUnavailable) else f"{type(exc).__name__}: {exc}"
        say(f"[stopped] {reason}")
        reports = [Report(**r) for r in graph.get_state(config).values.get("reports", [])]
        return TaskResult(False, f"Stopped: {reason}\nCarry on from the last finished step "
                          f"with: ingest agent --resume {run_id}", reports=reports, log=lines,
                          calls=llm.usage.calls - calls_before)
    finally:
        tools.close()
        connection.close()
    return task_result(state["plan"]["deliverable"], folder, state["before"],
                       [Report(**r) for r in state.get("reports", [])], state.get("ok", True),
                       lines, llm.usage.calls - calls_before, say)
