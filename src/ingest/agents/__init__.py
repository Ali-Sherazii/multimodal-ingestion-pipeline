"""The agent harness (the supervisor's step 6): agents started at run time for a task.

    run_task("a Word summary of the Kestrel branch revenue table", pipeline, user, folder)

A parent (the orchestrator) plans the task into steps and starts one agent per step. Each
agent works alone with its tools (search the user's documents, make and read files through
MCP, run Python in a sandbox), then reports back: what it did, what it found, the files it
made, and a snapshot of what the next agent needs. Its own memory is then deleted; only the
report travels on. Every file made is checked by a fresh verifier agent against the task and
the documents; if it fails, a fixer agent gets the problems, and the file is checked again.

The order (plan, each step, check, fix, check again) runs as a LangGraph graph (graph.py,
agent_engine: langgraph), saved after every step so a stopped task can be resumed; the
agents themselves are in harness.py.
"""
from ingest.agents.harness import Agent, Report, TaskResult, run_task

__all__ = ["Agent", "Report", "TaskResult", "run_task"]
