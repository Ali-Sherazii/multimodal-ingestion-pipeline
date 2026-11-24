"""Command line. `ingest` alone opens an interactive menu; subcommands run one action directly:
ingest run PATH... | sync FOLDER... | reindex | search "query" | chat | agent "task" | show
| export | delete PATH | stats | web | worker | user ... | llm | tune-db | mcp
"""
import argparse
import getpass
import os
import sys
from collections.abc import Callable
from pathlib import Path

from ingest.browse import export, location, show
from ingest.config import Settings, load_settings
from ingest.logs import log, setup_logging
from ingest.retrieve import Retriever

MENU = """
What would you like to do?
  1  Ingest a file or folder
  2  Search the stored content
  3  Delete a file's data
  4  Show stats
  5  View stored chunks and their embeddings
  6  Export all chunks + embeddings to a JSON file
  q  Quit"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ingest", description="Multimodal ingestion pipeline. "
                                "Run with no command for an interactive menu.")
    p.add_argument("--config", help="path to a YAML config file (default: ./config.yaml)")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="ingest files and/or folders")
    run.add_argument("paths", nargs="+")
    run.add_argument("--force", action="store_true",
                     help="re-ingest files even if they are unchanged")
    run.add_argument("--re-extract", action="store_true",
                     help="read files again instead of reusing their saved extraction")
    run.add_argument("--frame-captions", action="store_true",
                     help="also caption video frames (overrides config)")
    where = run.add_mutually_exclusive_group()
    where.add_argument("--gpu", action="store_true",
                       help="run PDFs, video and pictures on a Modal cloud GPU (overrides config)")
    where.add_argument("--local", action="store_true",
                       help="run everything on this machine, e.g. when offline (overrides config)")

    sync = sub.add_parser("sync", help="make the database match folders: ingest new and changed "
                          "files, remove deleted ones, re-apply access.yaml")
    sync.add_argument("folders", nargs="+")
    sync.add_argument("--force", action="store_true", help="re-index files even if unchanged")

    reindex = sub.add_parser("reindex", help="re-chunk and re-embed every saved extraction into the "
                             "current collection, without reading any file again")
    reindex.add_argument("--force", action="store_true",
                         help="also files already in the collection")

    delete = sub.add_parser("delete", help="remove one file's chunks from the database")
    delete.add_argument("path")

    search = sub.add_parser("search", help="the best passages for a question, with citations")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--file-type", choices=["text", "pdf", "docx", "video"])
    search.add_argument("--file", help="only this stored file (its full path, as `list` shows)")
    search.add_argument("--chunks", action="store_true",
                        help="show the matched chunks alone, without the text around them")
    search.add_argument("--groups", help="only what these groups may read, e.g. hr,everyone "
                        "(default: everything, as an administrator)")
    search.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")

    sub.add_parser("stats", help="show how many chunks are stored")
    sub.add_parser("gpu-check", help="start the Modal GPU once and report what it sees")

    show_cmd = sub.add_parser("show", help="print stored chunks with their embeddings")
    show_cmd.add_argument("--file", help="only chunks from files whose path contains this")
    show_cmd.add_argument("--limit", type=int, default=5)
    show_cmd.add_argument("--full", action="store_true", help="print every number of each embedding")

    export_cmd = sub.add_parser("export", help="write all chunks + embeddings to a JSON file")
    export_cmd.add_argument("--file", help="only chunks from files whose path contains this")
    export_cmd.add_argument("--out", type=Path, help="output path (default: exports/chunks-<time>.json)")

    web = sub.add_parser("web", help="open the upload and chunk-browser web page")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--no-browser", action="store_true", help="don't open the browser")
    web.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 in a "
                     "container: add users first (`ingest user add`), so it needs a login, and "
                     "serve it over HTTPS)")

    worker = sub.add_parser("worker", help="server mode: run upload jobs from the Redis queue "
                            "(queue_url)")
    worker.add_argument("--burst", action="store_true", help="stop when the queue is empty")

    chat = sub.add_parser("chat", help="chat with your documents (answers with citations, "
                          "using Gemini)")
    chat.add_argument("--user", help="as this user: only what they may read (default: "
                      "everything, as an administrator)")

    agent = sub.add_parser("agent", help="give a task to the agents: they search the documents, "
                           "write files and check them, e.g. \"a Word summary of Table 3\"")
    agent.add_argument("task", nargs="?")
    agent.add_argument("--resume", metavar="ID",
                       help="carry on with a stopped task from its last finished step")
    agent.add_argument("--graph", action="store_true",
                       help="print the agents' LangGraph graph (Mermaid) and stop")
    agent.add_argument("--user", help="as this user: only what they may read")
    agent.add_argument("--out", type=Path, default=Path("outputs"),
                       help="where files the agents make are saved (default: outputs)")

    user = sub.add_parser("user", help="manage users: once one exists, the web app needs a login")
    user_sub = user.add_subparsers(dest="action", required=True)
    add = user_sub.add_parser("add", help="add a user (asks for a password)")
    add.add_argument("name")
    add.add_argument("--groups", default="", help="groups they belong to, e.g. hr,managers")
    add.add_argument("--admin", action="store_true", help="may read and manage everything")
    user_sub.add_parser("list", help="list users")
    rm = user_sub.add_parser("remove", help="remove a user (their files stay, unreadable to others)")
    rm.add_argument("name")
    pw = user_sub.add_parser("passwd", help="set a user's password")
    pw.add_argument("name")
    grp = user_sub.add_parser("groups", help="set a user's groups")
    grp.add_argument("name")
    grp.add_argument("groups", help="comma-separated, e.g. hr,managers ('' for none)")

    sub.add_parser("llm", help="check the language model (llm_backend: the Gemini key and its "
                               "models, or our own model on Modal)")
    tune = sub.add_parser("tune-db", help="apply config.yaml's vector database settings "
                          "(graph, compression, disk) to the existing collection")
    tune.add_argument("--yes", action="store_true", help="really change the collection")
    sub.add_parser("mcp", help="serve the documents and file tools over MCP (stdio), for agents "
                   "and MCP clients such as Claude Desktop")
    return p


def clean_path(raw: str) -> str:
    """Accept a path typed, pasted or dragged into a terminal.

    PowerShell wraps dragged paths as & 'C:\\My Files\\a.pdf'; cmd uses "C:\\My Files\\a.pdf".
    """
    s = raw.strip()
    if s.startswith("& "):
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s


def _print_hits(hits) -> None:
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        where = " | ".join(x for x in [p["source_path"].rsplit("/", 1)[-1],
                                        p.get("section_heading") or "",
                                        location(p), p["modality"]] if x)
        print(f"\n#{rank}  score={hit.score:.3f}  {where}")
        print("    " + p["content"][:400].replace("\n", "\n    "))


def _print_passages(passages) -> None:
    if not passages:
        print("No results.")
    for rank, passage in enumerate(passages, 1):
        print(f"\n#{rank}  [{passage.citation}]  score={passage.score:.3f}")
        print("    " + passage.text[:1200].replace("\n", "\n    "))


def _storage(pipeline, settings: Settings) -> str:
    if settings.qdrant_url:
        where = settings.qdrant_url
    elif settings.qdrant_path:
        where = str(Path(settings.qdrant_path).resolve())
    else:
        where = "memory"
    return (f"{pipeline.store.count()} chunks (text + metadata + embeddings) in Qdrant "
            f"collection '{settings.collection}' at {where}")


def _run(pipeline, settings: Settings, paths: list[str], force: bool,
         re_extract: bool = False) -> int:
    return _finish(pipeline, settings, pipeline.run(paths, force=force, re_extract=re_extract))


def _finish(pipeline, settings: Settings, report) -> int:
    saved = report.save(settings.reports_dir.resolve())
    print(report.summary())
    print(f"\nDatabase: {_storage(pipeline, settings)}")
    print(f"Report:   {saved}")
    return 1 if report.totals["failed"] else 0


def interactive(pipeline, settings: Settings, ask: Callable[[str], str] = input) -> int:
    print("\nMultimodal Ingestion Pipeline")
    print(f"Database: {_storage(pipeline, settings)}")
    try:
        while True:
            print(MENU)
            choice = ask("> ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                return 0

            if choice == "1":
                path = clean_path(ask("File or folder path (you can drag it here): "))
                if not path:
                    continue
                if not Path(path).exists():
                    print(f"Not found: {path}")
                    continue
                force = ask("Re-ingest even if unchanged? [y/N]: ").strip().lower() in {"y", "yes"}
                _run(pipeline, settings, [path], force)

            elif choice == "2":
                query = ask("Search for: ").strip()
                if query:
                    _print_passages(Retriever(pipeline).search(query, limit=5))

            elif choice == "3":
                path = clean_path(ask("Path of the file to remove from the database: "))
                if path:
                    n = pipeline.delete(path)
                    print(f"Deleted {n} chunk(s)." if n else "Nothing stored for that path.")

            elif choice == "4":
                print(_storage(pipeline, settings))

            elif choice == "5":
                name = ask("Only chunks from a file? Type part of its name, or Enter for all: ")
                show(pipeline.store, limit=5, file_filter=name.strip() or None)

            elif choice == "6":
                path, n = export(pipeline.store, settings.exports_dir)
                print(f"Exported {n} chunk(s) with embeddings to {path}")

            else:
                print("Please type a number from 1 to 6, or q.")
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


def main(argv: list[str] | None = None) -> int:
    # Windows without Developer Mode can't make symlinks; the model cache still works
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not sys.stdout.isatty():  # piped/redirected output on Windows would default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    overrides = {}
    if getattr(args, "frame_captions", False):
        overrides["frame_captions"] = True
    if getattr(args, "gpu", False):
        overrides["gpu_backend"] = "modal"
    if getattr(args, "local", False):
        overrides["gpu_backend"] = "local"
    settings = load_settings(args.config, **overrides)
    setup_logging(settings.logs_dir, settings.log_level)

    if args.command == "gpu-check":  # needs no database or local models
        from ingest import gpu_modal
        for key, value in gpu_modal.gpu_info(settings).items():
            print(f"  {key}: {value}")
        return 0
    if args.command == "user":
        return _users(args, settings)
    if args.command == "llm":
        return _llm_check(settings)

    from ingest.pipeline import Pipeline  # heavy imports after argument errors are reported
    try:
        pipeline = Pipeline(settings)
    except (RuntimeError, ConnectionError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        return _dispatch(args, pipeline, settings)
    finally:
        pipeline.close()


def _dispatch(args, pipeline, settings: Settings) -> int:
    if args.command is None:
        return interactive(pipeline, settings)

    if args.command == "run":
        return _run(pipeline, settings, args.paths, args.force, args.re_extract)

    if args.command == "sync":
        return _finish(pipeline, settings, pipeline.sync(args.folders, force=args.force))

    if args.command == "reindex":
        return _finish(pipeline, settings, pipeline.reindex(force=args.force))

    if args.command == "delete":
        n = pipeline.delete(args.path)
        log.info(f"deleted {n} chunk(s) for {args.path}")
        return 0 if n else 1

    if args.command == "search":
        groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups else None
        if args.chunks or args.no_rerank:
            _print_hits(pipeline.search(args.query, args.limit, args.file_type, groups=groups,
                                        rerank=False if args.no_rerank else None,
                                        source_path=args.file))
        else:
            _print_passages(Retriever(pipeline).search(args.query, args.limit, args.file_type,
                                                       groups=groups, source_path=args.file))
        return 0

    if args.command == "stats":
        print(_storage(pipeline, settings))
        return 0

    if args.command == "show":
        show(pipeline.store, args.limit, args.file, args.full)
        return 0

    if args.command == "export":
        path, n = export(pipeline.store, settings.exports_dir, args.out, args.file)
        print(f"Exported {n} chunk(s) with embeddings to {path}")
        return 0

    if args.command == "web":
        from ingest.web.app import serve
        print(f"Database: {_storage(pipeline, settings)}")
        serve(pipeline, settings, port=args.port, open_browser=not args.no_browser,
              host=args.host)
        return 0

    if args.command == "worker":
        if not settings.queue_url:
            print("Error: set queue_url (e.g. INGEST_QUEUE_URL=redis://localhost:6379/0)",
                  file=sys.stderr)
            return 1
        from ingest.jobs import run_worker
        run_worker(pipeline, settings.queue_url, burst=args.burst)
        return 0

    if args.command == "chat":
        return _chat(pipeline, settings, args.user)

    if args.command == "agent":
        from ingest.agents import run_task
        if args.graph:
            from ingest.agents.graph import diagram
            print(diagram(settings))
            return 0
        if not args.task and not args.resume:
            print("Error: give a task, or --resume ID", file=sys.stderr)
            return 1
        user = _as_user(settings, args.user)
        if user is None:
            return 1
        # Progress lines already reach the console through the log: no second copy
        result = run_task(args.task or "", pipeline, user, args.out.resolve(),
                          echo=lambda line: None, resume=args.resume)
        print(f"\n{result.summary}")
        for path in result.files:
            print(f"  file: {path}")
        return 0 if result.ok else 1

    if args.command == "tune-db":
        from ingest.store import _tuning
        t = _tuning(settings)
        print(f"Collection '{settings.collection}': graph m={settings.hnsw_m}, "
              f"build effort {settings.hnsw_ef_construct}, compression {settings.quantization}, "
              f"vectors on disk: {settings.vectors_on_disk}")
        if settings.bm25 and pipeline.store.count() and not pipeline.store.has_bm25:
            print("Note: this collection has no BM25 word weights and tune-db can't add them: "
                  "set a new `collection` name and run `ingest reindex` (nothing re-read).")
        if not args.yes:
            print("Nothing changed. Run again with --yes to apply (Qdrant rebuilds the index "
                  "in the background; searches keep working meanwhile).")
            return 0
        pipeline.store.apply_tuning()
        print(f"Applied. ({t['fusion']} fusion and search effort apply to every search already.)")
        return 0

    if args.command == "mcp":
        from ingest.mcp_server import serve as serve_mcp
        serve_mcp(pipeline)
        return 0
    return 2


def _app_db(settings: Settings):
    from ingest.appdb import Database
    from ingest.users import Users
    return Users(Database(settings.app_db), settings.session_hours)


def _as_user(settings: Settings, name: str | None):
    """The user to act as (default: the local administrator, who reads everything)."""
    from ingest.users import LOCAL
    if not name:
        return LOCAL
    user = _app_db(settings).get(name.strip().lower())
    if user is None:
        print(f"Error: no user '{name}' (see `ingest user list`)", file=sys.stderr)
    return user


def _password(confirm: bool = True) -> str:
    if not sys.stdin.isatty():  # piped in: one line
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Password (8+ characters): ")
    if confirm and getpass.getpass("Again: ") != first:
        raise ValueError("The two passwords differ")
    return first


def _users(args, settings: Settings) -> int:
    users = _app_db(settings)
    try:
        if args.action == "add":
            user = users.add(args.name, _password(), args.groups.split(","), args.admin)
            print(f"Added {user.name}" + (" (admin)" if user.admin else "")
                  + (f", groups: {', '.join(user.groups)}" if user.groups else "")
                  + ". The web app now asks everyone to log in.")
        elif args.action == "list":
            found = users.all()
            if not found:
                print("No users: the web app works without a login.")
            for u in found:
                print(f"  {u.name:<20} {'admin' if u.admin else '':<6} "
                      f"groups: {', '.join(u.groups) or '-'}")
        elif args.action == "remove":
            print(f"Removed {args.name}." if users.remove(args.name.lower())
                  else f"No user '{args.name}'.")
        elif args.action == "passwd":
            users.set_password(args.name.lower(), _password())
            print("Password changed.")
        elif args.action == "groups":
            if not users.get(args.name.lower()):
                raise ValueError(f"No user '{args.name}'")
            users.set_groups(args.name.lower(), args.groups.split(","))
            print("Groups changed.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _llm_check(settings: Settings) -> int:
    from ingest.llm import LLMUnavailable, make_llm, model_label
    print("Gemini key in .env: " + ("yes" if settings.gemini_api_key else "no"))
    if settings.llm_backend == "modal":
        return _modal_llm_check(settings)
    try:
        llm = make_llm(settings)
        names = llm.models()
        print(f"Models this key may use ({len(names)}): " + ", ".join(names))
        print("If a model is busy, these answer instead: "
              + (", ".join(settings.llm_fallback_models) or "none"))
        for role, model in (("answers", settings.llm_model), ("router", settings.router_model)):
            # room to think first: newer models count their thinking in the reply's tokens
            reply = llm.generate("Reply with the single word: ready", model=model, max_tokens=256)
            used = "" if llm.last_model == model else f" (busy: {llm.last_model} answered)"
            print(f"  {role}: {model} -> {reply!r}{used}")
    except LLMUnavailable as exc:
        print(f"Not usable: {exc}", file=sys.stderr)
        return 1
    print(f"(llm_backend: gemini. Our own model on Modal instead: llm_backend: modal; "
          f"now {model_label(settings)})")
    return 0


def _modal_llm_check(settings: Settings) -> int:
    """llm_backend modal: start the GPU if needed, and ask once."""
    import time

    from ingest.llm import LLMUnavailable, make_llm, model_label
    llm = make_llm(settings)
    print(f"Model: {model_label(settings)}, Modal app '{settings.modal_llm_app}'; "
          + ("Gemini answers whenever it can't" if llm.backup else "no Gemini key, so no backup"))
    print("Starting it if it isn't running (after a quiet spell this takes ~2.5 minutes)...",
          flush=True)
    started = time.monotonic()
    try:
        info = llm.info()
    except Exception as exc:
        print(f"Not usable: the model on Modal didn't start ({type(exc).__name__}: {exc}). "
              "Set it up with: modal run src/ingest/llm_modal.py::download, then "
              "modal deploy src/ingest/llm_modal.py", file=sys.stderr)
        return 1
    print(f"  {info['model']} on {info['gpu']} (vLLM {info['vllm']}): answered after "
          f"{time.monotonic() - started:.0f} s")
    try:
        reply = llm.generate("Reply with the single word: ready", max_tokens=256)
    except LLMUnavailable as exc:
        print(f"Not usable: {exc}", file=sys.stderr)
        return 1
    print(f"  answers, router, memory and agents: {llm.last_model} -> {reply!r}")
    print("The GPU stops by itself 4 minutes after the last request.")
    return 0


def _chat(pipeline, settings: Settings, name: str | None) -> int:
    from ingest.appdb import Database
    from ingest.chat import ChatEngine
    from ingest.chats import Chats
    user = _as_user(settings, name)
    if user is None:
        return 1
    chats = Chats(Database(settings.app_db))
    engine = ChatEngine(pipeline, chats)
    if engine.llm is None:
        print(f"Note: {engine.llm_error}\n(Answers show the matching passages instead.)")
    chat = chats.create(user.name)
    print("Ask about your documents. Empty line or q to quit; /new starts a new chat.")
    try:
        while True:
            question = input("\nyou> ").strip()
            if question.lower() in {"", "q", "quit", "exit"}:
                return 0
            if question == "/new":
                chat = chats.create(user.name)
                print("(new chat)")
                continue
            reply = engine.ask(chat["id"], user, question)
            print(f"\nassistant> {reply['content']}")
            for c in reply["citations"]:
                print(f"   [{c['n']}] {c['citation']}")
            if reply["route"]["kind"] == "follow_up":
                print(f"   (read as: {reply['route']['queries'][0]})")
            if reply.get("notice") and reply["notice"] != engine.llm_error:
                print(f"   Note: {reply['notice']}")  # e.g. today's free quota ran out
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
