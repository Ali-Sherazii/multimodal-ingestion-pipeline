"""Web app: upload files, browse their chunks, search, and chat with your documents.

Laptop mode: listens on 127.0.0.1 only, and uploads are processed one at a time by a
background thread, so the page stays responsive while a long PDF or video is ingested.
Server mode (queue_url set, docker-compose.yml): uploads become jobs on a Redis queue that
separate `ingest worker` processes run (jobs.py).

Users (users.py): with none added, one person uses it without logging in. Once users exist,
every request needs a login, and each user sees, searches and chats with only what they may
read: their own uploads, files shared with their groups, and files for everyone.
"""
import queue
import re
import shutil
import threading
import uuid
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from ingest.appdb import Database
from ingest.chat import ChatEngine
from ingest.chats import Chats
from ingest.config import Settings
from ingest.llm import model_label, separate_agent_llm
from ingest.logs import log
from ingest.pipeline import Pipeline, source_key
from ingest.models import FileResult
from ingest.report import RunReport
from ingest.retrieve import Retriever
from ingest.store import dense_vector
from ingest.users import LOCAL, User, Users

COOKIE = "ingest_session"

STATIC = Path(__file__).parent / "static"
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(filename: str | None) -> str:
    """Keep only the file's own name, without folders or characters Windows forbids."""
    name = _UNSAFE.sub("_", Path(filename or "").name).strip(" .")
    if not name:
        raise HTTPException(400, "A file has no name")
    return name


@dataclass
class Job:
    id: str
    files: list[dict]
    options: dict
    status: str = "queued"  # queued | running | done
    report: str | None = None
    owner: str | None = None


class Worker:
    """Runs uploaded files through the pipeline one at a time, on a background thread."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="ingest-worker").start()

    @property
    def busy(self) -> bool:
        return any(job.status != "done" for job in self.jobs.values())

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(reversed(self.jobs.values()))

    def submit(self, paths: list[Path], options: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], options=options, owner=options.get("owner"),
                  files=[{"name": p.name, "source_path": source_key(p), "path": str(p),
                          "status": "queued"} for p in paths])
        self.jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def _loop(self) -> None:
        while True:
            job = self.jobs[self._queue.get()]
            job.status = "running"
            report = RunReport.start()
            try:
                pipeline = self.pipeline.with_options(
                    job.options["frame_captions"], job.options["describe_pictures"],
                    job.options["use_gpu"])
                for f in job.files:
                    f["status"] = "processing"
                    result = pipeline.ingest_file(Path(f["path"]), force=job.options["force"],
                                                  groups=job.options.get("groups") or None,
                                                  owner=job.options.get("owner"),
                                                  chat_id=job.options.get("chat_id"))
                    report.add(result)
                    report.save(self.pipeline.settings.reports_dir.resolve())
                    f.update(status=result.status, chunks=result.chunks, reason=result.reason,
                             warning=result.warning, seconds=result.seconds,
                             file_type=result.file_type)
            except Exception as exc:  # e.g. a model failed to load; ingest_file handles the rest
                log.error("upload job failed", extra={"error": f"{type(exc).__name__}: {exc}"})
                for f in job.files:
                    if f["status"] in ("queued", "processing"):
                        f.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            report.finish()
            job.report = str(report.save(self.pipeline.settings.reports_dir.resolve()))
            job.status = "done"



class QueueFrontend:
    """Server mode: each uploaded file becomes a job on the Redis queue; `ingest worker`
    processes run them. Same interface as Worker, so the endpoints don't care which."""

    def __init__(self, settings: Settings, connection=None):
        from ingest.jobs import get_queue
        self.settings = settings
        self.queue = get_queue(settings.queue_url, connection)
        self.jobs: dict[str, Job] = {}

    @property
    def busy(self) -> bool:
        return any(self._refresh(job).status != "done" for job in self.jobs.values())

    def submit(self, paths: list[Path], options: dict) -> Job:
        from ingest.jobs import process_file
        timeout = int((self.settings.extract_timeout_minutes or 120) * 60 + 1800)
        job = Job(id=uuid.uuid4().hex[:12], options=options, files=[], owner=options.get("owner"))
        for p in paths:
            queued = self.queue.enqueue(process_file, str(p), options, job_timeout=timeout,
                                        result_ttl=7 * 24 * 3600, failure_ttl=7 * 24 * 3600)
            job.files.append({"name": p.name, "source_path": source_key(p), "path": str(p),
                              "status": "queued", "queue_job": queued.id})
        self.jobs[job.id] = job
        return job

    def _refresh(self, job: Job) -> Job:
        if job.status == "done":
            return job
        from rq.job import Job as QueuedJob
        queued = QueuedJob.fetch_many([f["queue_job"] for f in job.files],
                                      connection=self.queue.connection)
        for f, q in zip(job.files, queued):
            state = q.get_status(refresh=True) if q else "lost"
            if state == "started":
                f["status"] = "processing"
            elif state == "finished":
                result = q.return_value() or {}
                f.update({k: result.get(k) for k in
                          ("status", "chunks", "reason", "warning", "seconds", "file_type")})
            elif state in ("failed", "stopped", "canceled", "lost"):
                error = (getattr(q, "exc_info", None) or "the job was lost").strip().splitlines()
                f.update(status="failed", reason=error[-1][:300])
        if all(f["status"] not in ("queued", "processing") for f in job.files):
            report = RunReport.start()
            for f in job.files:
                report.add(FileResult(path=f["source_path"], status=f["status"],
                                      file_type=f.get("file_type"), chunks=f.get("chunks") or 0,
                                      reason=f.get("reason"), warning=f.get("warning"),
                                      seconds=f.get("seconds") or 0.0))
            report.finish()
            job.report = str(report.save(self.settings.reports_dir.resolve()))
            job.status = "done"
        elif any(f["status"] == "processing" for f in job.files):
            job.status = "running"
        return job

    def get(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        return self._refresh(job) if job else None

    def all(self) -> list[Job]:
        return [self._refresh(job) for job in reversed(self.jobs.values())]


def _reading_order(payload: dict) -> tuple:
    index = payload.get("chunk_index")
    # Chunks stored before chunk_index existed fall back to page / time order
    return (index is None, index or 0, payload.get("page_number") or 0,
            payload.get("start_time") or 0.0)


def create_app(pipeline: Pipeline, settings: Settings, db: Database | None = None,
               llm=None) -> FastAPI:
    app = FastAPI(title="Ingestion Pipeline", docs_url="/api/docs", redoc_url=None)

    @app.exception_handler(ConnectionError)
    @app.exception_handler(ResponseHandlingException)
    async def database_unreachable(request, exc):
        """The database couldn't be reached even after trying again: a short answer the page
        shows, and one line in the log, instead of a traceback."""
        log.warning(f"{request.url.path}: database unreachable: {exc}")
        return JSONResponse(status_code=503, content={
            "detail": "The database can't be reached right now (internet connection?). "
                      "Try again in a moment."})
    store = pipeline.store
    worker = QueueFrontend(settings) if settings.queue_url else Worker(pipeline)
    uploads = settings.uploads_dir.resolve()
    uploads.mkdir(parents=True, exist_ok=True)
    db = db or Database(settings.app_db)
    users, chats = Users(db, settings.session_hours), Chats(db)
    engine = ChatEngine(pipeline, chats, llm=llm)
    retriever = Retriever(pipeline)

    def current_user(request: Request) -> User:
        """The logged-in user; with no users set up, the one local user (no login)."""
        if not users.any():
            return LOCAL
        user = users.from_session(request.cookies.get(COOKIE))
        if user is None:
            raise HTTPException(401, "Please log in")
        return user

    def owner_of(source: str) -> str | None:
        for point in store.iter_points(with_vectors=False, source_path=source,
                                       payload=["owner"]):
            return point.payload.get("owner")
        return None

    def readable(user: User, source: str) -> bool:
        return user.reads is None or source in store.files(user.reads)

    def own_chat(user: User, chat_id: str) -> dict:
        chat = chats.get(chat_id, owner=user.name)
        if chat is None:
            raise HTTPException(404, "No such chat")
        return chat

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    # ---- login ----

    @app.get("/api/me")
    def me(request: Request):
        """Who is logged in (None if no one), and whether a login is needed at all."""
        needed = users.any()
        user = users.from_session(request.cookies.get(COOKIE)) if needed else LOCAL
        return {"user": user.as_dict() if user else None, "login_required": needed}

    @app.post("/api/login")
    def login(response: Response, name: str = Body(...), password: str = Body(...)):
        user = users.check(name, password)
        if user is None:
            raise HTTPException(401, "Wrong user name or password")
        response.set_cookie(COOKIE, users.session(user), httponly=True, samesite="lax",
                            max_age=int(settings.session_hours * 3600))
        log.info(f"login: {user.name}")
        return {"user": user.as_dict()}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE)
        return {"ok": True}

    # ---- status, upload, documents ----

    @app.get("/api/status")
    def status(user: User = Depends(current_user)):
        if settings.qdrant_url:
            where, kind = settings.qdrant_url, "Qdrant server"
        elif settings.qdrant_path:
            where, kind = str(Path(settings.qdrant_path).resolve()), "local folder"
        else:
            where, kind = "memory", "in memory"
        chunks = store.count() if user.reads is None else sum(store.files(user.reads).values())
        available = engine.llm is not None  # loads the client, or notes why it can't
        return {"chunks": chunks, "database": where, "database_kind": kind,
                "collection": settings.collection, "uploads_dir": str(uploads),
                "busy": worker.busy, "gpu_backend": settings.gpu_backend,
                "mode": "queue" if settings.queue_url else "local",
                "modal_gpu": settings.modal_gpu,
                "llm": {"available": available, "model": model_label(settings),
                        "router_model": model_label(settings, "router"),
                        "agents_model": model_label(settings, "agents"),
                        "error": engine.llm_error}}

    @app.post("/api/upload")
    def upload(files: list[UploadFile] = File(...), force: bool = Form(False),
               frame_captions: bool = Form(False), describe_pictures: bool = Form(False),
               use_gpu: bool = Form(False), groups: str = Form(""), chat_id: str = Form(""),
               user: User = Depends(current_user)):
        """`groups`: share with these (comma-separated). With users set up, an upload is
        private to its uploader unless shared; without, `groups` (or access.yaml) decides."""
        if chat_id:
            own_chat(user, chat_id)
        shared = [g.strip() for g in groups.split(",") if g.strip()]
        if users.any():
            refused = [g for g in shared if not user.may_share_with(g)]
            if refused:
                raise HTTPException(403, f"You can't share with: {', '.join(refused)}")
            access = [user.private, *shared]
        else:
            access = shared
        folder = uploads / user.name if users.any() else uploads
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for up in files:
            name = safe_name(up.filename)
            stem, suffix, n = Path(name).stem, Path(name).suffix, 2
            while folder / name in paths:  # two files with the same name in one upload
                name, n = f"{stem} ({n}){suffix}", n + 1
            dest = folder / name
            with open(dest, "wb") as out:
                shutil.copyfileobj(up.file, out)
            paths.append(dest)
        job = worker.submit(paths, {"force": force, "frame_captions": frame_captions,
                                    "describe_pictures": describe_pictures, "use_gpu": use_gpu,
                                    "groups": access, "chat_id": chat_id or None,
                                    "owner": user.name if users.any() else None})
        log.info(f"web upload: {len(paths)} file(s) queued", extra={"status": job.id})
        return asdict(job)

    @app.get("/api/search")
    def search(q: str, limit: int = 5, file_type: str | None = None, source: str | None = None,
               groups: str = "", user: User = Depends(current_user)):
        """The best passages for a question, each with its context and citation, from what
        this user may read. `source`: only that file; `groups` (admins only): read as these
        groups would."""
        access = user.reads
        if access is None and groups.strip():
            access = [g.strip() for g in groups.split(",") if g.strip()]
        found = retriever.search(q, max(1, min(limit, 20)), file_type or None, groups=access,
                                 source_path=source or None)
        return [p.as_dict() for p in found]

    @app.get("/api/jobs")
    def jobs(user: User = Depends(current_user)):
        """Uploads in this session, newest first (lets a refreshed page resume showing progress)."""
        return [asdict(job) for job in worker.all()
                if user.reads is None or job.owner == user.name]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, user: User = Depends(current_user)):
        job = worker.get(job_id)
        if job is None or (user.reads is not None and job.owner != user.name):
            raise HTTPException(404, "No such upload")
        return asdict(job)

    @app.get("/api/documents")
    def documents(user: User = Depends(current_user)):
        docs: dict[str, dict] = {}
        for point in store.iter_points(with_vectors=False, groups=user.reads, payload=[
                "source_path", "file_type", "ingested_at", "modality", "owner", "access",
                "chat_id"]):
            p = point.payload
            doc = docs.setdefault(p["source_path"], {
                "source_path": p["source_path"], "name": p["source_path"].rsplit("/", 1)[-1],
                "file_type": p["file_type"], "chunks": 0, "ingested_at": p["ingested_at"],
                "modalities": {}, "owner": p.get("owner"), "access": p.get("access") or [],
                "chat_id": p.get("chat_id")})
            doc["chunks"] += 1
            doc["ingested_at"] = max(doc["ingested_at"], p["ingested_at"])
            doc["modalities"][p["modality"]] = doc["modalities"].get(p["modality"], 0) + 1
        return sorted(docs.values(), key=lambda d: d["ingested_at"], reverse=True)

    @app.get("/api/chunks")
    def chunks(source: str, user: User = Depends(current_user)):
        rows = []
        for point in store.iter_points(with_vectors=True, source_path=source, groups=user.reads):
            vector = dense_vector(point)
            rows.append({**point.payload, "embedding_dim": len(vector),
                         "embedding_preview": [round(x, 4) for x in vector[:8]]})
        if not rows:
            raise HTTPException(404, "No chunks stored for this document")
        rows.sort(key=_reading_order)
        return {"source_path": source, "name": source.rsplit("/", 1)[-1], "chunks": rows}

    @app.delete("/api/documents")
    def delete(source: str, user: User = Depends(current_user)):
        if not readable(user, source):
            raise HTTPException(404, "No such document")
        if user.reads is not None and owner_of(source) != user.name:
            raise HTTPException(403, "Only the person who uploaded it (or an admin) can delete it")
        deleted = pipeline.delete_key(source)
        # Remove our own uploaded copy, but never touch files elsewhere on disk
        path = Path(source)
        if uploads in path.resolve().parents and path.is_file():
            path.unlink()
        log.info(f"web delete: {deleted} chunk(s) for {path.name}")
        return {"deleted": deleted}

    # ---- chat ----

    @app.get("/api/chats")
    def chat_list(user: User = Depends(current_user)):
        return [{k: c[k] for k in ("id", "title", "updated")} for c in chats.of(user.name)]

    @app.post("/api/chats")
    def chat_new(user: User = Depends(current_user)):
        return chats.create(user.name)

    @app.get("/api/chats/{chat_id}")
    def chat_get(chat_id: str, user: User = Depends(current_user)):
        chat = own_chat(user, chat_id)
        files = sorted(engine.files(user, chat_id))
        return {**chat, "messages": chats.messages(chat_id), "files": files}

    @app.patch("/api/chats/{chat_id}")
    def chat_rename(chat_id: str, title: str = Body(..., embed=True),
                    user: User = Depends(current_user)):
        own_chat(user, chat_id)
        chats.rename(chat_id, title.strip() or "New chat")
        return {"ok": True}

    @app.delete("/api/chats/{chat_id}")
    def chat_delete(chat_id: str, user: User = Depends(current_user)):
        own_chat(user, chat_id)
        chats.delete(chat_id)
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/ask")
    def chat_ask(chat_id: str, question: str = Body(...), only_chat_files: bool = Body(False),
                 user: User = Depends(current_user)):
        """Ask in this chat: routed, searched (what this user may read), answered with
        citations; the chat's memory is updated in the background."""
        own_chat(user, chat_id)
        if not question.strip():
            raise HTTPException(400, "Ask a question")
        return engine.ask(chat_id, user, question, only_chat_files=only_chat_files)

    # ---- agents ----

    tasks: dict[str, dict] = {}
    outputs = Path(settings.outputs_dir).resolve()

    def output_folder(user: User) -> Path:
        return outputs / user.name

    @app.post("/api/agent")
    def agent_start(task: str = Body(..., embed=True), user: User = Depends(current_user)):
        """Start agents on a task in the background; follow it with GET /api/agent/{id}."""
        if not task.strip():
            raise HTTPException(400, "Describe the task")
        if any(t["owner"] == user.name and t["status"] == "running" for t in tasks.values()):
            raise HTTPException(409, "Your agents are still working on the last task")
        job = {"id": uuid.uuid4().hex[:12], "owner": user.name, "task": task.strip(),
               "status": "running", "log": [], "summary": None, "files": [], "ok": None}
        tasks[job["id"]] = job

        def work():
            from ingest.agents import run_task
            try:
                # Each step is shown on the page while the agents work, not only at the end
                # the chat's model, unless the agents have one of their own
                result = run_task(job["task"], pipeline, user, output_folder(user),
                                  echo=job["log"].append,
                                  llm=None if separate_agent_llm(settings) else engine.llm)
                job.update(summary=result.summary, ok=result.ok, log=result.log,
                           files=[p.name for p in result.files])
            except Exception as exc:
                log.error("agent task failed", extra={"error": f"{type(exc).__name__}: {exc}"})
                job.update(summary=f"The task failed: {type(exc).__name__}: {exc}", ok=False)
            job["status"] = "done"
        threading.Thread(target=work, daemon=True, name="agents").start()
        return job

    @app.get("/api/agent/{task_id}")
    def agent_status(task_id: str, user: User = Depends(current_user)):
        job = tasks.get(task_id)
        if job is None or job["owner"] != user.name:
            raise HTTPException(404, "No such task")
        return job

    @app.get("/api/outputs")
    def output_list(user: User = Depends(current_user)):
        folder = output_folder(user)
        files = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) \
            if folder.exists() else []
        return [{"name": p.name, "size": p.stat().st_size} for p in files if p.is_file()]

    @app.get("/api/outputs/{name}")
    def output_file(name: str, user: User = Depends(current_user)):
        path = output_folder(user) / safe_name(name)
        if not path.is_file():
            raise HTTPException(404, "No such file")
        return FileResponse(path, filename=path.name)

    return app


def serve(pipeline: Pipeline, settings: Settings, port: int = 8000,
          open_browser: bool = True, host: str = "127.0.0.1") -> None:
    import uvicorn
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"\nIngestion Pipeline web app running at {url}  (press Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()
    uvicorn.run(create_app(pipeline, settings), host=host, port=port,
                log_level="warning")
