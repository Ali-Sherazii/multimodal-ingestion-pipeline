"""Server mode: uploads become jobs on a Redis queue (RQ), and any number of `ingest worker`
processes, on this or other machines, take them off it.

Compared with the web app's own background thread (laptop mode), jobs survive a restart of
the web app, several workers share the load, and a job whose worker fails is kept in Redis
with its error. Each worker loads its models once and then runs one file per job; the
extraction inside still runs in its own process (isolation.py), so a crash fails one file.
"""
from dataclasses import asdict
from pathlib import Path

from ingest.logs import log

QUEUE_NAME = "ingest"
_pipeline = None  # this worker's pipeline, loaded once by run_worker


def connect(url: str):
    from redis import Redis
    return Redis.from_url(url)


def get_queue(url: str, connection=None):
    from rq import Queue
    return Queue(QUEUE_NAME, connection=connection or connect(url))


def process_file(path: str, options: dict) -> dict:
    """One job, run by a worker: ingest one uploaded file and return its FileResult."""
    if _pipeline is None:
        raise RuntimeError("process_file runs inside `ingest worker`")
    pipeline = _pipeline.with_options(**{k: options[k] for k in
                                         ("frame_captions", "describe_pictures", "use_gpu")
                                         if k in options})
    result = pipeline.ingest_file(Path(path), force=options.get("force", False),
                                  groups=options.get("groups") or None,
                                  owner=options.get("owner"), chat_id=options.get("chat_id"))
    return asdict(result)


def run_worker(pipeline, url: str, burst: bool = False, connection=None) -> None:
    """Take jobs off the queue until stopped (or, with burst, until it is empty).

    RQ's SimpleWorker runs each job in this process rather than forking a new one per job
    (which Windows can't do anyway), so the models stay loaded between jobs.
    """
    global _pipeline
    from rq import SimpleWorker
    _pipeline = pipeline
    connection = connection or connect(url)
    log.info(f"worker waiting for jobs on {url} (queue '{QUEUE_NAME}')")
    SimpleWorker([get_queue(url, connection)], connection=connection).work(burst=burst)
