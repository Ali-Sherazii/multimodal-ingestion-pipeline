"""Run extraction in a separate worker process (FR-13, NFR-06).

Docling, Whisper and PDFium are native code: when one of them runs out of memory or hits a
bad file it can kill its whole process without a Python error. Running extraction in a
worker process means such a crash fails only that one file; the run continues with a fresh
worker. The worker is reused between files, so models load once, not once per file.
"""
import concurrent.futures as cf
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ingest.config import Settings
from ingest.logs import log


class ExtractorCrashed(Exception):
    """The worker process died or hung while extracting a file."""


def _init_worker(logs_dir: Path, level: str) -> None:
    from ingest.logs import setup_logging
    setup_logging(logs_dir, level)


def _call(extract, path: Path, values: dict):
    """Runs in the worker. The settings arrive as plain values and are rebuilt here: a worker
    can start on newer code than a long-running server (e.g. `ingest web` left open across
    an update), and any setting the server doesn't know about then gets its default instead
    of failing with "'Settings' object has no attribute ..."."""
    return extract(path, Settings.model_validate(values))


class IsolatedExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool: cf.ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> cf.ProcessPoolExecutor:
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                initargs=(self.settings.logs_dir.resolve(), self.settings.log_level))
        return self._pool

    def run(self, extract, path: Path, settings: Settings):
        future = self._ensure_pool().submit(_call, extract, path, settings.model_dump())
        limit = settings.extract_timeout_minutes
        try:
            return future.result(timeout=limit * 60 if limit else None)
        except BrokenProcessPool:
            self._discard(kill=False)
            raise ExtractorCrashed(
                "the extractor process crashed while reading this file, most likely out of "
                "memory. The run continued. Close other programs, lower pdf_page_batch, or "
                "use --gpu, then ingest it again") from None
        except cf.TimeoutError:
            self._discard(kill=True)
            raise ExtractorCrashed(f"extraction took longer than {limit} minutes and was "
                                   "stopped (extract_timeout_minutes)") from None

    def _discard(self, kill: bool) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        if kill:  # a hung worker never finishes on its own
            for process in list(getattr(pool, "_processes", {}).values()):
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        log.warning("extractor worker restarted")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
