"""Structured logging (NFR-07): JSON lines to logs/ingest.jsonl, readable lines to the console."""
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("ingest")

_EXTRA_FIELDS = ("file", "stage", "duration_ms", "error", "warning", "chunks", "status")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        bits = [f"{record.levelname:<7} {record.getMessage()}"]
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                bits.append(f"{key}={getattr(record, key)}")
        return "  ".join(bits)


def setup_logging(logs_dir: Path, level: str = "INFO") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log.handlers.clear()
    log.setLevel(level)
    log.propagate = False

    file_handler = logging.FileHandler(logs_dir / "ingest.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    console = logging.StreamHandler()
    console.setFormatter(ConsoleFormatter())
    log.addHandler(file_handler)
    log.addHandler(console)

    # Model libraries are chatty at INFO
    for noisy in ("docling", "httpx", "faster_whisper", "sentence_transformers",
                  "transformers", "huggingface_hub", "google_genai", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # RapidOCR sets its own logger to INFO when it loads, so raising the level is not enough
    logging.getLogger("RapidOCR").disabled = True


@contextmanager
def timed_stage(stage: str, file: str, timings: dict[str, float]):
    """Log a stage's start, duration and any error; record seconds in `timings`."""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        ms = round((time.perf_counter() - start) * 1000)
        log.error("stage failed", extra={"file": file, "stage": stage,
                                         "duration_ms": ms, "error": f"{type(exc).__name__}: {exc}"})
        raise
    ms = round((time.perf_counter() - start) * 1000)
    timings[stage] = ms / 1000
    log.debug("stage done", extra={"file": file, "stage": stage, "duration_ms": ms})
