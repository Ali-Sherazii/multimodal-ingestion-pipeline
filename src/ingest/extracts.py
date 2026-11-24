"""Saved extractions ("extract once"). Each file's extracted segments are kept on disk, named
by the SHA-256 of its contents. A re-run, a moved or copied file, a new chunk size or a new
embedding model then re-chunks and re-embeds them without repeating the expensive part
(OCR, layout, tables, formulas, Whisper). A saved extraction is reused only if it was made
with the same extraction settings and EXTRACTION_VERSION; otherwise the file is read again.
"""
import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import Settings
from ingest.models import Extraction, FileType, Modality, Segment

# Bump when an extractor changes what it produces. 2: fractions written with words rebuilt
# from OCR positions in video frames. 3: PDF/Word tables rebuilt from their cells with
# caption and notes; diagonal watermarks removed before reading.
EXTRACTION_VERSION = 3

# Settings that change what extraction produces. Not gpu_backend: an extraction made on the
# GPU is reused by a later --local run (and the other way round) rather than redone.
_EXTRACTION_SETTINGS = {
    "pdf_ocr", "pdf_force_full_page_ocr", "pdf_picture_descriptions", "pdf_hidden_text_filter",
    "pdf_watermark_filter",
    "pdf_garbage_retry", "picture_prompt", "docx_image_ocr", "picture_min_px",
    "whisper_model", "gpu_whisper_model", "whisper_language", "video_scenes",
    "video_scene_ocr", "video_tables", "formulas", "scene_min_seconds", "scene_min_change",
    "frame_captions", "frame_interval_seconds", "vision_model", "gpu_vision_model",
    "vision_prompt"}


def extraction_key(settings: Settings) -> str:
    values = settings.model_dump(mode="json", include=_EXTRACTION_SETTINGS)
    values["version"] = EXTRACTION_VERSION
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


_LOCK = threading.RLock()  # uploads are indexed in worker threads: one record change at a time


def copies_of(record: dict) -> list[dict]:
    """Where this content is stored: each {path, owner, access, chat_id}. A record saved
    before copies were kept knows only its source_path; its owner and access are None
    (unknown: `Pipeline.reindex` works them out from the path)."""
    if "copies" in record:
        return [dict(c) for c in record["copies"]]
    return [{"path": record["source_path"], "owner": None, "access": None, "chat_id": None}]


def _segment(row: dict) -> Segment:
    return Segment(**{**row, "modality": Modality(row["modality"])})


class ExtractStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def _path(self, document_id: str) -> Path:
        return self.folder / f"{document_id}.json"

    def _read(self, document_id: str) -> dict | None:
        try:
            return json.loads(self._path(document_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def load(self, document_id: str, key: str | None) -> Extraction | None:
        """The saved extraction, if it was made with these settings (key None: any)."""
        record = self._read(document_id)
        if record is None or (key is not None and record.get("extraction_key") != key):
            return None
        return Extraction([_segment(r) for r in record["segments"]], record.get("warnings", []))

    def _write(self, document_id: str, record: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path(document_id)
        tmp = path.with_suffix(".tmp")  # written whole, then swapped in: never half a file
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def save(self, document_id: str, file_type: FileType, source_path: str,
             extraction: list[Segment], key: str, made_with: str) -> None:
        with _LOCK:
            old = self._read(document_id)
            self._write(document_id, {
                "document_id": document_id, "file_type": file_type.value,
                "source_path": source_path, "extraction_key": key,
                "extraction_version": EXTRACTION_VERSION, "made_with": made_with,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "warnings": list(getattr(extraction, "warnings", [])),
                "segments": [{**asdict(s), "modality": s.modality.value} for s in extraction],
                # Read again: the stored copies stay the same
                "copies": copies_of(old) if old else [],
            })

    def set_source(self, document_id: str, source_path: str) -> None:
        """Record a file's new location after a move or rename."""
        with _LOCK:
            record = self._read(document_id)
            if record is not None and record.get("source_path") != source_path:
                record["source_path"] = source_path
                self._write(document_id, record)

    # ---- the stored copies of this content: who has it, where, and who may read it ----
    # So `ingest reindex` puts each one back as it was: a private upload stays private.

    def _change(self, document_id: str, change) -> None:
        with _LOCK:
            record = self._read(document_id)
            if record is None:
                return
            before = record.get("copies")
            record["copies"] = change(copies_of(record))
            if record["copies"] != before:
                self._write(document_id, record)

    def add_copy(self, document_id: str, path: str, access: list[str], owner: str | None,
                 chat_id: str | None) -> None:
        copy = {"path": path, "owner": owner, "access": list(access), "chat_id": chat_id}
        self._change(document_id, lambda copies: [
            *[c for c in copies if c["path"] != path or
              (c["owner"] != owner and c["access"] is not None)], copy])

    def update_copies(self, document_id: str, path: str, **fields) -> None:
        """Change every copy at this path (access, chat_id), as the database does."""
        self._change(document_id, lambda copies: [
            {**c, **fields} if c["path"] == path else c for c in copies])

    def move_copies(self, document_id: str, old_paths: list[str], path: str,
                    access: list[str]) -> None:
        self._change(document_id, lambda copies: [
            {**c, "path": path, "access": list(access)} if c["path"] in old_paths else c
            for c in copies])

    def forget_copies(self, document_id: str, path: str) -> None:
        """This path's copies were deleted from the database: don't bring them back."""
        self._change(document_id, lambda copies: [c for c in copies if c["path"] != path])

    def records(self) -> Iterator[dict]:
        """Every saved extraction (for `ingest reindex`)."""
        for path in sorted(self.folder.glob("*.json")):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
