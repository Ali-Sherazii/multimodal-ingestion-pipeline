"""Extractor registry: one function per file type, all returning list[Segment].

Extractors import their heavy libraries lazily, so a text-only run never loads Docling or Whisper.
"""
from collections.abc import Callable
from pathlib import Path

from ingest.config import Settings
from ingest.models import FileType, Segment

Extractor = Callable[[Path, Settings], list[Segment]]


def get_extractor(file_type: FileType) -> Extractor:
    if file_type is FileType.TEXT:
        from ingest.extractors.text import extract_text
        return extract_text
    if file_type in (FileType.PDF, FileType.DOCX):
        from ingest.extractors.docling_doc import extract_document
        return extract_document
    if file_type is FileType.VIDEO:
        from ingest.extractors.video import extract_video
        return extract_video
    raise ValueError(f"No extractor for {file_type}")
