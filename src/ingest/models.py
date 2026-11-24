"""Data passed between stages. Every extractor returns Segments; everything after is shared."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    VIDEO = "video"


class Modality(str, Enum):
    TEXT = "text"
    TRANSCRIPT = "transcript"
    FRAME_CAPTION = "frame_caption"
    PICTURE_DESCRIPTION = "picture_description"
    TABLE = "table"  # a PDF or Word table with its caption and notes (tables.py)


@dataclass
class Segment:
    """One piece of extracted content with its location (the 'Normalise' format)."""
    text: str
    modality: Modality = Modality.TEXT
    section_heading: str | None = None
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None


class Extraction(list):
    """An extractor's segments, plus warnings about content it could not read reliably."""

    def __init__(self, segments=(), warnings=()):
        super().__init__(segments)
        self.warnings: list[str] = list(warnings)


@dataclass
class DetectedFile:
    path: Path
    file_type: FileType
    document_id: str  # SHA-256 of the file contents


@dataclass
class Chunk:
    """One stored unit. Fields match Section 8 of the requirements."""
    chunk_id: str
    document_id: str
    source_path: str
    file_type: FileType
    content: str
    modality: Modality
    ingested_at: str
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    section_heading: str | None = None
    chunk_index: int | None = None  # position within its document (reading or time order)
    flags: list[str] = field(default_factory=list)  # e.g. "instruction-like text"
    access: list[str] = field(default_factory=list)  # groups allowed to read it (access.py)
    owner: str | None = None    # the user who uploaded it (users.py); None = shared by rules
    chat_id: str | None = None  # the chat it was uploaded into, if any
    embedding: list[float] | None = None
    sparse: dict[int, float] | None = None  # keyword weights per token id (hybrid search)
    bm25: dict[int, float] | None = None    # BM25 word weights (embed.Bm25)

    def payload(self) -> dict:
        """Everything except the vector, as stored alongside it."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "file_type": self.file_type.value,
            "content": self.content,
            "page_number": self.page_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "modality": self.modality.value,
            "section_heading": self.section_heading,
            "ingested_at": self.ingested_at,
            "flags": self.flags,
            "access": self.access,
            "owner": self.owner,
            "chat_id": self.chat_id,
        }


@dataclass
class FileResult:
    path: str
    status: str  # "processed" | "skipped" | "failed"
    file_type: str | None = None
    chunks: int = 0
    reason: str | None = None
    warning: str | None = None  # processed, but some content may be missing or suspicious
    seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
