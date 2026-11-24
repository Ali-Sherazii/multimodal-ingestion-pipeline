"""Detect (FR-02, FR-12): identify file type from content and extension; hash for de-duplication."""
import hashlib
import zipfile
from pathlib import Path

from charset_normalizer import from_bytes

from ingest.models import DetectedFile, FileType

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


class UnsupportedFile(Exception):
    """The file is not a type this pipeline handles. Reported as skipped, not failed."""


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_docx(path: Path) -> bool:
    # A .docx is a ZIP archive, like .xlsx, .pptx and .zip. Only Word files contain word/document.xml
    try:
        with zipfile.ZipFile(path) as z:
            return "word/document.xml" in z.namelist()
    except zipfile.BadZipFile:
        return False


def _is_video(head: bytes) -> bool:
    is_iso_media = head[4:8] == b"ftyp"            # .mp4 / .mov / .m4v
    is_matroska = head[:4] == b"\x1a\x45\xdf\xa3"  # .mkv / .webm
    return is_iso_media or is_matroska


def _looks_like_text(head: bytes) -> bool:
    if head.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):  # UTF-8/16 byte-order marks
        return True
    if not head:
        return True
    if b"\x00" in head:
        return False
    return from_bytes(head).best() is not None


def detect_type(path: Path) -> FileType:
    with open(path, "rb") as f:
        head = f.read(8192)
    ext = path.suffix.lower()

    if head.startswith(b"%PDF-"):
        return FileType.PDF
    if head.startswith(b"PK\x03\x04") and _is_docx(path):
        return FileType.DOCX
    if _is_video(head):
        return FileType.VIDEO
    if ext in TEXT_EXTENSIONS and _looks_like_text(head):
        return FileType.TEXT

    if ext == ".doc":
        raise UnsupportedFile("legacy .doc is not supported; save it as .docx")
    if ext in {".pdf", ".docx", *VIDEO_EXTENSIONS, *TEXT_EXTENSIONS}:
        raise UnsupportedFile(f"content does not match its {ext} extension (corrupt, encrypted or renamed?)")
    raise UnsupportedFile(f"unsupported file type: {ext or 'no extension'}")


def detect(path: Path) -> DetectedFile:
    if path.stat().st_size == 0:
        raise UnsupportedFile("empty file")
    return DetectedFile(path=path, file_type=detect_type(path), document_id=file_hash(path))
