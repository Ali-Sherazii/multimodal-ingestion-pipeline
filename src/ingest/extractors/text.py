"""Text extractor (FR-03): .txt, .md and .csv, decoded to UTF-8 whatever the source encoding."""
import csv
import io
import re
from pathlib import Path

from charset_normalizer import from_bytes

from ingest.config import Settings
from ingest.models import Segment

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def decode(raw: bytes) -> str:
    """Decode bytes in any common encoding to a str."""
    for bom, encoding in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"),
                          (b"\xfe\xff", "utf-16")):
        if raw.startswith(bom):
            return raw.decode(encoding)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    matches = list(from_bytes(raw))
    if not matches:
        return raw.decode("cp1252", errors="replace")
    best = matches[0]
    # Western text often scores identically as cp1250/cp1252/cp1257 and the detector picks
    # the first. cp1252 is what Windows Notepad saves as "ANSI", so it wins a tie.
    for m in matches:
        if m.encoding == "cp1252" and m.chaos <= best.chaos and m.coherence >= best.coherence:
            return str(m)
    return str(best)


def _markdown_sections(text: str) -> list[Segment]:
    """Split Markdown at headings so each piece knows its nearest heading."""
    segments, heading, lines = [], None, []
    in_code_block = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        match = None if in_code_block else _MD_HEADING.match(line)
        if match:
            if lines:
                segments.append(Segment(text="\n".join(lines), section_heading=heading))
            heading, lines = match.group(1), []
        else:
            lines.append(line)
    if lines:
        segments.append(Segment(text="\n".join(lines), section_heading=heading))
    return segments


def _csv_rows(text: str) -> list[Segment]:
    """Render each row as 'column: value' so a chunk still makes sense without the header row."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [f"{col}: {val}" for col, val in zip(header, row) if val.strip()]
        if pairs:
            lines.append("; ".join(pairs))
    return [Segment(text="\n".join(lines))]


def extract_text(path: Path, settings: Settings) -> list[Segment]:
    text = decode(path.read_bytes())
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _markdown_sections(text)
    if suffix == ".csv":
        return _csv_rows(text)
    return [Segment(text=text)]
