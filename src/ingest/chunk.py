"""Chunk (FR-08, FR-09): split segments into overlapping chunks that keep their metadata."""
import re
import uuid
from collections.abc import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.models import Chunk, DetectedFile, Modality, Segment

_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")


def _table_parts(text: str) -> tuple[str, list[str], str] | None:
    """(text above, the table's lines, text below) for text holding one Markdown table: a
    table segment is its caption and introduction, the table, then its notes."""
    lines = text.strip().splitlines()
    start = next((i for i in range(len(lines) - 1) if lines[i].lstrip().startswith("|")
                  and _TABLE_SEPARATOR.match(lines[i + 1].strip())), None)
    if start is None:
        return None
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    if any(line.lstrip().startswith("|") for line in lines[end:]):  # a second table
        return None
    return "\n".join(lines[:start]).strip(), lines[start:end], "\n".join(lines[end:]).strip()


def _split_table(text: str, size: int, count: Callable[[str], int] = len,
                 bare: bool = True) -> list[str] | None:
    """A Markdown table that fits `size` stays whole; a bigger one is split by whole rows,
    repeating the header row in every piece, and its caption, introduction and notes too if
    they take at most half the room (otherwise: the caption in every piece, the rest once).
    `count` measures text: characters by default, or tokens (see chunk_segments).

    Returns None if the text isn't a single Markdown table (with `bare`: and nothing else).
    """
    parts = _table_parts(text)
    if parts is None or (bare and (parts[0] or parts[2])):
        return None
    above, lines, below = parts
    if count(text) <= size:
        return [text.strip()]
    newline = count("\n")  # 1 character, or no tokens
    cost = lambda part: count(part) + 2 * newline if part else 0
    head, body = lines[:2], lines[2:]
    head_len = sum(count(l) + newline for l in head)
    repeat = cost(above) + cost(below) + head_len <= size // 2
    top, tail = (above, below) if repeat else (above.splitlines()[0] if above else "", "")
    room = size - cost(top) - cost(tail) - head_len
    groups, rows, length = [], [], 0
    for row in body:
        if rows and length + count(row) + newline > room:
            groups.append(rows)
            rows, length = [], 0
        rows.append(row)  # a row longer than `size` still goes in whole: never cut a row
        length += count(row) + newline
    groups.append(rows)
    pieces = []
    for k, rows in enumerate(groups):
        first = above if k == 0 else top
        last = below if k == len(groups) - 1 else tail
        pieces.append("\n\n".join(p for p in (first, "\n".join(head + rows), last) if p))
    return pieces


def _merge_transcript(segments: list[Segment], size: int, overlap: int) -> list[Segment]:
    """Join consecutive Whisper segments into ~size-character windows.

    The last few segments of each window (up to `overlap` characters) are repeated
    at the start of the next, so a sentence cut at a boundary still appears whole.
    Windows never span two scenes: speech under one slide stays with that slide's title.
    """
    windows: list[Segment] = []
    current: list[Segment] = []
    length = 0

    def emit():
        windows.append(Segment(text=" ".join(s.text for s in current),
                               modality=Modality.TRANSCRIPT,
                               section_heading=current[0].section_heading,
                               start_time=current[0].start_time,
                               end_time=current[-1].end_time))

    for seg in segments:
        # A new scene (slide) starts a new window, with no overlap carried across it
        new_scene = bool(current) and seg.section_heading != current[0].section_heading
        if new_scene:
            emit()
            current, length = [], 0
        elif current and length + len(seg.text) + 1 > size:
            emit()
            carried, carried_len = [], 0
            for prev in reversed(current):
                if carried_len + len(prev.text) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev.text) + 1
            current, length = carried, carried_len
        current.append(seg)
        length += len(seg.text) + 1
    if current:
        emit()
    return windows


def chunk_segments(segments: list[Segment], file: DetectedFile, source_path: str,
                   ingested_at: str, size: int, overlap: int, table_size: int | None = None,
                   count: Callable[[str], int] = len) -> list[Chunk]:
    """Text in `size`-character chunks. Tables stay whole up to `table_size`, measured by
    `count` (the reranker's tokens: it reads at most 512, heading included), so all rows can
    be compared in one chunk; bigger tables are split by rows. Without `table_size`, tables
    are split at `size` characters."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def table(seg: Segment) -> list[str] | None:
        bare = seg.modality is not Modality.TABLE  # e.g. a video frame's table: only a table
        if table_size is None:
            return _split_table(seg.text, size, bare=bare)
        room = table_size - (count(seg.section_heading) if seg.section_heading else 0)
        return _split_table(seg.text, room, count, bare)

    transcript = [s for s in segments if s.modality is Modality.TRANSCRIPT]
    others = [s for s in segments if s.modality is not Modality.TRANSCRIPT]
    pieces: list[tuple[Segment, str]] = [
        (seg, text) for seg in others
        for text in (table(seg) or splitter.split_text(seg.text))
    ]
    pieces += [(w, w.text) for w in _merge_transcript(transcript, size, overlap)]
    # Stable sort: documents (no timestamps) keep reading order, video interleaves by time
    pieces.sort(key=lambda p: p[0].start_time or 0.0)

    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=file.document_id,
            source_path=source_path,
            file_type=file.file_type,
            content=text,
            modality=seg.modality,
            ingested_at=ingested_at,
            page_number=seg.page_number,
            start_time=seg.start_time,
            end_time=seg.end_time,
            section_heading=seg.section_heading,
            chunk_index=index,
        )
        for index, (seg, text) in enumerate(pieces)
    ]
