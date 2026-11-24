"""View what is stored: print chunks with their embeddings, or export everything to JSON.

The database file itself can't be read directly: local Qdrant saves each chunk as a pickled
Python object, so a SQLite viewer only shows binary blobs.
"""
import json
from datetime import datetime
from pathlib import Path

from ingest.store import VectorStore, dense_vector


def location(payload: dict) -> str:
    if payload.get("page_number") is not None:
        return f"page {payload['page_number']}"
    if payload.get("start_time") is not None:
        s, e = payload["start_time"], payload["end_time"]
        return f"{int(s // 60)}:{s % 60:04.1f}-{int(e // 60)}:{e % 60:04.1f}"
    return ""


def _reading_order(point) -> tuple:
    p = point.payload
    return (p["source_path"], p.get("page_number") or 0, p.get("start_time") or 0.0)


def _matching(store: VectorStore, file_filter: str | None) -> list:
    """All chunks (optionally only from files whose path contains `file_filter`), grouped by file."""
    needle = (file_filter or "").lower()
    points = [pt for pt in store.iter_points() if needle in pt.payload["source_path"].lower()]
    return sorted(points, key=_reading_order)


def format_vector(vector: list[float], full: bool = False) -> str:
    if full:
        return "[" + ", ".join(f"{x:.6f}" for x in vector) + "]"
    head = ", ".join(f"{x:.4f}" for x in vector[:8])
    return f"{len(vector)} numbers [{head}, ... {len(vector) - 8} more]"


def show(store: VectorStore, limit: int = 5, file_filter: str | None = None,
         full: bool = False) -> int:
    points = _matching(store, file_filter)
    for i, point in enumerate(points[:limit], 1):
        p = point.payload
        name = p["source_path"].rsplit("/", 1)[-1]
        where = " | ".join(x for x in [name, p.get("section_heading") or "", location(p),
                                        p["modality"]] if x)
        text = p["content"].replace("\n", " ")
        print(f"\n[{i}] {where}")
        print(f"    chunk_id:  {p['chunk_id']}")
        print(f"    text:      {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"    embedding: {format_vector(dense_vector(point), full)}")
    if not points:
        print("No chunks stored" + (f" for files matching '{file_filter}'." if file_filter else "."))
    elif len(points) > limit:
        print(f"\n(showing {limit} of {len(points)} chunks)")
    return min(limit, len(points))


def _chunk_json(point) -> str:
    # One field per line, but the embedding on a single line so the file stays readable
    fields = [f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
              for k, v in point.payload.items()]
    fields.append(f'    "embedding": {format_vector(dense_vector(point), full=True)}')
    return "  {\n" + ",\n".join(fields) + "\n  }"


def export(store: VectorStore, exports_dir: Path, out: Path | None = None,
           file_filter: str | None = None) -> tuple[Path, int]:
    points = _matching(store, file_filter)
    path = out or exports_dir / f"chunks-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(_chunk_json(pt) for pt in points)
    path.write_text(f"[\n{body}\n]\n" if points else "[]\n", encoding="utf-8")
    return path.resolve(), len(points)
