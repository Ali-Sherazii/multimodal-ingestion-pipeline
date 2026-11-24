"""Retrieval: a question in, the best passages out, each with what is needed to understand it
and where it comes from.

Search (pipeline.search) finds chunks: hybrid search (meaning + keywords), plus the chunks
holding the question's ids and figures as written, re-scored by the cross-encoder. A chunk on
its own is often too little to read, so each match is turned into a passage:

- a document's text: the chunks just before and after it in the same section, joined
  without the overlap repeated (up to passage_context_chars more text);
- a table: the chunk already carries its caption, header and notes (tables.py), so as is;
- a video moment: what was on screen then, and the speech from video_context_seconds
  before to video_context_seconds after it.

Matches already inside an earlier passage are dropped, so the same text never comes twice.
Each passage has a citation: the file, its page or time, and its section, table caption or
slide title.
"""
import re
from dataclasses import asdict, dataclass, field

# ---- ids and figures to look for as written ----

_QUOTED = re.compile(r'"([^"]{3,80})"|“([^”]{3,80})”')
_REFERENCE = re.compile(r"\b(?:Table|Figure|Fig\.|Section|Appendix)\s+\d+[A-Za-z]?\b", re.I)
_EDGES = "\"'()[]{}<>,.;:!?“”"
MAX_EXACT = 4


def exact_terms(query: str) -> list[str]:
    """What a matching chunk should contain as written: quoted phrases, references such as
    "Table 9", and words mixing digits with letters or separators: ids (KF-26-5001, AP-317,
    FY2026) and figures (3,663, 12,300,000, 0.37). Plain words and bare numbers like 2026
    are left to the normal search."""
    found = [a or b for a, b in _QUOTED.findall(query)] + _REFERENCE.findall(query)
    for word in _QUOTED.sub(" ", query).split():
        word = word.strip(_EDGES).rstrip("%")
        if (len(word) >= 3 and re.search(r"\d", word)
                and (re.search(r"[A-Za-z]", word) or re.search(r"\d[,./-]\d", word))):
            found.append(word)
    out: list[str] = []
    for term in found:
        if term.lower() not in (t.lower() for t in out):
            out.append(term)
    return out[:MAX_EXACT]


# ---- passages ----

@dataclass
class Passage:
    text: str                      # the matched chunk with the context around it
    citation: str                  # file, page or time, section / table / slide
    score: float
    source_path: str
    file_type: str
    modality: str
    match: str                     # the matched chunk on its own
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    section_heading: str | None = None
    chunk_ids: list[str] = field(default_factory=list)  # the matched chunk first

    def as_dict(self) -> dict:
        return asdict(self)


def clock(seconds: float | None) -> str:
    s = int(seconds or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _caption(content: str) -> str | None:
    first = content.lstrip().split("\n", 1)[0]
    return first if re.match(r"Table\s+\d", first) and len(first) <= 150 else None


def citation(payload: dict, start: float | None = None, end: float | None = None) -> str:
    """"report.pdf, p. 9 · Table 2: Shipment log" / "lecture.mp4, 1:00-1:09 · Bayes' theorem"."""
    name = payload["source_path"].replace("\\", "/").rsplit("/", 1)[-1]
    if payload.get("start_time") is not None or start is not None:
        where = f"{clock(start if start is not None else payload.get('start_time'))}-" \
                f"{clock(end if end is not None else payload.get('end_time'))}"
    elif payload.get("page_number"):
        where = f"p. {payload['page_number']}"
    else:
        where = ""
    what = _caption(payload.get("content", "")) or payload.get("section_heading") or ""
    return " · ".join(p for p in (f"{name}, {where}" if where else name, what) if p)


def _join(first: str, second: str) -> str:
    """Two consecutive chunks as one text: the overlap they share is written once."""
    for size in range(min(len(first), len(second), 400), 15, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return f"{first}\n{second}"


def _tail(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    cut = text[-size:]
    return "…" + cut[cut.find(" ") + 1:] if " " in cut else "…" + cut


def _head(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    cut = text[:size]
    return (cut[:cut.rfind(" ")] if " " in cut else cut) + "…"


class Retriever:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.store = pipeline.store
        settings = pipeline.settings
        self.context_chars = settings.passage_context_chars
        self.video_seconds = settings.video_context_seconds

    def search(self, query: str, limit: int = 5, file_type: str | None = None,
               groups: list[str] | None = None, source_path: str | None = None,
               modality: str | None = None, context: bool = True, chat_id: str | None = None,
               hybrid: bool | None = None, terms: list[str] = ()) -> list[Passage]:
        hits = self.pipeline.search(query, limit=limit * 3, file_type=file_type, groups=groups,
                                    source_path=source_path, modality=modality,
                                    chat_id=chat_id, hybrid=hybrid, terms=terms)
        passages: list[Passage] = []
        shown: set[str] = set()
        for hit in hits:
            if hit.payload.get("chunk_id", str(hit.id)) in shown:
                continue  # already inside an earlier passage
            passage = self.passage(hit, groups) if context else self._alone(hit)
            shown.update(passage.chunk_ids)
            passages.append(passage)
            if len(passages) == limit:
                break
        return passages

    def _alone(self, hit) -> Passage:
        p = hit.payload
        return Passage(text=p["content"], citation=citation(p), score=float(hit.score),
                       source_path=p["source_path"], file_type=p["file_type"],
                       modality=p["modality"], match=p["content"], page_number=p.get("page_number"),
                       start_time=p.get("start_time"), end_time=p.get("end_time"),
                       section_heading=p.get("section_heading"),
                       chunk_ids=[p.get("chunk_id", str(hit.id))])

    def passage(self, hit, groups: list[str] | None = None) -> Passage:
        p = hit.payload
        if p["modality"] == "table":
            return self._alone(hit)  # caption, header and notes are in the chunk
        if p.get("start_time") is not None:
            return self._moment(hit, groups)
        return self._section(hit, groups)

    def _section(self, hit, groups) -> Passage:
        """The chunks just before and after, if they belong to the same section."""
        passage = self._alone(hit)
        p, index = hit.payload, hit.payload.get("chunk_index")
        if index is None or self.context_chars <= 0:
            return passage
        near = {n.payload.get("chunk_index"): n.payload for n in
                self.store.around(p["document_id"], index - 1, index + 1, groups)}
        same = lambda n: (n and n["modality"] == p["modality"]
                          and n.get("section_heading") == p.get("section_heading"))
        before, after = near.get(index - 1), near.get(index + 1)
        before, after = (before if same(before) else None), (after if same(after) else None)
        text = p["content"]
        room = self.context_chars // (2 if before and after else 1)
        if before:
            text = _join(_tail(before["content"], room + 100), text)
            passage.chunk_ids.append(before.get("chunk_id"))
        if after:
            text = _join(text, _head(after["content"], room + 100))
            passage.chunk_ids.append(after.get("chunk_id"))
        passage.text = text
        return passage

    def _moment(self, hit, groups) -> Passage:
        """A video moment: what was on screen, and the speech around it."""
        p = hit.payload
        start, end = p.get("start_time") or 0.0, p.get("end_time") or p.get("start_time") or 0.0
        middle = (start + end) / 2
        found = self.store.during(p["document_id"], start - self.video_seconds,
                                  end + self.video_seconds, groups)
        found = sorted((n.payload for n in found), key=lambda n: (n.get("start_time") or 0))
        screen = [n for n in found if n["modality"] != "transcript"
                  and (n.get("start_time") or 0) <= middle <= (n.get("end_time") or 0)]
        speech = [n for n in found if n["modality"] == "transcript"]
        if p["modality"] != "transcript" and p.get("chunk_id") not in {n.get("chunk_id") for n in screen}:
            screen.insert(0, p)
        parts, ids = [], [p.get("chunk_id")]
        if screen:
            t0, t1 = min(n["start_time"] for n in screen), max(n["end_time"] for n in screen)
            parts.append(f"On screen ({clock(t0)}-{clock(t1)}):\n"
                         + "\n\n".join(n["content"] for n in screen))
            ids += [n.get("chunk_id") for n in screen]
        if speech:
            said = speech[0]["content"]
            for n in speech[1:]:
                said = _join(said, n["content"])
            limit = len(p["content"]) + self.context_chars if p["modality"] == "transcript" \
                else self.context_chars
            if len(said) > limit and p["modality"] == "transcript":
                at = said.find(p["content"][:60])
                lo = max(0, at - (limit - len(p["content"])) // 2) if at >= 0 else 0
                said = ("…" if lo else "") + _head(said[lo:], limit)
            elif len(said) > limit:
                said = _head(said, limit)
            t0, t1 = speech[0]["start_time"], speech[-1]["end_time"]
            parts.append(f"Speech ({clock(t0)}-{clock(t1)}):\n{said}")
            ids += [n.get("chunk_id") for n in speech]
        times = [n["start_time"] for n in screen + speech] + [start]
        ends = [n["end_time"] for n in screen + speech] + [end]
        passage = self._alone(hit)
        passage.text = "\n\n".join(parts) or p["content"]
        passage.start_time, passage.end_time = min(times), max(ends)
        passage.citation = citation(p, start, end)  # the matched moment, not the whole window
        passage.chunk_ids = list(dict.fromkeys(i for i in ids if i))
        return passage
