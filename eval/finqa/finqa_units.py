"""FinQA retrieval units and scoring: pure functions, no models, no Modal (tests import this).

FinQA (Chen et al. 2021, github.com/czyssrs/FinQA) gives, per question, one annual-report page
as `pre_text` + `post_text` (lists of sentences) and `table` (rows; row 0 is used as the
header), and the gold evidence as `qa.gold_inds`: `text_N` = sentence N of pre_text + post_text
joined, `table_N` = row N of `table` (row 0 included: in some tables it is a data row).

The corpus is every unique page of train + dev + test (pages are deduplicated by the page
part of `id`, which equals `filename`). Two kinds of retrieval unit:

  table rows (both variants)  one unit per row of `table`, row 0 included, each other row with
                              row 0's cells prepended as its headers ("header: value | ...")
  text, variant "paper"       sentences packed into passages of at most 512 tokens with about
                              64 tokens of whole sentences carried into the next passage
                              (FinAgent-RAG, arXiv 2605.05409v1 section 4.3)
  text, variant "ours"        the page's sentences, one per line, through the pipeline's own
                              chunker (chunk.py, config.yaml: 400 characters, 100 overlap)

Every unit records the gold-addressable items it holds ("text_3", "table_2"), so scoring is
exact. A sentence is in a chunk when the whole sentence is; one longer than a chunk counts
for each chunk that holds at least half of it. Items no unit holds are reported, not hidden.
"""
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path

GOLD_KEY = re.compile(r"(text|table)_(\d+)")
KS = (5, 10, 20)


def page_of(question_id: str) -> str:
    return question_id.rsplit("-", 1)[0]


def load_pages(data: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Every unique page across the three splits ({page: {sentences, table, split}}), and the
    questions of each split ({split: [{id, page, question, gold}]})."""
    pages: dict[str, dict] = {}
    questions: dict[str, list[dict]] = {}
    for split in ("train", "dev", "test"):
        rows = json.loads((data / f"{split}.json").read_text(encoding="utf-8"))
        questions[split] = []
        for r in rows:
            page = page_of(r["id"])
            pages.setdefault(page, {"sentences": r["pre_text"] + r["post_text"],
                                    "table": r["table"], "split": split})
            questions[split].append({"id": r["id"], "page": page, "question": r["qa"]["question"],
                                     "gold": list(r["qa"]["gold_inds"])})
    return pages, questions


@dataclass
class Unit:
    uid: str                  # "<page>#t<n>" (text unit n) or "<page>#r<i>" (table row i)
    page: str
    kind: str                 # "text" or "table"
    index: int                # position in the page: text units first, then rows
    text: str
    items: list[str] = field(default_factory=list)   # gold-addressable items it holds


def row_units(page: str, table: list[list[str]], first_index: int = 0) -> list[Unit]:
    """One unit per table row; rows after the first carry row 0's cells as headers."""
    if not table:
        return []
    header, out = table[0], []
    for i, row in enumerate(table):
        if i == 0:
            text = " | ".join(c.strip() for c in row if c.strip())
        else:
            text = " | ".join(f"{h.strip()}: {v.strip()}" if h.strip() else v.strip()
                              for h, v in zip_longest(header, row, fillvalue="") if v.strip())
        out.append(Unit(f"{page}#r{i}", page, "table", first_index + i, text, [f"table_{i}"]))
    return out


def paper_passages(page: str, sentences: list[str], count: Callable[[str], int],
                   size: int = 512, overlap: int = 64) -> list[Unit]:
    """Whole sentences packed into passages of at most `size` tokens; each next passage starts
    with the last sentences of the one before that fit in `overlap` tokens (at least one new
    sentence per passage). A sentence longer than `size` is a passage of its own."""
    tokens = [count(s) for s in sentences]
    out, start = [], 0
    while start < len(sentences):
        end, used = start, 0
        while end < len(sentences) and (end == start or used + tokens[end] <= size):
            used += tokens[end]
            end += 1
        text = " ".join(s.strip() for s in sentences[start:end] if s.strip())
        if text:
            out.append(Unit(f"{page}#t{len(out)}", page, "text", len(out), text,
                            [f"text_{j}" for j in range(start, end) if sentences[j].strip()]))
        if end >= len(sentences):
            break
        back, carried = end, 0
        while back - 1 > start and carried + tokens[back - 1] <= overlap:
            back -= 1
            carried += tokens[back]
        start = back
    return out


def our_chunks(page: str, sentences: list[str], size: int = 400, overlap: int = 100) -> list[Unit]:
    """The page's sentences, one per line, through the pipeline's chunker (no normalising: the
    FinQA text is already clean, and character positions must stay exact for the mapping)."""
    from ingest.chunk import chunk_segments
    from ingest.models import DetectedFile, FileType, Segment
    text = "\n".join(sentences)
    spans, at = [], 0
    for s in sentences:
        spans.append((at, at + len(s)))
        at += len(s) + 1
    chunks = chunk_segments([Segment(text)], DetectedFile(Path(page), FileType.PDF, page), page,
                            "-", size, overlap)
    out, cursor = [], 0
    for c in chunks:
        begin = text.find(c.content, cursor)
        if begin < 0:
            begin = text.find(c.content)
        if begin < 0:  # cannot be located: holds no addressable item (counted by the caller)
            out.append(Unit(f"{page}#t{len(out)}", page, "text", len(out), c.content, []))
            continue
        end = begin + len(c.content)
        cursor = begin + 1
        items = []
        for j, (s0, s1) in enumerate(spans):
            if s1 <= s0:
                continue
            inside = max(0, min(end, s1) - max(begin, s0))
            if inside == s1 - s0 or (s1 - s0 > size and inside >= (s1 - s0) / 2):
                items.append(f"text_{j}")
        out.append(Unit(f"{page}#t{len(out)}", page, "text", len(out), c.content, items))
    return out


def build_units(pages: dict[str, dict], variant: str, count: Callable[[str], int] | None = None,
                chunk_size: int = 400, chunk_overlap: int = 100) -> list[Unit]:
    units: list[Unit] = []
    for page in sorted(pages):
        p = pages[page]
        if variant == "paper":
            text = paper_passages(page, p["sentences"], count)
        elif variant == "ours":
            text = our_chunks(page, p["sentences"], chunk_size, chunk_overlap)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        units += text + [u for u in row_units(page, p["table"], len(text)) if u.text]
    return units


def fingerprint(units: list[Unit]) -> str:
    """Identical units on this machine and on Modal, or the run stops."""
    h = hashlib.sha256()
    for u in units:
        h.update(f"{u.uid}\x1f{u.text}\x1e".encode("utf-8"))
    return h.hexdigest()[:16]


def coverage(units: list[Unit], questions: list[dict]) -> dict:
    """Which gold items no unit holds (they can never be found), per question."""
    held: dict[str, set[str]] = {}
    for u in units:
        held.setdefault(u.page, set()).update(u.items)
    missing, malformed = {}, {}
    for q in questions:
        bad = [g for g in q["gold"] if not GOLD_KEY.fullmatch(g)]
        lost = [g for g in q["gold"] if GOLD_KEY.fullmatch(g) and g not in held.get(q["page"], set())]
        if bad:
            malformed[q["id"]] = bad
        if lost:
            missing[q["id"]] = lost
    return {"missing": missing, "malformed": malformed}


def score_question(ranked_items: list[list[str]], gold: list[str], ks=KS) -> dict:
    """ranked_items = the items each ranked unit holds, best first; gold = the question's gold
    items (all of them: one no unit holds counts as missed). Any-hit R@k, item recall R@k
    (the paper's definition: the share of gold evidence in the top k) and MRR@10 (any-hit)."""
    gold_set = set(gold)
    out = {}
    for k in ks:
        found = set().union(*map(set, ranked_items[:k])) & gold_set if ranked_items else set()
        out[f"any@{k}"] = 1.0 if found else 0.0
        out[f"items@{k}"] = len(found) / len(gold_set) if gold_set else 0.0
    first = next((r for r, items in enumerate(ranked_items[:10], 1) if gold_set & set(items)), None)
    out["mrr@10"] = 1 / first if first else 0.0
    return out


def average(rows: list[dict]) -> dict:
    keys = rows[0].keys() if rows else []
    return {k: round(100 * sum(r[k] for r in rows) / len(rows), 1) for k in keys}


def units_to_json(units: list[Unit]) -> list[dict]:
    return [asdict(u) for u in units]
