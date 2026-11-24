"""Table stress test: do the chunks of a 22-page report full of hard tables (eval/tables,
fictional) still hold what is needed to answer its 35 questions?

For each question in answer_key.json, eval/tables/checks.yaml says what one chunk must
contain: values that must sit in the same table row (the row still names its branch), plus
words elsewhere in the chunk (its header, caption or footnotes). A chunk only counts if it
comes from one of the question's pages. This checks extraction and chunking, without an
embedding model; with --search, each question is also searched for (bge-m3 + reranker) and
must find such a chunk in the top 5.

    python eval/tables_eval.py                 # chunk the saved extraction and check it
    python eval/tables_eval.py --extract       # read the PDF again first (on Modal with the
                                               # config's gpu_backend), and save that extraction
    python eval/tables_eval.py --search        # also search for every question
"""
import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
PDF = TABLES / "kestrel_annual_report_2026.pdf"
SAVED = TABLES / "cache" / "extraction.json"
TOP = 5


def plain(text: str) -> str:
    text = re.sub(r"[‐-―−]", "-", text)  # every kind of dash
    return re.sub(r"\s+", " ", text).strip().lower()


def holds(chunk: dict, evidence: dict) -> bool:
    text = f"{chunk.get('section_heading') or ''}\n{chunk['content']}"
    if not all(plain(w) in plain(text) for w in evidence.get("chunk", [])):
        return False
    row = [plain(w) for w in evidence.get("row", [])]
    return not row or any(all(w in plain(line) for w in row) for line in text.splitlines())


def answered(chunks: list[dict], check: dict, pages: list[int]) -> bool:
    on_page = [c for c in chunks if not pages or c.get("page_number") in pages]
    found = any(holds(c, check) for c in on_page)
    return found and (not check.get("also") or any(holds(c, check["also"]) for c in on_page))


def chunk(extraction, settings) -> list[dict]:
    from ingest.chunk import chunk_segments
    from ingest.models import DetectedFile, FileType
    from ingest.normalise import normalise
    from ingest.rerank import token_counter
    tables = settings.table_chunk_tokens
    model = settings.reranker_model or settings.embedding_model
    count = token_counter(model, settings.model_revisions.get(model)) if tables else len
    detected = DetectedFile(PDF, FileType.PDF, "tables-eval")
    chunks = chunk_segments(normalise(extraction), detected, PDF.name, "-", settings.chunk_size,
                            settings.chunk_overlap, table_size=tables or None, count=count)
    return [{"content": c.content, "section_heading": c.section_heading,
             "page_number": c.page_number, "modality": c.modality.value} for c in chunks]


def search_ranks(extraction, settings, questions) -> list[int | None]:
    """Index the extraction in a throwaway local database and search for every question."""
    from ingest.detect import detect
    from ingest.models import FileResult
    from ingest.pipeline import Pipeline
    with tempfile.TemporaryDirectory(prefix="tables-eval-") as tmp:
        pipeline = Pipeline(settings.model_copy(update={
            "qdrant_url": None, "qdrant_path": Path(tmp) / "qdrant",
            "extracts_dir": Path(tmp) / "extracted", "isolate_extraction": False}))
        try:
            pipeline._index(detect(PDF), PDF.name, extraction, ["everyone"],
                            FileResult(path=PDF.name, status="failed"))
            ranks = []
            for q in questions:
                hits = [{"content": h.payload["content"],
                         "section_heading": h.payload.get("section_heading"),
                         "page_number": h.payload.get("page_number")}
                        for h in pipeline.search(q["question"], limit=TOP)]
                ranks.append(next((i for i, h in enumerate(hits, 1)
                                   if answered([h], q["check"], q["pages"])), None))
            return ranks
        finally:
            pipeline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--extract", action="store_true", help="read the PDF again first")
    parser.add_argument("--extraction", type=Path, default=SAVED,
                        help="a saved extraction to check (default: the last --extract)")
    parser.add_argument("--search", action="store_true", help="also search for every question")
    parser.add_argument("--show", type=int, nargs="*", help="print the chunks for these ids")
    args = parser.parse_args()

    from ingest.config import load_settings
    from ingest.extracts import ExtractStore
    settings = load_settings(HERE.parent / "config.yaml", qdrant_url=None, qdrant_api_key=None)

    if args.extract:
        from ingest.extractors.docling_doc import extract_document
        from ingest.extracts import EXTRACTION_VERSION
        from ingest.models import FileType
        extraction = extract_document(PDF, settings)
        ExtractStore(SAVED.parent).save("extraction", FileType.PDF, str(PDF), extraction,
                                        f"v{EXTRACTION_VERSION}", settings.gpu_backend)
        print(f"extracted again ({settings.gpu_backend}); saved to {SAVED}")
        args.extraction = SAVED
    record = json.loads(args.extraction.read_text(encoding="utf-8"))
    extraction = ExtractStore(args.extraction.parent).load(args.extraction.stem, None)

    key = json.loads((TABLES / "answer_key.json").read_text(encoding="utf-8"))["questions"]
    checks = {c["id"]: c for c in yaml.safe_load((TABLES / "checks.yaml").read_text(encoding="utf-8"))}
    questions = [{**q, "id": i, "check": checks[i]} for i, q in enumerate(key, 1)]

    chunks = chunk(extraction, settings)
    separator = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$", re.M)  # a table's header line
    tables = [c for c in chunks if separator.search(c["content"])]
    headless = [c for c in chunks if c["content"].lstrip().startswith("|")
                and not separator.search(c["content"])]
    watermark = [c for c in chunks if re.search(r"RAFT\s*-\s*CONFIDENTIA", c["content"])]
    for q in questions:
        q["ok"] = answered(chunks, q["check"], q["pages"])
    ranks = search_ranks(extraction, settings, questions) if args.search else None

    print(f"{PDF.name}: extraction v{record.get('extraction_version')} ({record.get('made_with')}), "
          f"{len(chunks)} chunks, {len(tables)} with a table")
    print(f"table pieces without their header row: {len(headless)};  "
          f"chunks with watermark text: {len(watermark)}\n")
    width = max(len(q["question"]) for q in questions)
    for q, rank in zip(questions, ranks or [None] * len(questions)):
        mark = "ok  " if q["ok"] else "MISS"
        found = f"  search: {rank or '-'}" if ranks else ""
        print(f"{mark} {q['id']:2}. {q['question'][:width]:{width}}  [{q['kind']}]{found}")

    by_kind: dict[str, list[bool]] = defaultdict(list)
    for q in questions:
        by_kind[q["kind"]].append(q["ok"])
    print("\nby kind: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by_kind.items())))
    print(f"answerable from the chunks: {sum(q['ok'] for q in questions)}/{len(questions)}")
    if ranks:
        print(f"found by search in the top {TOP}: {sum(r is not None for r in ranks)}/{len(ranks)}")

    for i in args.show or []:
        q = questions[i - 1]
        print(f"\n--- {i}. {q['question']}")
        for c in chunks:
            if c.get("page_number") in q["pages"] and any(
                    plain(w) in plain(c["content"]) for w in q["check"].get("row", [])[:1]):
                print(f"[p{c['page_number']} | {c['section_heading']}]\n{c['content']}\n")

    out = HERE / "results" / f"tables-{datetime.now():%Y%m%d-%H%M}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"extraction_version": record.get("extraction_version"),
                               "chunks": len(chunks), "headless_table_pieces": len(headless),
                               "watermark_chunks": len(watermark),
                               "questions": [{k: q[k] for k in ("id", "question", "kind", "ok")}
                                             for q in questions], "search_ranks": ranks},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
