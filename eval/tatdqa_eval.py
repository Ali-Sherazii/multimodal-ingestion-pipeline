"""TAT-DQA retrieval benchmark for this pipeline, run on Modal GPUs.

TAT-DQA (Zhu et al. 2022, arXiv 2207.11871; CC-BY-4.0; huggingface.co/datasets/next-tat/TAT-DQA)
asks questions about pages of real companies' annual reports: tables and text, often with
arithmetic. Its test split has 277 documents (1-2 PDF pages each) and 1,663 questions, each
with the values its answer comes from ("facts": for arithmetic, the numbers used).

Two questions are scored for every TAT-DQA question:
  within its document   do the top passages contain all its facts? (search limited to that
                        file: chunking, tables and ranking inside one report)
  across all documents  is the right report among the top passages? (TAT-DQA questions are
                        written about one report and often don't name the company, so this
                        is hard by design)
each for the retrieval layer's variants: passages (matched chunk + context) or chunks alone,
and with or without looking up ids and figures as written (exact_match).

--vidore scores it the way the ViDoRe benchmark does (vidore/tatdqa_test: the same 277 files
and questions), so our numbers sit beside published ones (ColPali, ColQwen ...): its question
list (each distinct question once), documents ranked by their best chunk, nDCG@5. Three set-ups:
bge-m3 dense only (like the papers' text baselines), hybrid + exact, and that plus the reranker.

Where things run (nothing touches the vector database, local or Qdrant Cloud):
  Modal GPUs     download TAT-DQA, read every PDF with the pipeline's own extractor, then
                 index the chunks (bge-m3) in a throwaway local Qdrant inside the container
                 and run every search there (hybrid + exact + reranker, as configured)
  this machine   caches the extractions in eval/cache/, writes eval/results/

    python eval/tatdqa_eval.py --limit 200     # 200 questions (all 277 documents indexed)
    python eval/tatdqa_eval.py                 # all 1,663 questions
    python eval/tatdqa_eval.py --vidore        # document ranking, nDCG@5, ViDoRe's question list
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import modal

from ingest import gpu_modal
from ingest.config import load_settings
from ingest.extracts import EXTRACTION_VERSION

HERE = Path(__file__).resolve().parent
REPO = "next-tat/TAT-DQA"
GOLD, DOCS = "tatdqa_dataset_test_gold.json", "tatdqa_docs_test.zip"
CACHE = HERE / "cache" / f"tatdqa-extract-v{EXTRACTION_VERSION}.jsonl"
VIDORE = "vidore/tatdqa_test"
VIDORE_CACHE = HERE / "cache" / "vidore-tatdqa-questions.json"
BATCH = 28  # documents per GPU container

bench_image = (gpu_modal.base_image
               .pip_install("sentence-transformers==6.1.0", "qdrant-client==1.19.1")
               .add_local_python_source("ingest"))
app = modal.App("ingest-tatdqa", image=bench_image)


def _download(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, name, repo_type="dataset", cache_dir="/tmp/tatdqa"))


# ---- runs on the GPU ----

@app.cls(gpu=["L4", "A10G"],  # an A10G when no L4 is free: they can be scarce
         cpu=4, memory=16384, volumes={"/models": gpu_modal.models_volume},
         timeout=3600, scaledown_window=60, max_containers=10)
class Bench:
    @modal.method()
    def extract(self, names: list[str], settings: dict) -> list[dict]:
        """Read these PDFs from the test archive with the pipeline's extractor."""
        from ingest.extractors.docling_doc import extract_document
        remote = gpu_modal._remote_settings(settings)
        out = []
        with zipfile.ZipFile(_download(DOCS)) as archive, tempfile.TemporaryDirectory() as tmp:
            members = {Path(m).name: m for m in archive.namelist() if m.lower().endswith(".pdf")}
            for name in names:
                record = {"name": name, "segments": [], "warnings": [], "error": None}
                started = time.perf_counter()
                try:
                    path = Path(tmp) / name
                    path.write_bytes(archive.read(members[name]))
                    extraction = extract_document(path, remote)
                    record["segments"] = gpu_modal._segments_out(extraction)
                    record["warnings"] = list(extraction.warnings)
                except Exception as exc:  # one bad file must not lose the rest of the batch
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["seconds"] = round(time.perf_counter() - started, 2)
                out.append(record)
        gpu_modal.models_volume.commit()
        return out

    @modal.method()
    def search(self, questions: list[dict], records_file: str, settings: dict) -> dict:
        """Index every document in a throwaway local Qdrant, then run each question through
        the retrieval layer in each variant. Returns, per question and variant, the rank of
        the first passage from the right document (across all documents), and whether all
        its facts are in the top passage / the top 3 (within its document)."""
        from ingest.retrieve import Retriever
        pipeline, base, indexed = _index_all(_records(records_file), settings)
        has = lambda text, facts: all(_norm(f) in _norm(text) for f in facts)
        out = {"chunks": pipeline.store.count(), "index_seconds": indexed, "variants": {}}
        rerankers = {}
        for variant, (context, exact, reranker) in VARIANTS.items():
            pipeline.settings = base.model_copy(update={
                "exact_match": exact, "reranker_model": reranker or base.reranker_model})
            name = pipeline.settings.reranker_model
            if name not in rerankers:  # load each reranker once
                pipeline._reranker = None
                rerankers[name] = pipeline.reranker
            pipeline._reranker = rerankers[name]
            retriever = Retriever(pipeline)
            rows, started = [], time.perf_counter()
            for q in questions:
                across = retriever.search(q["question"], limit=5, context=context)
                rank = next((i for i, p in enumerate(across, 1) if p.source_path == q["doc"]), None)
                within = retriever.search(q["question"], limit=3, source_path=q["doc"],
                                          context=context)
                facts = q["facts"]
                rows.append({"uid": q["uid"], "rank": rank,
                             "top1": bool(facts) and bool(within) and has(within[0].text, facts),
                             "top3": bool(facts) and has(" ".join(p.text for p in within), facts)})
            out["variants"][variant] = {"rows": rows,
                                        "seconds": round(time.perf_counter() - started, 1)}
        gpu_modal.models_volume.commit()
        return out

    @modal.method()
    def passages(self, questions: list[dict], records_file: str, settings: dict) -> list[list]:
        """Each question's passages from its own document, as the chat would get them
        (answer_passages of them, with context): for eval/rag_eval.py, which has the
        language model answer from them."""
        from ingest.config import Settings
        from ingest.retrieve import Retriever
        pipeline, _, _ = _index_all(_records(records_file), settings)
        retriever = Retriever(pipeline)
        limit = Settings(_env_file=None, **settings).answer_passages
        out = []
        for q in questions:
            found = retriever.search(q["question"], limit=limit, source_path=q["doc"])
            out.append([p.as_dict() for p in found])
        gpu_modal.models_volume.commit()
        return out

    @modal.method()
    def rank_documents(self, questions: list[dict], records_file: str, settings: dict) -> dict:
        """ViDoRe's way of scoring: rank the documents themselves, each by its best chunk
        (ViDoRe keeps each file's best page), and return the rank of the right one."""
        pipeline, _, indexed = _index_all(_records(records_file), settings)
        out = {"chunks": pipeline.store.count(), "index_seconds": indexed, "variants": {}}
        for variant, (hybrid, exact, rerank) in DOC_SETUPS.items():
            rows, started = [], time.perf_counter()
            for q in questions:
                hits = pipeline.search(q["question"], limit=DOC_DEPTH, hybrid=hybrid,
                                       rerank=rerank, exact=exact)
                docs = list(dict.fromkeys(h.payload["source_path"] for h in hits))
                rank = docs.index(q["doc"]) + 1 if q["doc"] in docs else None
                rows.append({"rank": rank, "found": len(docs)})
            out["variants"][variant] = {"rows": rows,
                                        "seconds": round(time.perf_counter() - started, 1)}
        gpu_modal.models_volume.commit()
        return out


def _records(name: str) -> list[dict]:
    """The extractions put on the models volume by upload_records (on the GPU)."""
    gpu_modal.models_volume.reload()
    return json.loads(gzip.decompress(Path("/models", name).read_bytes()))


def _index_all(records: list[dict], settings: dict):
    """A pipeline on a throwaway local Qdrant holding every document's chunks (on the GPU)."""
    from ingest.config import Settings
    from ingest.models import DetectedFile, FileResult, FileType
    from ingest.pipeline import Pipeline
    tmp = Path(tempfile.mkdtemp())
    base = Settings(_env_file=None, **{**settings, "qdrant_url": None, "qdrant_api_key": None,
                                       "qdrant_path": str(tmp / "qdrant"), "device": "cuda",
                                       "extracts_dir": str(tmp / "extracted"),
                                       "reports_dir": str(tmp / "reports"),
                                       "logs_dir": str(tmp / "logs"), "gpu_backend": "local",
                                       "isolate_extraction": False,
                                       "access_file": str(tmp / "none.yaml")})
    pipeline = Pipeline(base)
    started = time.perf_counter()
    for r in records:
        if r["error"] or not r["segments"]:
            continue
        segments = gpu_modal._segments_in(r["segments"])
        result = FileResult(path=r["name"], status="failed")
        pipeline._index(DetectedFile(Path(r["name"]), FileType.PDF, r["name"]), r["name"],
                        segments, ["everyone"], result)
    return pipeline, base, round(time.perf_counter() - started, 1)


DOC_SETUPS = {  # --vidore: name: (hybrid, exact ids and figures, reranker as configured)
    "bge-m3 dense only": (False, False, False),
    "hybrid + exact, no reranker": (True, True, False),
    "hybrid + exact + reranker (as configured)": (True, True, True),
}
DOC_DEPTH = 30  # chunks looked at per question (the reranker's candidates): enough for 5 documents

VARIANTS = {  # name: (passages with context, exact ids and figures, reranker; None = config's)
    "passages + exact (as configured)": (True, True, None),
    "chunks + exact": (False, True, None),
    "passages + exact, bge-reranker-v2-m3": (True, True, "BAAI/bge-reranker-v2-m3"),
    "passages + exact, multilingual MiniLM": (True, True,
                                              "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
}
PARTS = 8  # question batches searched in parallel, each container indexing every document
PART_MINUTES = 40  # a part not back by then is taken as lost (usually ~10 min)
ATTEMPTS, WAIT = 5, 30  # tries when the laptop's connection drops, and seconds between them


def _norm(text: str) -> str:
    text = re.sub(r"[‐-―−]", "-", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()


# ---- runs on this machine ----

def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def load_questions() -> tuple[list[dict], list[str]]:
    from huggingface_hub import hf_hub_download
    gold = json.loads(Path(hf_hub_download(REPO, GOLD, repo_type="dataset")).read_text(encoding="utf-8"))
    questions, docs = [], []
    for d in gold:
        name = f"{d['doc']['uid']}.pdf"  # the archive's file name (test/<uid>.pdf); several
        docs.append(name)                # documents can come from the same annual report
        for q in d["questions"]:
            facts = [f for f in (q.get("facts") or []) if str(f).strip()]
            questions.append({"uid": q["uid"], "question": q["question"], "doc": name,
                              "answer_type": q.get("answer_type"), "facts": facts})
    return questions, sorted(set(docs))


def load_vidore() -> list[dict]:
    """ViDoRe's TAT-DQA questions (the same test split, same files) as its evaluator uses them:
    each distinct question text once, its right document the file of the last row asking it
    (vidore-benchmark's queries2filename). Only questions and file names are fetched, not the
    page images."""
    if not VIDORE_CACHE.exists():
        import urllib.request
        rows, offset, total = [], 0, 1
        while offset < total:
            url = (f"https://datasets-server.huggingface.co/rows?dataset={VIDORE}&config=default"
                   f"&split=test&offset={offset}&length=100")
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        page = json.loads(response.read())
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(10)
            rows += [{"query": r["row"]["query"], "file": r["row"]["image_filename"],
                      "page": r["row"]["page"]} for r in page["rows"]]
            offset, total = offset + 100, page["num_rows_total"]
        VIDORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VIDORE_CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    rows = json.loads(VIDORE_CACHE.read_text(encoding="utf-8"))
    gold = {r["query"]: Path(r["file"]).name for r in rows if r["query"]}  # last row wins
    return [{"question": q, "doc": doc} for q, doc in gold.items()]


def extractions(names: list[str], settings: dict) -> list[dict]:
    """Every document's extraction: from the cache, the rest read on Modal (in parallel)."""
    cached = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cached[r["name"]] = r
    todo = [n for n in names if n not in cached]
    if todo:
        log(f"reading {len(todo)} PDF(s) on Modal ({len(cached)} cached)")
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f, modal.enable_output(), app.run():
            for records in Bench().extract.map(batches, kwargs={"settings": settings}):
                for r in records:
                    cached[r["name"]] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                log(f"  {len(cached)}/{len(names)} read")
    return [cached[n] for n in names]


def retrying(step: str, call):
    """Run call(), again after WAIT seconds when the laptop's connection drops (up to ATTEMPTS)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except (modal.exception.ConnectionError, ConnectionError, OSError, TimeoutError) as exc:
            if attempt == ATTEMPTS:
                raise
            log(f"{step}: connection lost ({type(exc).__name__}: {exc}); trying again in "
                f"{WAIT} s (attempt {attempt + 1} of {ATTEMPTS})")
            time.sleep(WAIT)


def upload_records(records: list[dict]) -> str:
    """Put the extractions on the models volume once (gzipped, ~0.3 MB) rather than sending
    them with every part: the laptop's upload is slow (each 1.3 MB copy took ~2 min) and drops."""
    data = gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0)
    name = f"tatdqa/extract-{hashlib.sha256(data).hexdigest()[:16]}.json.gz"
    local = CACHE.with_suffix(".json.gz")
    local.write_bytes(data)

    def upload():
        with gpu_modal.models_volume.batch_upload(force=True) as batch:
            batch.put_file(str(local), f"/{name}")
    retrying("uploading the extractions", upload)
    return name


def run_parts(method: str, batches: list[list[dict]], records_file: str, settings: dict,
              folder: Path) -> list[dict]:
    """Run each batch of questions on its own GPU, saving each part as it comes back. After a
    dropped connection only the parts not yet saved run again."""
    folder.mkdir(parents=True, exist_ok=True)
    path = lambda i: folder / f"part-{i}-of-{len(batches)}.json"

    def run():
        todo = [i for i in range(len(batches)) if not path(i).exists()]
        if not todo:
            return
        log(f"{len(batches) - len(todo)} of {len(batches)} part(s) saved; running {len(todo)}")
        with modal.enable_output(), app.run():
            bench = getattr(Bench(), method)
            calls = {i: bench.spawn(batches[i], records_file=records_file, settings=settings)
                     for i in todo}
            deadline = time.monotonic() + PART_MINUTES * 60
            while calls:
                for i, call in list(calls.items()):
                    try:
                        part = call.get(timeout=0)
                    except (TimeoutError, modal.exception.TimeoutError):
                        continue
                    path(i).write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
                    del calls[i]
                    log(f"  part {i + 1} of {len(batches)} back")
                if calls and time.monotonic() > deadline:
                    raise TimeoutError(f"{len(calls)} part(s) not back after {PART_MINUTES} min")
                time.sleep(5)
    retrying("searching", run)
    return [json.loads(path(i).read_text(encoding="utf-8")) for i in range(len(batches))]


def pct(values) -> float:
    values = list(values)
    return round(100 * sum(values) / len(values), 1) if values else 0.0


def report(questions: list[dict], result: dict) -> str:
    lines = ["| Variant | right report @1 | @5 | MRR@5 | facts in top passage | in top 3 |",
             "|---|---|---|---|---|---|"]
    by_type: dict[str, dict] = {}
    for variant, data in result["variants"].items():
        rows = data["rows"]
        scored = [r for r, q in zip(rows, questions) if q["facts"]]
        lines.append(f"| {variant} | {pct(r['rank'] == 1 for r in rows)} | "
                     f"{pct(r['rank'] is not None for r in rows)} | "
                     f"{round(100 * sum(1 / r['rank'] for r in rows if r['rank']) / len(rows), 1)} | "
                     f"{pct(r['top1'] for r in scored)} | {pct(r['top3'] for r in scored)} |")
        groups = defaultdict(list)
        for r, q in zip(rows, questions):
            if q["facts"]:
                groups[q["answer_type"]].append(r["top3"])
        by_type[variant] = {k: pct(v) for k, v in groups.items()}
    lines.append("\nFacts in top 3 passages, by answer type: " + "; ".join(
        f"{v}: " + ", ".join(f"{k} {s}" for k, s in sorted(t.items())) for v, t in by_type.items()))
    return "\n".join(lines)


def vidore_report(result: dict) -> str:
    """nDCG@5 as ViDoRe scores it (one right document per question: 1 at rank 1, 1/log2(k+1)
    at rank k up to 5, else 0), with the plainer measures beside it."""
    lines = ["| Set-up | nDCG@5 | right document 1st | in top 5 | MRR@10 |",
             "|---|---|---|---|---|"]
    short = []
    for variant, data in result["variants"].items():
        ranks = [r["rank"] for r in data["rows"]]
        ndcg = sum(1 / math.log2(k + 1) for k in ranks if k and k <= 5) / len(ranks)
        mrr = sum(1 / k for k in ranks if k and k <= 10) / len(ranks)
        lines.append(f"| {variant} | {round(100 * ndcg, 1)} | {pct(k == 1 for k in ranks)} | "
                     f"{pct(bool(k) and k <= 5 for k in ranks)} | {round(100 * mrr, 1)} |")
        short.append(f"{variant} {sum(1 for r in data['rows'] if not r['rank'] and r['found'] < 5)}")
    lines.append(f"\nQuestions whose document was not found and whose {DOC_DEPTH} chunks held fewer "
                 "than 5 documents (scored 0, though it might have ranked 4th or 5th): "
                 + "; ".join(short))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="score only the first N questions")
    parser.add_argument("--vidore", action="store_true",
                        help="rank documents and score nDCG@5 on ViDoRe's question list instead")
    args = parser.parse_args()
    settings = load_settings(HERE.parent / "config.yaml", qdrant_url=None, qdrant_api_key=None)
    sent = gpu_modal._settings_out(settings) | settings.model_dump(mode="json", include={
        "embedding_model", "embedding_query_prompt", "embedding_batch_size", "hybrid_search",
        "reranker_model", "rerank_candidates", "exact_match", "passage_context_chars",
        "table_chunk_tokens", "chunk_size", "chunk_overlap", "model_revisions",
        "video_context_seconds"})
    questions, names = load_questions()
    if args.vidore:
        questions = load_vidore()
        unknown = {q["doc"] for q in questions} - set(names)
        if unknown:
            raise SystemExit(f"ViDoRe names {len(unknown)} file(s) not in TAT-DQA's test split, "
                             f"e.g. {sorted(unknown)[:3]}")
    if args.limit:
        questions = questions[:args.limit]
    log(f"TAT-DQA test{' (ViDoRe question list)' if args.vidore else ''}: {len(names)} documents, "
        f"{len(questions)} question(s) to score")
    records = extractions(names, sent)
    failed = [r["name"] for r in records if r["error"]]
    log(f"extracted: {len(records) - len(failed)} ok, {len(failed)} failed {failed[:3]}")
    records_file = upload_records(records)
    log("indexing and searching on Modal ...")
    size = -(-len(questions) // PARTS)
    batches = [questions[i:i + size] for i in range(0, len(questions), size)]
    method = "rank_documents" if args.vidore else "search"
    run = hashlib.sha256(json.dumps([method, batches, sent, records_file]).encode()).hexdigest()
    parts = run_parts(method, batches, records_file, sent,
                      HERE / "cache" / "tatdqa-parts" / f"{method}-{run[:16]}")
    result = parts[0]
    for part in parts[1:]:  # parts are in the batches' order: rows line up with the questions
        for variant, data in part["variants"].items():
            result["variants"][variant]["rows"] += data["rows"]
    table = vidore_report(result) if args.vidore else report(questions, result)
    print(f"\nTAT-DQA test ({len(questions)} questions, {len(names)} documents, "
          f"{result['chunks']} chunks; indexed in {result['index_seconds']} s)\n\n{table}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = "tatdqa-vidore" if args.vidore else "tatdqa"
    out = HERE / "results" / f"{name}-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run": stamp, "questions": len(questions), "documents": len(names),
                               "settings": sent, "failed": failed, "result": result,
                               "question_list": questions if args.vidore else None},
                              ensure_ascii=False), encoding="utf-8")
    title = "TAT-DQA, scored like ViDoRe (nDCG@5)" if args.vidore else "TAT-DQA results"
    (out.with_suffix(".md")).write_text(f"# {title} {stamp}\n\n{table}\n", encoding="utf-8")
    log(f"saved {out} (+ .md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
