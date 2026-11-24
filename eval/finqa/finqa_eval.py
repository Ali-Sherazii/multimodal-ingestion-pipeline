"""FinQA retrieval benchmark, scored like Table 11 of FinAgent-RAG (Shu et al., arXiv
2605.05409v1; the paper has since been withdrawn by its authors).

Retrieval only, no LLM: does the ranked list of small units (before any passage expansion)
hold the gold evidence of each FinQA test question? The corpus is every unique page of train +
dev + test (2,789), so each question searches all of them. Two unit variants (finqa_units.py),
each its own Qdrant collection in local-folder mode: "paper" (512-token passages, 64 overlap)
and "ours" (the pipeline's chunker); table rows are units in both. Three set-ups, all through
the pipeline's own code (Pipeline.search: store.py, retrieve.py's exact terms, rerank.py):

  dense                       bge-m3 dense vectors only, cosine (closest to the paper's Naive RAG)
  full + MiniLM               dense + sparse + exact terms, merged by RRF, MiniLM reranks the top 30
  full + bge-reranker-v2-m3   the same, bge-reranker-v2-m3 reranks the top 30 (the pipeline's
                              default since this benchmark)

Scores: any-hit R@k (a gold item in the top k), item recall R@k (the share of the question's
gold items in the top k: the paper's definition), MRR@10 (any-hit), for k = 5, 10, 20.

Where things run (nothing touches the pipeline's own database):
  Modal GPUs     download FinQA, build the units (checked identical to this machine's), embed
                 them (saved in parts on the Modal volume "ingest-finqa", so a stopped build
                 resumes), build each collection there; then search, 100 questions per call
  this machine   builds the units for scoring, caches every batch of rankings and reranker
                 scores in eval/cache/finqa/ (a re-run only asks for what is missing), scores

    python eval/finqa/finqa_eval.py --limit 50     # the first 50 test questions
    python eval/finqa/finqa_eval.py                # all 1,147
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import modal

from finqa_units import (average, build_units, coverage, fingerprint, load_pages, score_question)
from ingest import gpu_modal
from ingest.config import load_settings

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
DATA = HERE / "data"
CACHE = EVAL / "cache" / "finqa"
RAW = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
FILES = ("train.json", "dev.json", "test.json")
COLLECTIONS = {"paper": "finqa_paper_units", "ours": "finqa_our_units"}
RERANKERS = {"MiniLM": None, "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3"}  # None = config's
SETUPS = ["dense", "full + MiniLM", "full + bge-reranker-v2-m3"]
PART, BATCH, DEPTH, POOL = 4000, 100, 20, 30
PAPER = {  # FinAgent-RAG v1, Table 11: Recall@k = "the fraction of gold evidence passages retrieved"
    "Naive RAG (bge-base-en-v1.5 dense, FAISS)": (62.3, 74.8, 83.5),
    "Advanced RAG (dense + BM25 + query expansion)": (68.1, 79.4, 87.2),
    "FinAgent-RAG iteration 1 (fine-tuned retriever)": (69.7, 80.6, 88.1),
    "FinAgent-RAG iteration 3 (LLM query refinement; not comparable)": (82.3, 90.1, 94.3),
}

volume = modal.Volume.from_name("ingest-finqa", create_if_missing=True)
image = (gpu_modal.base_image
         .pip_install("sentence-transformers==6.1.0", "qdrant-client==1.19.1")
         .add_local_python_source("ingest")
         .add_local_file(str(HERE / "finqa_units.py"), "/root/finqa_units.py"))
app = modal.App("ingest-finqa", image=image)


def _settings(values: dict, folder: str, collection: str):
    """The pipeline's settings for a container: GPU, its own throwaway folders."""
    from ingest.config import Settings
    tmp = Path(tempfile.gettempdir())
    return Settings(_env_file=None, **{**values, "qdrant_url": None, "qdrant_api_key": None,
                                       "qdrant_path": folder, "collection": collection,
                                       "device": "cuda", "gpu_backend": "local",
                                       "isolate_extraction": False,
                                       "extracts_dir": str(tmp / "extracted"),
                                       "reports_dir": str(tmp / "reports"),
                                       "logs_dir": str(tmp / "logs"),
                                       "access_file": str(tmp / "none.yaml")})


def _data(sha: dict[str, str]) -> Path:
    """FinQA on the volume, downloaded once, checked against this machine's copy."""
    import urllib.request
    folder = Path("/finqa/data")
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = folder / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha[name]:
            path.write_bytes(urllib.request.urlopen(RAW + name, timeout=300).read())
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha[name]:
                raise RuntimeError(f"{name} on GitHub differs from this machine's copy")
    volume.commit()
    return folder


# ---- runs on the GPU ----

@app.cls(gpu="L4", cpu=4, memory=32768, volumes={"/models": gpu_modal.models_volume,
                                                 "/finqa": volume},
         timeout=3600, scaledown_window=60, max_containers=8)
class Bench:
    @modal.method()
    def build(self, variant: str, values: dict, expect: dict) -> dict:
        """Units, their embeddings (in parts on the volume: a stopped build resumes), then the
        variant's collection, built in local-folder mode and copied to the volume."""
        import numpy as np
        from qdrant_client import QdrantClient
        from ingest.embed import SentenceTransformerEmbedder
        from ingest.models import Chunk, FileType, Modality
        from ingest.rerank import token_counter
        from ingest.store import VectorStore
        root = Path(f"/finqa/{variant}")
        done = root / "index.json"
        if done.exists() and json.loads(done.read_text())["fingerprint"] == expect["fingerprint"]:
            return json.loads(done.read_text())
        started = time.perf_counter()
        s = _settings(values, "/tmp/unused", COLLECTIONS[variant])
        pages, _ = load_pages(_data(expect["sha"]))
        count = token_counter(s.embedding_model, s.model_revisions.get(s.embedding_model))
        units = build_units(pages, variant, count, s.chunk_size, s.chunk_overlap)
        if fingerprint(units) != expect["fingerprint"]:
            raise RuntimeError("the units built here differ from this machine's")
        parts = root / f"emb-{expect['fingerprint']}"
        parts.mkdir(parents=True, exist_ok=True)
        embedder = None
        for p in range(0, len(units), PART):
            path = parts / f"part-{p // PART:03d}.json"
            if path.exists():
                continue
            embedder = embedder or SentenceTransformerEmbedder(
                s.embedding_model, "cuda", 64, s.embedding_query_prompt,
                revision=s.model_revisions.get(s.embedding_model), sparse=True)
            dense, sparse = embedder.embed_documents([u.text for u in units[p:p + PART]])
            np.save(parts / f"part-{p // PART:03d}.npy", np.asarray(dense, dtype=np.float16))
            path.write_text(json.dumps([{str(k): v for k, v in sp.items()} for sp in sparse]))
            volume.commit()
        embedded = round(time.perf_counter() - started, 1)
        folder = Path(f"/tmp/qdrant-{variant}")
        shutil.rmtree(folder, ignore_errors=True)
        client = QdrantClient(path=str(folder))
        store = VectorStore(client, COLLECTIONS[variant])
        store.ensure_collection(1024, s.embedding_model, sparse=True)
        by_page: dict[str, list] = {}
        for p in range(0, len(units), PART):
            dense = np.load(parts / f"part-{p // PART:03d}.npy").astype(np.float32)
            sparse = json.loads((parts / f"part-{p // PART:03d}.json").read_text())
            for u, d, sp in zip(units[p:p + PART], dense, sparse, strict=True):
                by_page.setdefault(u.page, []).append(Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, u.uid)), document_id=u.page,
                    source_path=u.page, file_type=FileType.PDF, content=u.text,
                    modality=Modality.TABLE if u.kind == "table" else Modality.TEXT,
                    ingested_at="finqa", chunk_index=u.index, access=["everyone"],
                    embedding=d.tolist(), sparse={int(k): v for k, v in sp.items()}))
        for chunks in by_page.values():
            store.write_document(chunks)
        n = store.count()
        client.close()
        target = root / "qdrant"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(folder, target)
        info = {"fingerprint": expect["fingerprint"], "units": len(units), "stored": n,
                "collection": COLLECTIONS[variant], "embed_seconds": embedded,
                "total_seconds": round(time.perf_counter() - started, 1)}
        done.write_text(json.dumps(info))
        volume.commit()
        return info

    @modal.method()
    def search(self, questions: list[dict], variant: str, values: dict) -> list[dict]:
        """For each question: the dense top 20, and the full stack's 30 candidates with each
        reranker's scores, all through Pipeline.search and the pipeline's Reranker."""
        from qdrant_client import QdrantClient
        from ingest.pipeline import Pipeline
        from ingest.rerank import Reranker
        from ingest.store import VectorStore
        folder = Path(f"/tmp/qdrant-{variant}")
        if not folder.exists():  # a private copy: local-folder mode allows one process per folder
            volume.reload()
            shutil.copytree(Path(f"/finqa/{variant}/qdrant"), folder)
        s = _settings(values, str(folder), COLLECTIONS[variant])
        store = VectorStore(QdrantClient(path=str(folder)), COLLECTIONS[variant])
        pipeline = Pipeline(s, store=store)
        rerankers = {name: (Reranker(model, "cuda") if model else pipeline.reranker)
                     for name, model in RERANKERS.items()}
        out = []
        for q in questions:
            unit = lambda h: [h.payload["source_path"], h.payload["chunk_index"]]
            dense = pipeline.search(q["question"], limit=DEPTH, hybrid=False, rerank=False,
                                    exact=False)
            hits = pipeline.search(q["question"], limit=POOL, hybrid=True, rerank=False,
                                   exact=True)
            row = {"id": q["id"], "dense": [unit(h) for h in dense],
                   "candidates": [unit(h) for h in hits],
                   "fused": [round(float(h.score), 6) for h in hits], "scores": {}}
            ids = [str(h.id) for h in hits]
            for name, reranker in rerankers.items():
                scored = {str(h.id): float(h.score) for h in reranker.rerank(q["question"], list(hits))}
                row["scores"][name] = [round(scored[i], 5) for i in ids]
            out.append(row)
        store.close()
        gpu_modal.models_volume.commit()
        return out


# ---- runs on this machine ----

def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def ranked(row: dict, setup: str) -> list[list]:
    """The top 20 units of a set-up, as [page, index] pairs."""
    if setup == "dense":
        return row["dense"][:DEPTH]
    name = setup.split(" + ", 1)[1]
    order = sorted(range(len(row["candidates"])), key=lambda i: -row["scores"][name][i])
    return [row["candidates"][i] for i in order][:DEPTH]


ATTEMPTS, WAIT = 5, 30


def run_remote(variants: list[str], values: dict, prints: dict, sha: dict, test: list) -> None:
    """Build what is not built, then search every batch not yet saved, saving each as it comes."""
    with modal.enable_output(), app.run():
        bench = Bench()
        builds = {v: bench.build.spawn(v, values, {"fingerprint": prints[v], "sha": sha})
                  for v in variants}
        for v, call in builds.items():
            log(f"{v}: index ready {call.get()}")
        for v in variants:
            folder = CACHE / f"{v}-{prints[v]}"
            folder.mkdir(parents=True, exist_ok=True)
            batches = [test[i:i + BATCH] for i in range(0, len(test), BATCH)]
            todo = [(i, b) for i, b in enumerate(batches)
                    if not (folder / f"batch-{i:03d}-{len(b)}.json").exists()]
            log(f"{v}: {len(batches) - len(todo)} of {len(batches)} batch(es) cached, "
                f"searching {len(todo)}")
            # read the results to the end (stopping early leaves Modal's stream half closed)
            for k, rows in enumerate(bench.search.map([b for _, b in todo],
                                                      kwargs={"variant": v, "values": values})):
                i, b = todo[k]
                (folder / f"batch-{i:03d}-{len(b)}.json").write_text(json.dumps(rows))
                log(f"  {v}: batch {i + 1}/{len(batches)} saved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N test questions")
    parser.add_argument("--variants", nargs="+", choices=list(COLLECTIONS),
                        default=list(COLLECTIONS))
    parser.add_argument("--score-only", action="store_true",
                        help="score the cached rankings again, without contacting Modal")
    args = parser.parse_args()

    settings = load_settings(EVAL.parent / "config.yaml", qdrant_url=None, qdrant_api_key=None)
    values = settings.model_dump(mode="json", include={
        "embedding_model", "embedding_query_prompt", "embedding_batch_size", "hybrid_search",
        "reranker_model", "rerank_candidates", "exact_match", "chunk_size", "chunk_overlap",
        "model_revisions"})
    sha = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in FILES}
    pages, questions = load_pages(DATA)
    test = questions["test"][:args.limit] if args.limit else questions["test"]
    from ingest.rerank import token_counter
    count = token_counter(settings.embedding_model,
                          settings.model_revisions.get(settings.embedding_model))
    units = {v: build_units(pages, v, count, settings.chunk_size, settings.chunk_overlap)
             for v in args.variants}
    prints = {v: fingerprint(u) for v, u in units.items()}
    log(f"FinQA: {len(pages)} pages pooled; {len(test)} test question(s); units: "
        + ", ".join(f"{v} {len(u)} ({prints[v]})" for v, u in units.items()))

    # The internet here drops in bursts: a lost connection ends the attempt, and the next one
    # continues from the batches already saved
    for attempt in range(0 if args.score_only else 1, ATTEMPTS + 1):
        if attempt == 0:
            break
        try:
            run_remote(args.variants, values, prints, sha, test)
            break
        except (modal.exception.ConnectionError, ConnectionError, OSError) as exc:
            if attempt == ATTEMPTS:
                raise
            log(f"connection lost ({type(exc).__name__}); trying again in {WAIT} s "
                f"(attempt {attempt + 1} of {ATTEMPTS})")
            time.sleep(WAIT)

    # ---- scoring ----
    results, per_q, examples = {}, {}, []
    for v in args.variants:
        items = {(u.page, u.index): u for u in units[v]}
        folder = CACHE / f"{v}-{prints[v]}"
        rows = {}
        for i in range(0, len(test), BATCH):
            b = test[i:i + BATCH]
            for row in json.loads((folder / f"batch-{i // BATCH:03d}-{len(b)}.json").read_text()):
                rows[row["id"]] = row
        cov = coverage(units[v], test)
        for setup in SETUPS:
            scores, page5, page20 = [], [], []
            for q in test:
                top = [items[(p, k)] for p, k in ranked(rows[q["id"]], setup)]
                scores.append(score_question([u.items if u.page == q["page"] else [] for u in top],
                                             q["gold"]))
                per_q[(v, setup, q["id"])] = (scores[-1], top)
                page5.append(any(u.page == q["page"] for u in top[:5]))
                page20.append(any(u.page == q["page"] for u in top[:20]))
            results[f"{setup} | {v}"] = average(scores)
            results[f"_pages | {setup} | {v}"] = {
                "right page in top 5": round(100 * sum(page5) / len(test), 1),
                "right page in top 20": round(100 * sum(page20) / len(test), 1)}
        results[f"_candidates | {v}"] = round(100 * sum(
            any(p == q["page"] for p, _ in rows[q["id"]]["candidates"]) for q in test) / len(test), 1)
        results[f"_coverage | {v}"] = {"gold items held by no unit": sum(map(len, cov["missing"].values())),
                                       "questions affected": cov["missing"],
                                       "malformed keys": cov["malformed"]}
    # misses of the default set-up, on our units: what came back instead
    texts = {v: {(u.page, u.index): u for u in units[v]} for v in args.variants}
    v0 = "ours" if "ours" in args.variants else args.variants[0]
    for q in test:
        s, top = per_q[(v0, "full + MiniLM", q["id"])]
        if s["any@5"] == 0.0 and len(examples) < 5:
            page = pages[q["page"]]
            gold_text = []
            for g in q["gold"]:
                kind, n = g.split("_")
                n = int(n)
                if kind == "text" and 0 <= n < len(page["sentences"]):
                    gold_text.append(page["sentences"][n][:200])
                elif kind == "table" and 0 <= n < len(page["table"]):
                    gold_text.append(" | ".join(page["table"][n])[:200])
            examples.append({"id": q["id"], "question": q["question"], "gold": gold_text,
                             "retrieved": [{"page": u.page, "text": " ".join(u.text.split())[:200]}
                                           for u in top[:3]]})

    write_report(results, examples, test, settings, prints, args.limit)
    return 0


def write_report(results: dict, examples: list, test: list, settings, prints: dict, limit) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = "finqa_retrieval" + (f"-first{limit}" if limit else "")
    rows = [k for k in results if not k.startswith("_")]
    metrics = ("any@5", "any@10", "any@20", "items@5", "items@10", "items@20", "mrr@10")
    unit_name = {"paper": "paper-style", "ours": "ours"}   # in tables
    prose = {"paper": "paper-style", "ours": "our"}         # in sentences: "our units"
    head = ("| Set-up | Units | any-hit R@5 | R@10 | R@20 | item recall R@5 | R@10 | R@20 | MRR@10 |\n"
            "|---|---|---|---|---|---|---|---|---|")
    lines = [head] + [f"| {k.split(' | ')[0]} | {unit_name[k.split(' | ')[1]]} | "
                      + " | ".join(str(results[k][m]) for m in metrics) + " |" for k in rows]
    get = lambda setup, v: results.get(f"{setup} | {v}")
    triple = lambda x: f"{x['items@5']} / {x['items@10']} / {x['items@20']}" if x else "-"
    best = max(rows, key=lambda k: results[k]["items@5"])
    naive = PAPER["Naive RAG (bge-base-en-v1.5 dense, FAISS)"]
    iter1 = PAPER["FinAgent-RAG iteration 1 (fine-tuned retriever)"]
    compare = ["| Comparison point | Paper, item recall R@5 / R@10 / R@20 | Ours, item recall R@5 / R@10 / R@20 |",
               "|---|---|---|",
               f"| Naive RAG vs our dense only, paper-style units | {' / '.join(map(str, naive))} | {triple(get('dense', 'paper'))} |",
               f"| Naive RAG vs our dense only, our units | {' / '.join(map(str, naive))} | {triple(get('dense', 'ours'))} |",
               f"| Iteration 1 vs our best ({best.split(' | ')[0]}, {prose[best.split(' | ')[1]]} units) | "
               f"{' / '.join(map(str, iter1))} | {triple(results[best])} |"]
    readout = [f"- **Best set-up:** {best.split(' | ')[0]} on {prose[best.split(' | ')[1]]} units, item "
               f"recall {triple(results[best])}: {round(naive[0] - results[best]['items@5'])} to "
               f"{round(naive[2] - results[best]['items@20'])} points below the paper's claimed Naive RAG."]
    for v in prints:
        a, b = get("full + MiniLM", v), get("full + bge-reranker-v2-m3", v)
        if a and b:
            readout.append(f"- **Reranker ({prose[v]} units):** bge-reranker-v2-m3 instead of MiniLM moves "
                           f"item recall R@5 from {a['items@5']} to {b['items@5']} and MRR@10 from "
                           f"{a['mrr@10']} to {b['mrr@10']}.")
    if get("dense", "paper") and get("dense", "ours"):
        readout.append(f"- **Unit size:** with dense search alone, our smaller units find more "
                       f"({get('dense', 'ours')['items@5']} vs {get('dense', 'paper')['items@5']} item recall R@5); "
                       "with the full stack they differ by at most "
                       f"{max(abs(get(x, 'ours')['items@5'] - get(x, 'paper')['items@5']) for x in SETUPS[1:]):.1f} "
                       "points.")
    v0 = "ours" if "ours" in prints else next(iter(prints))
    page = results.get(f"_pages | full + MiniLM | {v0}")
    if page:
        readout.append(f"- **Where it fails:** with the default set-up the right page is in the top 20 for "
                       f"{page['right page in top 20']}% of questions and among the 30 reranked candidates for "
                       f"{results[f'_candidates | {v0}']}%; the gold rows and sentences are then out-ranked "
                       "by similar rows from other companies and years (many questions do not name the "
                       "company, and a table row such as \"2018 | 301\" carries little context on its own).")
    readout.append("- **Exact-term matching** barely matters here: FinQA's text has no comma-formatted "
                   "figures or ids to match as written.")
    pages = ["| Set-up | Units | right page in top 5 (%) | in top 20 (%) |", "|---|---|---|---|"] + [
        f"| {k.split(' | ')[1]} | {unit_name[k.split(' | ')[2]]} | {v['right page in top 5']} | "
        f"{v['right page in top 20']} |" for k, v in results.items() if k.startswith("_pages")]
    paper = ["| Paper set-up | R@5 | R@10 | R@20 |", "|---|---|---|---|"] + [
        f"| {k} | {a} | {b} | {c} |" for k, (a, b, c) in PAPER.items()]
    lost = []
    for k, v in results.items():
        if k.startswith("_coverage"):
            ids = "; ".join(f"{q}: {', '.join(g)}" for q, g in v["questions affected"].items())
            lost.append(f"{unit_name[k.split(' | ')[1]]} {v['gold items held by no unit']}"
                        + (f" ({ids}: a sentence longer than two chunks)" if ids else ""))
    md = [f"# FinQA retrieval ({len(test)} test questions{', the first ' + str(limit) if limit else ''})",
          "",
          f"Run {stamp}. Retrieval only, no LLM. Corpus: all 2,789 unique FinQA pages (train + dev + "
          "test pooled), so every question searches every page. Units are scored as retrieved, before "
          "any passage expansion. Unit sets: " + ", ".join(f"{unit_name[v]} {p}" for v, p in prints.items())
          + ". bge-m3 embeddings; full stack = dense + sparse + exact terms, merged by RRF, top 30 reranked.",
          "", "## Results", "", "\n".join(lines), "",
          "Item recall R@k: the share of a question's gold items in the top k (the paper's definition). "
          "Any-hit R@k: at least one gold item in the top k. MRR@10: 1 / rank of the first gold item.", "",
          "Gold items held by no unit, counted as misses: " + "; ".join(lost) + ". No question was "
          "dropped. Test questions were used only for scoring; nothing was tuned on them.", "",
          "## Read-out", "", "\n".join(readout), "",
          "## Next to FinAgent-RAG (Table 11, arXiv 2605.05409v1)", "", "\n".join(compare), "",
          "\n".join(paper), "",
          "Caveats:", "",
          "- The paper was withdrawn by its authors \"due to significant methodological errors in "
          "the experimental design\"; these are claimed numbers from version 1, and its code is not released.",
          "- Its corpus is \"52,847 passages from 8,281 earnings report pages\"; FinQA has 8,281 "
          "questions but only 2,789 unique pages, so pages may have been indexed once per question "
          "(duplicates). Which splits it pooled is not stated. Ours: 2,789 unique pages, all splits.",
          "- Its retriever is bge-base-en-v1.5 with FAISS (Naive RAG) or a fine-tuned retriever "
          "(iteration 1); ours is bge-m3, not fine-tuned on FinQA. Iterations 2-3 refine the "
          "question with an LLM and are not comparable to a single-pass retriever.",
          "- FinQA's text is lowercased and pre-tokenised (\"$ 5735\", no thousands separators), "
          "unlike the PDFs this pipeline reads.", "",
          "## How often the right page is retrieved", "", "\n".join(pages), "",
          "The right page is among the 30 candidates handed to the reranker for "
          + ", ".join(f"{results[f'_candidates | {v}']}% ({prose[v]} units)" for v in prints)
          + " of questions.", "",
          "## Five misses of the default set-up (full + MiniLM, our units): what came back instead", ""]
    for e in examples:
        md += [f"**{e['id']}**: {e['question']}", "",
               "- Gold: " + " / ".join(f"`{g}`" for g in e["gold"])] + [
               f"- Retrieved #{i}: ({r['page']}) {r['text']}" for i, r in enumerate(e["retrieved"], 1)] + [""]
    out = EVAL / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run": stamp, "questions": len(test), "units": prints,
                               "settings": settings.model_dump(mode="json", include={
                                   "embedding_model", "reranker_model", "rerank_candidates",
                                   "exact_match", "chunk_size", "chunk_overlap"}),
                               "setups": SETUPS, "results": results, "paper": PAPER,
                               "misses": examples}, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(lines))
    log(f"saved {out} (+ .md)")


if __name__ == "__main__":
    sys.exit(main())
