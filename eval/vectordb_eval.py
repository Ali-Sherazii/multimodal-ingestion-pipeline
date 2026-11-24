"""Vector database tuning (the supervisor's step 2b), measured on Modal GPUs.

Which Qdrant settings put the right document in front of the most questions, for the least
memory and time? Every set-up runs on its own fresh Qdrant server (v1.19.2, the software
behind Qdrant Cloud) inside one Modal container, one after another, on the same chunks and
vectors:

  corpus     TAT-DQA's 277 test reports (the pages holding the answers) plus FinQA's 2,789
             report pages as decoys (similar financial pages): chunked and embedded exactly
             as the pipeline does (bge-m3 dense vectors + keyword weights)
  questions  ViDoRe's 1,646 TAT-DQA questions, each with its right document as ViDoRe has it

Measured per set-up:
  nDCG@5         the documents' ranking after the reranker (what users get) and before it (the
                 database's own ranking): ViDoRe's measure, as in tatdqa_eval.py --vidore
  in candidates  how often the right document is among the 30 chunks the reranker sees
  search ms      database time per question (median, 95th percentile); query vectors are
                 made beforehand, so this is the database alone
  memory MB      the Qdrant server's resident memory, and the vector bytes kept in RAM
  build s        upload plus building the index

The job runs detached on Modal: a dropped connection here doesn't stop it. Each finished
set-up is saved on the "ingest-vectordb" volume, and a re-run skips it.

    python eval/vectordb_eval.py --limit 100     # quick check: 100 questions
    python eval/vectordb_eval.py                 # all 1,646 questions, every set-up
    python eval/vectordb_eval.py --collect       # write the report from what is saved
"""
import argparse
import gzip
import hashlib
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

from ingest import gpu_modal
from ingest.config import load_settings

HERE = Path(__file__).resolve().parent
QDRANT = "v1.19.2"
CORPUS_VERSION = 1  # bump when the corpus is built differently
CANDIDATES = 30     # the reranker's candidates (rerank_candidates)

image = (gpu_modal.base_image
         .apt_install("curl")
         # the static (musl) build: the glibc one needs a newer glibc than the image has
         .run_commands(f"curl -sL https://github.com/qdrant/qdrant/releases/download/{QDRANT}/"
                       "qdrant-x86_64-unknown-linux-musl.tar.gz | tar xz -C /usr/local/bin",
                       "qdrant --version")
         .pip_install("sentence-transformers==6.1.0", "qdrant-client==1.19.1",
                      "fastembed==0.9.1", "psutil==7.2.2")
         .add_local_file(str(HERE / "finqa" / "finqa_units.py"), "/root/finqa_units.py")
         .add_local_python_source("ingest"))
app = modal.App("ingest-vectordb", image=image)
volume = modal.Volume.from_name("ingest-vectordb", create_if_missing=True)
finqa_volume = modal.Volume.from_name("ingest-finqa", create_if_missing=True)

# name: settings changed from BASE (the pipeline before this test). "exact" (no graph) is for
# comparison only; "keywords": bge-m3's word weights, BM25's, or both (the pipeline's bm25)
BASE = {"hnsw_m": 16, "hnsw_ef_construct": 100, "search_ef": None, "quantization": "none",
        "quantization_oversampling": 2.0, "vectors_on_disk": False, "fusion": "rrf"}
SETUPS = {
    "current: float32 vectors in RAM, graph m16": {},
    "exact search (no graph: the upper bound)": {"exact": True},
    "bigger graph: m32, build 256, search 256": {"hnsw_m": 32, "hnsw_ef_construct": 256,
                                                  "search_ef": 256},
    "int8 compression (4x smaller)": {"quantization": "int8"},
    "binary compression (32x smaller)": {"quantization": "binary",
                                         "quantization_oversampling": 3.0},
    "int8 in RAM, full vectors on disk": {"quantization": "int8", "vectors_on_disk": True},
    "DBSF instead of RRF to merge the rankings": {"fusion": "dbsf"},
    "BM25 instead of bge-m3 keyword weights": {"keywords": "bm25"},
    "bge-m3 keyword weights + BM25": {"keywords": "both"},
    "chosen: bge-m3 + BM25, int8 in RAM, vectors on disk": {
        "keywords": "both", "quantization": "int8", "vectors_on_disk": True},
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


def ndcg5(rank: int | None) -> float:
    """One right document: 1 at rank 1, 1/log2(rank+1) up to rank 5, else 0 (ViDoRe's)."""
    return 1 / math.log2(rank + 1) if rank and rank <= 5 else 0.0


def doc_rank(hits: list, doc: str) -> int | None:
    docs = list(dict.fromkeys(h.payload["source_path"] for h in hits))
    return docs.index(doc) + 1 if doc in docs else None


# ---- runs on the GPU ----

def _markdown(rows: list[list]) -> str:
    cells = [[str(c).replace("|", "/").strip() for c in r] for r in rows if r]
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * width]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells[1:]])


def _embedder(s):
    from ingest.embed import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(
        s.embedding_model, "cuda", s.embedding_batch_size, s.embedding_query_prompt,
        revision=s.model_revisions.get(s.embedding_model), sparse=True)


def _corpus(folder: Path, records_file: str, s, embedder) -> dict:
    """Chunks and their vectors, built once and kept on the volume (the slow part)."""
    saved = folder / "corpus.pkl.gz"
    if saved.exists():
        return pickle.loads(gzip.decompress(saved.read_bytes()))
    import numpy as np
    from fastembed import SparseTextEmbedding
    from finqa_units import load_pages

    from ingest.chunk import chunk_segments
    from ingest.embed import embed_chunks, text_for_embedding
    from ingest.models import DetectedFile, FileType, Modality, Segment
    from ingest.normalise import normalise
    from ingest.rerank import token_counter

    started = time.perf_counter()
    gpu_modal.models_volume.reload()
    records = json.loads(gzip.decompress(Path("/models", records_file).read_bytes()))
    docs = [(r["name"], gpu_modal._segments_in(r["segments"]))
            for r in records if not r["error"] and r["segments"]]
    pages, _ = load_pages(Path("/finqa/data"))
    for page, p in sorted(pages.items()):
        segments = [Segment(" ".join(p["sentences"]), page_number=1)]
        if p["table"]:
            segments.append(Segment(_markdown(p["table"]), Modality.TABLE, page_number=1))
        docs.append((f"finqa/{page}", segments))
    count = token_counter(s.reranker_model, s.model_revisions.get(s.reranker_model))
    chunks = []
    for name, segments in docs:
        detected = DetectedFile(Path(name), FileType.PDF, name)
        for c in chunk_segments(normalise(segments), detected, name, "bench",
                                s.chunk_size, s.chunk_overlap,
                                table_size=s.table_chunk_tokens or None, count=count):
            c.access = ["everyone"]
            chunks.append(c)
    print(f"{len(docs)} documents, {len(chunks)} chunks ({time.perf_counter() - started:.0f} s)",
          flush=True)
    embed_chunks(chunks, embedder)
    texts = [text_for_embedding(c) for c in chunks]
    bm25 = SparseTextEmbedding("Qdrant/bm25").embed(texts, batch_size=256)
    corpus = {
        "ids": [c.chunk_id for c in chunks],
        "payloads": [c.payload() for c in chunks],
        "dense": np.array([c.embedding for c in chunks], dtype=np.float32),
        "sparse": [c.sparse for c in chunks],
        "bm25": [(e.indices.tolist(), e.values.tolist()) for e in bm25],
        "model": embedder.model_name,
    }
    folder.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(gzip.compress(pickle.dumps(corpus), compresslevel=3))
    volume.commit()
    print(f"embedded ({time.perf_counter() - started:.0f} s)", flush=True)
    return corpus


def _queries(questions: list[dict], embedder) -> dict:
    """The questions' vectors (dense, bge-m3 keywords, BM25) and exact terms."""
    import numpy as np
    from fastembed import SparseTextEmbedding

    from ingest.retrieve import exact_terms
    asked = [embedder.embed_query(q["question"]) for q in questions]
    words = SparseTextEmbedding("Qdrant/bm25").query_embed([q["question"] for q in questions])
    return {
        "queries": [{**q, "exact": exact_terms(q["question"])} for q in questions],
        "query_dense": np.array([a[0] for a in asked], dtype=np.float32),
        "query_sparse": [a[1] for a in asked],
        "query_bm25": [(e.indices.tolist(), e.values.tolist()) for e in words],
    }


class _Qdrant:
    """A fresh Qdrant server with its own empty storage, for one set-up."""

    def __enter__(self):
        import os

        import requests
        self.folder = Path("/tmp/qdrant")
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir()
        env = {**os.environ, "QDRANT__STORAGE__STORAGE_PATH": str(self.folder / "storage"),
               "QDRANT__STORAGE__SNAPSHOTS_PATH": str(self.folder / "snapshots"),
               "QDRANT__TELEMETRY_DISABLED": "true", "QDRANT__LOG_LEVEL": "WARN"}
        self.log = open(self.folder / "qdrant.log", "w")
        self.process = subprocess.Popen(["qdrant"], cwd=self.folder, env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                if requests.get("http://localhost:6333/readyz", timeout=1).ok:
                    return self
            except requests.RequestException:
                pass
            if self.process.poll() is not None:
                break
            time.sleep(0.5)
        self.log.flush()
        said = (self.folder / "qdrant.log").read_text(errors="replace")[-2000:]
        raise RuntimeError(f"Qdrant did not start (exit {self.process.poll()}):\n{said}")

    def memory_mb(self) -> float:
        import psutil
        return round(psutil.Process(self.process.pid).memory_info().rss / 2**20, 1)

    def __exit__(self, *exc):
        self.process.terminate()
        self.process.wait(timeout=60)
        self.log.close()
        shutil.rmtree(self.folder, ignore_errors=True)


def _measure(name: str, changes: dict, corpus: dict, values: dict, score) -> dict:
    """Load the corpus into a fresh server with this set-up, then time and score searches."""
    from qdrant_client import QdrantClient, models

    from ingest.config import Settings
    from ingest.store import BM25, DENSE, SPARSE, VectorStore

    s = Settings(_env_file=None, **{**values, **{k: v for k, v in changes.items()
                                                  if k not in ("exact", "keywords")}})
    keywords = changes.get("keywords", "bge-m3")
    dense = corpus["dense"]
    with _Qdrant() as server:
        client = QdrantClient(url="http://localhost:6333", timeout=600)
        store = VectorStore(client, "bench", settings=s)  # the pipeline's own store
        if changes.get("exact"):
            store.tuning["params"] = models.SearchParams(exact=True)
        started = time.perf_counter()
        store.ensure_collection(dense.shape[1], corpus["model"], sparse=keywords != "bm25",
                                bm25=keywords != "bge-m3")

        def vectors(i: int) -> dict:
            v = {DENSE: dense[i].tolist()}
            if keywords != "bm25":
                v[SPARSE] = models.SparseVector(indices=list(corpus["sparse"][i]),
                                                values=list(corpus["sparse"][i].values()))
            if keywords != "bge-m3":
                v[BM25] = models.SparseVector(indices=corpus["bm25"][i][0],
                                              values=corpus["bm25"][i][1])
            return v
        client.upload_points("bench", batch_size=256, wait=True, points=(
            models.PointStruct(id=corpus["ids"][i], vector=vectors(i),
                               payload=corpus["payloads"][i]) for i in range(len(dense))))
        while client.get_collection("bench").status != models.CollectionStatus.GREEN:
            time.sleep(1)
        build = time.perf_counter() - started

        def search(i: int) -> list:
            q = corpus["queries"][i]
            vector, sparse = corpus["query_dense"][i].tolist(), corpus["query_sparse"][i]
            words = dict(zip(*corpus["query_bm25"][i])) if keywords != "bge-m3" else None
            return store.search(vector, sparse if keywords != "bm25" else None,
                                limit=CANDIDATES, exact=q["exact"], bm25=words)

        for i in range(min(20, len(corpus["queries"]))):  # warm up
            search(i)
        times, found = [], []
        for i in range(len(corpus["queries"])):
            t0 = time.perf_counter()
            found.append(search(i))
            times.append(1000 * (time.perf_counter() - t0))
        memory = server.memory_mb()
        info = client.get_collection("bench")
    reranked = score(found)
    gold = [q["doc"] for q in corpus["queries"]]
    n = len(gold)
    first = [doc_rank(h, d) for h, d in zip(found, gold)]
    final = [doc_rank(h, d) for h, d in zip(reranked, gold)]
    times.sort()
    vector_bytes = {"none": 4, "int8": 1, "binary": 1 / 8}[s.quantization] * dense.size
    if s.quantization != "none" and not s.vectors_on_disk:
        vector_bytes += 4 * dense.size  # full vectors kept in RAM for re-scoring too
    if s.quantization == "none" and s.vectors_on_disk:
        vector_bytes = 0
    return {
        "setup": name, "changes": changes, "questions": n, "chunks": len(dense),
        "ndcg5": round(100 * sum(map(ndcg5, final)) / n, 1),
        "ndcg5_database": round(100 * sum(map(ndcg5, first)) / n, 1),
        "first": round(100 * sum(r == 1 for r in final) / n, 1),
        "in_candidates": round(100 * sum(r is not None for r in first) / n, 1),
        "ms_median": round(times[n // 2], 1), "ms_p95": round(times[int(n * 0.95)], 1),
        "memory_mb": memory, "vector_ram_mb": round(vector_bytes / 2**20, 1),
        "build_s": round(build, 1), "indexed": info.indexed_vectors_count,
        "ranks": final,
    }


@app.cls(gpu=["L4", "A10G"], cpu=4, memory=32768, timeout=4 * 3600,
         volumes={"/models": gpu_modal.models_volume, "/vdb": volume, "/finqa": finqa_volume})
class Bench:
    @modal.method()
    def run(self, questions: list[dict], records_file: str, values: dict, corpus_id: str,
            run_id: str, setups: list[str]) -> list[str]:
        from ingest.config import Settings
        from ingest.rerank import Reranker, _text

        s = Settings(_env_file=None, **values)
        embedder = _embedder(s)
        corpus = _corpus(Path("/vdb") / corpus_id, records_file, s, embedder)
        corpus.update(_queries(questions, embedder))
        del embedder
        folder = Path("/vdb") / corpus_id / run_id
        reranker = Reranker(s.reranker_model, "cuda", s.model_revisions.get(s.reranker_model))
        cache: dict[tuple[int, str], float] = {}

        def score(found: list[list]) -> list[list]:
            """Each question's hits re-ordered by the reranker; scores are kept across set-ups
            (the same question and chunk always score the same)."""
            todo = {(i, h.id): (corpus["queries"][i]["question"], _text(h.payload))
                    for i, hits in enumerate(found) for h in hits if (i, h.id) not in cache}
            keys = sorted(todo, key=lambda k: len(todo[k][1]))  # similar lengths: less padding
            for start in range(0, len(keys), 512):
                batch = keys[start:start + 512]
                scores = reranker.model.predict([todo[k] for k in batch], batch_size=64,
                                                show_progress_bar=False)
                cache.update(zip(batch, map(float, scores)))
            return [sorted(hits, key=lambda h: cache[(i, h.id)], reverse=True)
                    for i, hits in enumerate(found)]

        done = []
        for name in setups:
            out = folder / "results" / f"{slug(name)}.json"
            if out.exists():
                done.append(name)
                continue
            print(f"set-up: {name}", flush=True)
            result = _measure(name, SETUPS[name], corpus, values, score)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result))
            volume.commit()
            print(f"  nDCG@5 {result['ndcg5']} (database {result['ndcg5_database']}), "
                  f"{result['ms_median']} ms, {result['memory_mb']} MB", flush=True)
            done.append(name)
        return done


# ---- runs on this machine ----

def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def saved_results(folder: str) -> dict[str, dict]:
    found = {}
    try:
        entries = volume.listdir(f"/{folder}/results")
    except Exception:  # nothing saved yet
        return found
    for entry in entries:
        data = b"".join(volume.read_file(entry.path))
        result = json.loads(data)
        found[result["setup"]] = result
    return found


def report(results: list[dict]) -> str:
    lines = ["| Set-up | nDCG@5 | before reranker | right document 1st | in candidates | "
             "search ms (median / 95%) | server memory MB | vectors in RAM MB | build s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['setup']} | {r['ndcg5']} | {r['ndcg5_database']} | {r['first']} | "
                     f"{r['in_candidates']} | {r['ms_median']} / {r['ms_p95']} | "
                     f"{r['memory_mb']} | {r['vector_ram_mb']} | {r['build_s']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    parser.add_argument("--collect", action="store_true",
                        help="only write the report from the set-ups saved on the volume")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    import tatdqa_eval as tat  # this machine only: ViDoRe's questions, cached extractions

    settings = load_settings(HERE.parent / "config.yaml", qdrant_url=None, qdrant_api_key=None)
    values = gpu_modal._settings_out(settings) | settings.model_dump(mode="json", include={
        "embedding_model", "embedding_query_prompt", "embedding_batch_size", "hybrid_search",
        "reranker_model", "rerank_candidates", "exact_match", "table_chunk_tokens",
        "chunk_size", "chunk_overlap", "model_revisions", "hnsw_m", "hnsw_ef_construct",
        "search_ef", "quantization", "quantization_oversampling", "vectors_on_disk", "fusion"})
    values.update(BASE)  # every set-up starts from the same base, whatever config.yaml says
    questions = tat.load_vidore()[:args.limit] if args.limit else tat.load_vidore()
    _, names = tat.load_questions()
    records = tat.extractions(names, values)
    records_file = tat.upload_records(records)
    # The corpus depends on the chunking and models, a run also on the questions and set-ups
    corpus_id = hashlib.sha256(json.dumps([CORPUS_VERSION, records_file, {
        k: values[k] for k in ("embedding_model", "chunk_size", "chunk_overlap",
                               "table_chunk_tokens", "reranker_model", "model_revisions")}]
                                          ).encode()).hexdigest()[:16]
    run_id = hashlib.sha256(json.dumps([questions, values]).encode()).hexdigest()[:16]
    folder = f"{corpus_id}/{run_id}"
    log(f"{len(questions)} questions; corpus {corpus_id}, run {run_id}")

    if not args.collect:
        with modal.enable_output(), app.run(detach=True):
            call = Bench().run.spawn(questions, records_file, values, corpus_id, run_id,
                                     args.setups)
            log("running on Modal (detached: it keeps going if this machine disconnects)")
            while True:
                try:
                    call.get(timeout=60)
                    break
                except (TimeoutError, modal.exception.TimeoutError):
                    log(f"  {len(saved_results(folder))} of {len(args.setups)} set-up(s) done")
                except (modal.exception.ConnectionError, ConnectionError, OSError) as exc:
                    log(f"lost the connection ({type(exc).__name__}); the job continues on "
                        "Modal: run this again with --collect later")
                    return 2

    results = saved_results(folder)
    ordered = [results[n] for n in args.setups if n in results]
    if not ordered:
        log("nothing saved yet")
        return 1
    table = report(ordered)
    print(f"\n{table}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = HERE / "results" / f"vectordb-{stamp}.json"
    out.write_text(json.dumps({"run": stamp, "folder": folder, "settings": values,
                               "questions": len(questions), "results": ordered}),
                   encoding="utf-8")
    out.with_suffix(".md").write_text(
        f"# Vector database tuning {stamp}\n\n{len(questions)} ViDoRe TAT-DQA questions; "
        f"{ordered[0]['chunks']} chunks (TAT-DQA reports + FinQA pages as decoys)\n\n{table}\n",
        encoding="utf-8")
    log(f"saved {out} (+ .md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
