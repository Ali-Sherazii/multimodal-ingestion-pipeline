"""Runs the stages for each file. One file failing never stops the others (FR-13, NFR-06).

Extraction runs in a worker process (isolation.py), so even a native crash, e.g. running
out of memory inside Docling, fails only that file. Its result is saved under the file's
content hash (extracts.py), so a re-run, a moved file or a new chunk size or embedding model
never repeats it. `ingest_file` depends only on its path and the settings, so files can be
spread over a job queue (jobs.py).
"""
import time
from datetime import datetime, timezone
from pathlib import Path

from ingest.access import AccessRules
from ingest.chunk import chunk_segments
from ingest.config import Settings
from ingest.detect import UnsupportedFile, detect
from ingest.embed import Embedder, SentenceTransformerEmbedder, embed_chunks
from ingest.extractors import get_extractor
from ingest.extracts import ExtractStore, copies_of, extraction_key
from ingest.intake import collect_files
from ingest.isolation import ExtractorCrashed, IsolatedExtractor
from ingest.logs import log, timed_stage
from ingest.models import DetectedFile, FileResult
from ingest.normalise import normalise
from ingest.pdf_checks import UnreadablePdf
from ingest.quality import find_instructions
from ingest.report import RunReport
from ingest.store import VectorStore


def source_key(path: Path) -> str:
    """How a file's location is stored: its absolute path, with forward slashes."""
    return Path(path).resolve().as_posix()


class Pipeline:
    def __init__(self, settings: Settings, embedder: Embedder | None = None,
                 store: VectorStore | None = None,
                 isolated: IsolatedExtractor | None = None, reranker=None,
                 count_tokens=None, bm25=None):
        self.settings = settings
        self._embedder = embedder
        self._bm25 = bm25
        self._reranker = reranker
        self._count_tokens = count_tokens
        self.store = store or VectorStore.from_settings(settings)
        self.extracts = ExtractStore(settings.extracts_dir)
        self.access = AccessRules.load(settings)
        # Shared by pipelines that differ only in settings (web uploads with extra options)
        self.isolated = isolated or (IsolatedExtractor(settings)
                                     if settings.isolate_extraction else None)

    def _revision(self, model: str) -> str | None:
        return self.settings.model_revisions.get(model)

    @property
    def embedder(self) -> Embedder:
        """Loaded on first use, so viewing, exporting or deleting never loads a model."""
        if self._embedder is None:
            s = self.settings
            log.info(f"loading embedding model {s.embedding_model}")
            self._embedder = SentenceTransformerEmbedder(
                s.embedding_model, s.device, s.embedding_batch_size, s.embedding_query_prompt,
                revision=self._revision(s.embedding_model), sparse=s.hybrid_search)
            if s.hybrid_search and not self._embedder.has_sparse:
                log.warning(f"{s.embedding_model} has no keyword (sparse) weights; searching "
                            "by meaning only (hybrid search needs BAAI/bge-m3)")
        return self._embedder

    @property
    def bm25(self):
        """BM25 word weights (embed.Bm25) when `bm25` is on; loaded on first use."""
        if self._bm25 is None and self.settings.bm25:
            from ingest.embed import Bm25
            self._bm25 = Bm25()
        return self._bm25 if self.settings.bm25 else None

    @property
    def reranker(self):
        if self._reranker is None and self.settings.reranker_model:
            from ingest.rerank import Reranker
            log.info(f"loading reranker {self.settings.reranker_model}")
            self._reranker = Reranker(self.settings.reranker_model, self.settings.device,
                                      self._revision(self.settings.reranker_model))
        return self._reranker

    @property
    def count_tokens(self):
        """Table chunks are measured in the tokens of the model that reads them last and
        reads the least: the reranker (512 tokens), else the embedding model."""
        if self._count_tokens is None:
            from ingest.rerank import token_counter
            model = self.settings.reranker_model or self.settings.embedding_model
            self._count_tokens = token_counter(model, self._revision(model))
        return self._count_tokens

    # ---- ingesting ----

    def run(self, inputs: list[str | Path], force: bool = False,
            re_extract: bool = False) -> RunReport:
        report = RunReport.start()
        files = collect_files(inputs)
        log.info(f"found {len(files)} file(s)")
        for i, path in enumerate(files, 1):
            log.info(f"[{i}/{len(files)}] {path.name}")
            report.add(self.ingest_file(path, force=force, re_extract=re_extract))
            # Saved after every file: if the run is killed, the results so far survive
            report.save(self.settings.reports_dir.resolve())
        report.finish()
        return report

    def ingest_file(self, path: Path, force: bool = False, re_extract: bool = False,
                    groups: list[str] | None = None, owner: str | None = None,
                    chat_id: str | None = None) -> FileResult:
        """`groups` overrides the access rules for this file (e.g. chosen at upload); `owner`:
        the user uploading it, `chat_id`: the chat it was uploaded into."""
        name = path.name
        key = source_key(path)
        result = FileResult(path=key, status="failed")
        timings = result.stage_seconds
        started = time.perf_counter()
        try:
            with timed_stage("detect", name, timings):
                detected = detect(path)
            result.file_type = detected.file_type.value
            access = groups or self.access.groups_for(key)

            if not force and (reason := self._already_stored(detected, key, access, owner,
                                                             chat_id)):
                result.status, result.reason = "skipped", reason
                return result

            segments = None if re_extract else self.extracts.load(
                detected.document_id, extraction_key(self.settings))
            if segments is not None:
                log.info("reusing the saved extraction: nothing read again", extra={"file": name})
                timings["extract"] = 0.0
            else:
                extract = get_extractor(detected.file_type)
                with timed_stage("extract", name, timings):
                    if self.isolated:
                        segments = self.isolated.run(extract, path, self.settings)
                    else:
                        segments = extract(path, self.settings)
                self.extracts.save(detected.document_id, detected.file_type, key, segments,
                                   extraction_key(self.settings), self.settings.gpu_backend)
            self._index(detected, key, segments, access, result, owner, chat_id)
        except UnsupportedFile as exc:
            result.status, result.reason = "skipped", str(exc)
        except (UnreadablePdf, ExtractorCrashed) as exc:
            result.status, result.reason = "failed", str(exc)
        except Exception as exc:
            result.status, result.reason = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            result.seconds = round(time.perf_counter() - started, 2)
            self._log_result(name, result)
        return result

    def _already_stored(self, detected: DetectedFile, key: str, access: list[str],
                        owner: str | None = None, chat_id: str | None = None) -> str | None:
        """Why this file needs no work, if it doesn't: same content stored at this path, or
        moved here (its old path is gone: only the stored path is updated), or a copy of
        a file already stored. Content is compared by hash, never by name. Another user's
        copy doesn't count: each user gets their own (from the saved extraction, not re-read)."""
        stored = {path for path, by in self.store.copies(detected.document_id) if by == owner}
        if not stored:
            return None
        if key in stored:
            if chat_id:
                self.store.set_chat(key, chat_id)
                self.extracts.update_copies(detected.document_id, key, chat_id=chat_id)
            return "already ingested, unchanged"
        if all(not Path(old).exists() for old in stored):
            self.store.move_document(detected.document_id, key, access, sorted(stored))
            self.extracts.set_source(detected.document_id, key)
            self.extracts.move_copies(detected.document_id, sorted(stored), key, access)
            return f"moved from {sorted(stored)[0]}: stored path updated, nothing re-read"
        return f"same content as {sorted(stored)[0]}, already ingested"

    def _index(self, detected: DetectedFile, key: str, segments, access: list[str],
               result: FileResult, owner: str | None = None, chat_id: str | None = None) -> None:
        """Normalise, chunk, flag, embed and store one file's extracted segments."""
        name = detected.path.name
        timings = result.stage_seconds
        warnings = list(getattr(segments, "warnings", []))
        with timed_stage("normalise", name, timings):
            segments = normalise(segments)
        if not segments:
            raise ValueError("no text could be extracted from this file")

        ingested_at = datetime.now(timezone.utc).isoformat()
        with timed_stage("chunk", name, timings):
            tables = self.settings.table_chunk_tokens
            chunks = chunk_segments(segments, detected, key, ingested_at,
                                    self.settings.chunk_size, self.settings.chunk_overlap,
                                    table_size=tables or None,
                                    count=self.count_tokens if tables else len)
        for c in chunks:
            c.access, c.owner, c.chat_id = access, owner, chat_id
        if self.settings.flag_instructions:
            flagged = [(c, m) for c in chunks if (m := find_instructions(c.content))]
            for c, _ in flagged:
                c.flags.append("instruction-like text")
            if flagged:
                warnings.append(f"{len(flagged)} chunk(s) contain instruction-like text, "
                                f"flagged for review: {flagged[0][1]!r}")
        with timed_stage("embed", name, timings):
            embed_chunks(chunks, self.embedder, self.bm25)
        with timed_stage("store", name, timings):
            self.store.ensure_collection(self.embedder.dimension, self.embedder.model_name,
                                         sparse=self.embedder.has_sparse,
                                         bm25=self.bm25 is not None)
            self.store.write_document(chunks)
        self.extracts.add_copy(detected.document_id, key, access, owner, chat_id)
        result.status, result.chunks = "processed", len(chunks)
        result.warning = "; ".join(warnings) or None

    @staticmethod
    def _log_result(name: str, result: FileResult) -> None:
        level = (log.error if result.status == "failed"
                 else log.warning if result.warning else log.info)
        level("file " + result.status, extra={
            "file": name, "status": result.status, "chunks": result.chunks,
            "duration_ms": round(result.seconds * 1000),
            **({"error": result.reason} if result.reason else {}),
            **({"warning": result.warning} if result.warning else {})})

    def with_options(self, frame_captions: bool = False, describe_pictures: bool = False,
                     use_gpu: bool | None = None) -> "Pipeline":
        """This pipeline, or a copy sharing its models and database with one upload's
        optional slow features turned on."""
        overrides = {}
        if frame_captions:
            overrides["frame_captions"] = True
        if describe_pictures:
            overrides["pdf_picture_descriptions"] = True
        if use_gpu is not None and use_gpu != (self.settings.gpu_backend == "modal"):
            overrides["gpu_backend"] = "modal" if use_gpu else "local"
        if not overrides:
            return self
        return Pipeline(self.settings.model_copy(update=overrides), embedder=self.embedder,
                        store=self.store, isolated=self.isolated, reranker=self._reranker,
                        count_tokens=self._count_tokens, bm25=self._bm25)

    # ---- maintenance ----

    def reindex(self, force: bool = False) -> RunReport:
        """Chunk, embed and store every saved extraction again, without re-reading any file:
        after a change of chunk size, embedding model or collection. Files that were
        deleted or changed since they were extracted are skipped (`ingest run` reads those)."""
        report = RunReport.start()
        for record in self.extracts.records():
            segments = None
            # Each stored copy comes back as it was: same owner, readers and chat
            for copy in copies_of(record):
                if copy["access"] is None:
                    copy = self._first_copy(copy["path"])
                key, owner = copy["path"], copy["owner"]
                result = FileResult(path=key, status="skipped", file_type=record["file_type"])
                started = time.perf_counter()
                try:
                    path = Path(key)
                    if not path.exists():
                        result.reason = "the file no longer exists"
                    elif (detected := detect(path)).document_id != record["document_id"]:
                        result.reason = "changed since it was extracted: run `ingest run` on it"
                    elif not force and (key, owner) in self.store.copies(detected.document_id):
                        result.reason = "already in this collection"
                    else:
                        if segments is None:
                            segments = self.extracts.load(record["document_id"], key=None)
                        self._index(detected, key, segments, copy["access"], result, owner,
                                    copy.get("chat_id"))
                except Exception as exc:
                    result.status, result.reason = "failed", f"{type(exc).__name__}: {exc}"
                result.seconds = round(time.perf_counter() - started, 2)
                self._log_result(Path(key).name, result)
                report.add(result)
        report.finish()
        return report

    def _first_copy(self, key: str) -> dict:
        """The copy of a saved extraction from before copies were recorded. A file in a
        user's own upload folder (uploads/NAME/) is theirs and stays private to them (never
        shared by guesswork); any other file gets its access.yaml groups."""
        uploads = self.settings.uploads_dir.resolve()
        path = Path(key).resolve()
        if uploads in path.parents and path.parent != uploads:
            owner = path.relative_to(uploads).parts[0]
            return {"path": key, "owner": owner, "access": [f"user:{owner}"], "chat_id": None}
        return {"path": key, "owner": None, "access": self.access.groups_for(key),
                "chat_id": None}

    def sync(self, folders: list[str | Path], force: bool = False) -> RunReport:
        """Make the database match these folders: ingest new and changed files, remove the
        chunks of files that were deleted from them, and re-apply access.yaml to the rest."""
        report = self.run(folders, force=force)
        roots = [source_key(Path(f)).rstrip("/") + "/" for f in folders]
        for stored, access in self.store.stored_paths().items():
            if not any(stored.startswith(root) for root in roots):
                continue
            if not Path(stored).exists():
                n = self.delete_key(stored)
                report.add(FileResult(path=stored, status="removed",
                                      reason=f"deleted from the folder: {n} chunk(s) removed"))
                log.info("file removed", extra={"file": Path(stored).name, "chunks": n})
            elif (groups := self.access.groups_for(stored)) != access:
                self.store.set_access(stored, groups)
                if document_id := self.store.document_of(stored):
                    self.extracts.update_copies(document_id, stored, access=groups)
        report.finish()
        return report

    def close(self) -> None:
        if self.isolated:
            self.isolated.close()
        self.store.close()

    def delete(self, path: str | Path) -> int:
        return self.delete_key(source_key(Path(path)))

    def delete_key(self, key: str) -> int:
        """Remove a stored path's chunks, and its copy from the saved extraction's record
        (the extraction itself is kept), so `ingest reindex` doesn't bring it back."""
        document_id = self.store.document_of(key)
        n = self.store.delete_path(key)
        if document_id:
            self.extracts.forget_copies(document_id, key)
        return n

    # ---- searching ----

    def search(self, query: str, limit: int = 5, file_type: str | None = None,
               groups: list[str] | None = None, hybrid: bool | None = None,
               rerank: bool | None = None, source_path: str | None = None,
               modality: str | None = None, exact: bool | None = None,
               chat_id: str | None = None, terms: list[str] = ()):
        """Best chunks for a question: dense + keyword ranking merged (hybrid), plus chunks
        holding the question's ids and figures as written (exact_match), then the top
        candidates re-scored by the cross-encoder. `groups` limits it to what those groups
        may read; `file_type`, `source_path`, `modality`, `chat_id` to those; `hybrid` /
        `rerank` / `exact` override the settings (evaluation, query routing); `terms`: more
        words to find as written (from the question router). retrieve.py turns these chunks
        into passages with their context."""
        from ingest.retrieve import exact_terms
        dense, sparse = self.embedder.embed_query(query)
        use_hybrid = self.settings.hybrid_search if hybrid is None else hybrid
        found = exact_terms(query) if (self.settings.exact_match if exact is None else exact) else []
        found = list(dict.fromkeys([*found, *terms]))[:6]
        reranker = self.reranker if (rerank is None or rerank) else None
        pool = max(limit, self.settings.rerank_candidates) if reranker else limit
        words = self.bm25.query(query) if use_hybrid and self.bm25 and self.store.has_bm25 \
            else None
        hits = self.store.search(dense, sparse if use_hybrid else None, pool, file_type, groups,
                                 source_path=source_path, modality=modality, exact=found,
                                 chat_id=chat_id, bm25=words)
        return reranker.rerank(query, hits)[:limit] if reranker else hits
