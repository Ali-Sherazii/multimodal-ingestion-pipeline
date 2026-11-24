"""Store (FR-11, FR-12, FR-15): chunks, metadata and vectors in Qdrant.

New collections hold two vectors per chunk: "dense" (meaning) and, with bge-m3, "sparse"
(word weights). A search then runs both and merges the two rankings (reciprocal rank fusion),
so a question finds chunks that mean the same thing and chunks with its exact terms.
Collections made before this (one unnamed vector) still work, with dense search only.
"""
import functools
import threading
import time
import warnings

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ingest.config import Settings
from ingest.logs import log
from ingest.models import Chunk

_INDEXED_FIELDS = {
    "document_id": models.PayloadSchemaType.KEYWORD,
    "source_path": models.PayloadSchemaType.KEYWORD,
    "file_type": models.PayloadSchemaType.KEYWORD,
    "modality": models.PayloadSchemaType.KEYWORD,
    "ingested_at": models.PayloadSchemaType.KEYWORD,
    "access": models.PayloadSchemaType.KEYWORD,
    # Each user's chunks stored together on disk (Qdrant's multitenancy), so a search limited
    # to one user's files reads only theirs
    "owner": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    "chat_id": models.PayloadSchemaType.KEYWORD,
}
DENSE, SPARSE, BM25 = "dense", "sparse", "bm25"
_UPSERT_BATCH = 256
EXACT_POOL = 20  # candidates per exact word (id, figure) in a query


def _match(field: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def dense_vector(point) -> list[float]:
    """A stored point's dense vector, whichever layout its collection has."""
    vector = point.vector
    return (vector.get(DENSE) or []) if isinstance(vector, dict) else (vector or [])


# How long a database call keeps trying when the connection drops (a cloud database over a
# patchy internet connection): every call here can be repeated safely, as the same chunk
# ids are written again and deleting twice deletes once.
RETRY_SECONDS = 60.0


def _transient(exc: Exception) -> bool:
    """A dropped or refused connection, a timeout, or the server briefly unavailable."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code >= 500 or exc.status_code == 429)
    return isinstance(exc, (ResponseHandlingException, ConnectionError, TimeoutError))


def _reason(exc: Exception) -> str:
    text = str(exc)
    for code, words in (("10054", "connection closed by the server"),
                        ("10061", "connection refused"), ("10060", "no answer"),
                        ("11001", "address not found: no internet?"), ("timed out", "timed out")):
        if code in text:
            return words
    return type(exc).__name__


def _retrying(call, seconds: float | None = None):
    """call(), tried again through a dropped connection for up to `seconds` (RETRY_SECONDS)."""
    seconds = RETRY_SECONDS if seconds is None else seconds
    deadline, wait, warned = time.monotonic() + seconds, 1.0, False
    while True:
        try:
            result = call()
            if warned:
                log.info("database connection is back")
            return result
        except Exception as exc:
            if not _transient(exc):
                raise
            if time.monotonic() + wait > deadline:
                raise ConnectionError(f"lost the connection to the database for over "
                                      f"{seconds:.0f} s ({_reason(exc)}): check the internet "
                                      "connection, then try again") from exc
            if not warned:
                log.warning(f"lost the connection to the database ({_reason(exc)}); "
                            f"trying again for up to {seconds:.0f} s")
                warned = True
            time.sleep(wait)
            wait = min(wait * 2, 8.0)


def _locked(method):
    """One database call at a time (the web app reads while its upload worker writes), tried
    again through a dropped connection."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return _retrying(lambda: method(self, *args, **kwargs))
    return wrapper


def _tuning(settings: Settings | None) -> dict:
    """The collection settings (graph, compression, disk) and search settings to use."""
    s = settings or Settings(_env_file=None)
    quantization = {
        "none": None,
        "int8": models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
        "binary": models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
            always_ram=True)),
    }[s.quantization]
    return {
        "hnsw": models.HnswConfigDiff(m=s.hnsw_m, ef_construct=s.hnsw_ef_construct),
        "quantization": quantization,
        "on_disk": s.vectors_on_disk,
        "params": models.SearchParams(
            hnsw_ef=s.search_ef,
            quantization=models.QuantizationSearchParams(
                rescore=True, oversampling=s.quantization_oversampling) if quantization else None),
        "fusion": models.Fusion.DBSF if s.fusion == "dbsf" else models.Fusion.RRF,
    }


class VectorStore:
    """Opening needs no embedding model; the collection is created on the first write."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int | None = None,
                 model_name: str | None = None, settings: Settings | None = None):
        self.client = client
        self.collection = collection
        self._ready = False
        self._named: bool | None = None   # vectors stored by name ("dense"/"sparse")
        self._sparse = False             # the collection holds sparse vectors too
        self._bm25 = False               # ... and BM25 word weights
        self._lock = threading.RLock()
        self.tuning = _tuning(settings)
        if dimension is not None:
            self.ensure_collection(dimension, model_name or "")

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorStore":
        if settings.qdrant_url:
            key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            client = QdrantClient(url=settings.qdrant_url, api_key=key)
            _wait_until_ready(client)
        elif settings.qdrant_path:
            try:
                # Calls are serialised by _lock, so the web app's threads may share it
                client = QdrantClient(path=settings.qdrant_path,
                                      force_disable_check_same_thread=True)
            except RuntimeError as exc:
                if "already accessed" not in str(exc):
                    raise
                raise RuntimeError(
                    f"The local database ({settings.qdrant_path}) is already open in another "
                    "ingest window. Close that one first (type q), then try again.") from None
        else:
            raise ValueError("Set qdrant_url (server) or qdrant_path (local folder)")
        return cls(client, settings.collection, settings=settings)

    def _exists(self) -> bool:
        if not self._ready:
            self._ready = self.client.collection_exists(self.collection)
            if self._ready:
                self._read_layout()
        return self._ready

    def _read_layout(self) -> None:
        params = self.client.get_collection(self.collection).config.params
        self._named = isinstance(params.vectors, dict)
        self._sparse = bool(params.sparse_vectors and SPARSE in params.sparse_vectors)
        self._bm25 = bool(params.sparse_vectors and BM25 in params.sparse_vectors)

    @property
    def hybrid(self) -> bool:
        """Searches can use keyword (sparse) vectors as well as dense ones."""
        with self._lock:
            return self._exists() and self._sparse

    @property
    def has_bm25(self) -> bool:
        with self._lock:
            return self._exists() and self._bm25

    @_locked
    def ensure_collection(self, dimension: int, model_name: str, sparse: bool = False,
                          bm25: bool = False) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            existing = vectors[DENSE].size if isinstance(vectors, dict) else vectors.size
            built_with = (info.config.metadata or {}).get("embedding_model")
            # Vectors from two models can't be compared, even when they have the same size
            if existing != dimension or (built_with and model_name and built_with != model_name):
                raise ValueError(
                    f"Collection '{self.collection}' was built with "
                    f"{built_with or f'a {existing}-dim model'}, but the embedding model is now "
                    f"{model_name} ({dimension}-dim). Set a new collection name in "
                    "config.yaml (or switch the model back).")
            self._ready = True
            self._read_layout()
            self._add_indexes(info.payload_schema or {})
            if bm25 and not self._bm25 and not getattr(self, "_warned_bm25", False):
                self._warned_bm25 = True
                log.warning(f"collection '{self.collection}' has no BM25 word weights, so "
                            "searches skip them: set a new `collection` name in config.yaml and "
                            "run `ingest reindex` (from saved extractions, nothing re-read)")
            return
        t = self.tuning
        keywords = {}
        if sparse:
            keywords[SPARSE] = models.SparseVectorParams()
        if bm25:  # Qdrant weighs each word by its rarity in the collection (IDF)
            keywords[BM25] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dimension, on_disk=t["on_disk"],
                                                       distance=models.Distance.COSINE)},
            sparse_vectors_config=keywords or None,
            hnsw_config=t["hnsw"], quantization_config=t["quantization"],
            metadata={"embedding_model": model_name},
        )
        self._ready, self._named, self._sparse, self._bm25 = True, True, sparse, bm25
        self._add_indexes({})

    def _add_indexes(self, existing: dict) -> None:
        """Index the filtered fields; a collection made before a field existed gets it now."""
        with warnings.catch_warnings():
            # Local (folder) mode ignores indexes and warns; a Qdrant server uses them
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in _INDEXED_FIELDS.items():
                if field not in existing:
                    self.client.create_payload_index(self.collection, field, schema)

    def _count(self, *conditions, must_not=()) -> int:
        if not self._exists():
            return 0
        flt = models.Filter(must=list(conditions), must_not=list(must_not))
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    @_locked
    def has_document(self, document_id: str) -> bool:
        return self._count(_match("document_id", document_id)) > 0

    @_locked
    def paths_for(self, document_id: str) -> set[str]:
        """Where this content (by its hash) is stored: normally one path, or none."""
        return {path for path, _ in self.copies(document_id)}

    @_locked
    def document_of(self, source_path: str) -> str | None:
        """The content (hash) stored at this path, if any."""
        if not self._exists():
            return None
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=["document_id"],
                                       scroll_filter=models.Filter(
                                           must=[_match("source_path", source_path)]))
        return points[0].payload["document_id"] if points else None

    @_locked
    def copies(self, document_id: str) -> set[tuple[str, str | None]]:
        """Each (path, uploader) holding this content (by its hash)."""
        if not self._exists():
            return set()
        found: set[tuple[str, str | None]] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["source_path", "owner"],
                scroll_filter=models.Filter(must=[_match("document_id", document_id)]))
            found.update((p.payload["source_path"], p.payload.get("owner")) for p in points)
            if offset is None:
                return found

    @_locked
    def move_document(self, document_id: str, source_path: str, access: list[str],
                      old_paths: list[str] | None = None) -> None:
        """Point a document's chunks (those at `old_paths`, or all) at its new location
        (moved or renamed, same content)."""
        must = [_match("document_id", document_id)]
        if old_paths is not None:
            must.append(models.FieldCondition(key="source_path",
                                              match=models.MatchAny(any=list(old_paths))))
        self.client.set_payload(self.collection, {"source_path": source_path, "access": access},
                                points=models.Filter(must=must), wait=True)

    @_locked
    def set_access(self, source_path: str, access: list[str]) -> None:
        self.client.set_payload(self.collection, {"access": access}, wait=True,
                                points=models.Filter(must=[_match("source_path", source_path)]))

    def stored_paths(self) -> dict[str, list[str]]:
        """Every stored file's path, with the access groups its chunks carry."""
        paths: dict[str, list[str]] = {}
        for point in self.iter_points(with_vectors=False, payload=["source_path", "access"]):
            paths.setdefault(point.payload["source_path"], point.payload.get("access") or [])
        return paths

    @_locked
    def files(self, groups: list[str] | None = None, chat_id: str | None = None,
              limit: int = 2000) -> dict[str, int]:
        """The stored files these groups may read (all with None), with their chunk counts:
        one request (Qdrant's facet count), not a scan of every chunk."""
        if not self._exists():
            return {}
        must = []
        if groups is not None:
            must.append(models.FieldCondition(key="access", match=models.MatchAny(any=groups)))
        if chat_id:
            must.append(_match("chat_id", chat_id))
        found = self.client.facet(self.collection, key="source_path", limit=limit, exact=True,
                                  facet_filter=models.Filter(must=must) if must else None)
        return {hit.value: hit.count for hit in found.hits}

    @_locked
    def set_chat(self, source_path: str, chat_id: str | None) -> None:
        """Attach a stored file to a chat (or detach it: None)."""
        self.client.set_payload(self.collection, {"chat_id": chat_id}, wait=True,
                                points=models.Filter(must=[_match("source_path", source_path)]))

    def _vector(self, chunk: Chunk):
        if not self._named:
            return chunk.embedding
        vector = {DENSE: chunk.embedding}
        if self._sparse and chunk.sparse:
            vector[SPARSE] = models.SparseVector(indices=list(chunk.sparse),
                                                 values=list(chunk.sparse.values()))
        if self._bm25 and chunk.bm25:
            vector[BM25] = models.SparseVector(indices=list(chunk.bm25),
                                               values=list(chunk.bm25.values()))
        return vector

    @_locked
    def write_document(self, chunks: list[Chunk]) -> None:
        """Write one file's chunks, then remove that path's previous version.

        All-or-nothing (NFR-06): if any batch fails, the batches already written
        are deleted again, and the previous version stays untouched.
        """
        if not chunks:
            return
        source_path, ingested_at = chunks[0].source_path, chunks[0].ingested_at
        try:
            for i in range(0, len(chunks), _UPSERT_BATCH):
                batch = chunks[i:i + _UPSERT_BATCH]
                self.client.upsert(self.collection, wait=True, points=[
                    models.PointStruct(id=c.chunk_id, vector=self._vector(c), payload=c.payload())
                    for c in batch])
        except Exception:
            # Includes ids never written; deleting a missing id is a no-op
            self.client.delete(self.collection, wait=True, points_selector=models.PointIdsList(
                points=[c.chunk_id for c in chunks]))
            raise
        # Anything else this uploader has stored for this path is an older version (changed
        # file or --force); another user's copy at the same path is theirs, and stays
        owner = chunks[0].owner
        same_owner = (_match("owner", owner) if owner else
                      models.IsEmptyCondition(is_empty=models.PayloadField(key="owner")))
        self._delete(models.Filter(must=[_match("source_path", source_path), same_owner],
                                   must_not=[_match("ingested_at", ingested_at)]))

    @_locked
    def delete_path(self, source_path: str) -> int:
        n = self._count(_match("source_path", source_path))
        if n:
            self._delete(models.Filter(must=[_match("source_path", source_path)]))
        return n

    def _delete(self, flt: models.Filter) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt),
                           wait=True)

    @_locked
    def search(self, vector: list[float], sparse: dict[int, float] | None = None,
               limit: int = 5, file_type: str | None = None,
               groups: list[str] | None = None, source_path: str | None = None,
               modality: str | None = None, exact: list[str] = (),
               chat_id: str | None = None,
               bm25: dict[int, float] | None = None) -> list[models.ScoredPoint]:
        """Best chunks for a query. With `sparse` (and a hybrid collection), the dense and
        keyword rankings are merged by reciprocal rank fusion. `exact`: words the chunk should
        contain as written (an id such as KF-26-5001, a figure such as 3,663): chunks holding
        each one are ranked as a third list, so an exact match is never missing from the
        candidates. `groups`: only chunks readable by at least one of them; `file_type`,
        `source_path`, `modality`, `chat_id` (files uploaded into that chat): only those."""
        if not self._exists():
            return []
        must = [_match(field, value) for field, value in (
            ("file_type", file_type), ("source_path", source_path), ("modality", modality),
            ("chat_id", chat_id)) if value]
        if groups is not None:
            must.append(models.FieldCondition(key="access", match=models.MatchAny(any=groups)))
        flt = models.Filter(must=must) if must else None
        using = DENSE if self._named else None
        params = self.tuning["params"]
        pool = max(limit * 4, 50)  # each ranking's candidates before fusion
        rankings = [models.Prefetch(query=vector, using=using, limit=pool, filter=flt,
                                    params=params)]
        if self._sparse and sparse:
            keywords = models.SparseVector(indices=list(sparse), values=list(sparse.values()))
            rankings.append(models.Prefetch(query=keywords, using=SPARSE, limit=pool, filter=flt))
        if self._bm25 and bm25:
            words = models.SparseVector(indices=list(bm25), values=list(bm25.values()))
            rankings.append(models.Prefetch(query=words, using=BM25, limit=pool, filter=flt))
        for term in exact:
            holding = models.FieldCondition(key="content", match=models.MatchText(text=term))
            rankings.append(models.Prefetch(query=vector, using=using, limit=EXACT_POOL,
                                            filter=models.Filter(must=must + [holding]),
                                            params=params))
        if len(rankings) == 1:
            with warnings.catch_warnings():  # local (folder) mode always searches exactly
                warnings.filterwarnings("ignore", message="Local mode performs exact")
                return self.client.query_points(self.collection, query=vector, limit=limit,
                                                using=using, query_filter=flt,
                                                search_params=params, with_payload=True).points
        return self.client.query_points(self.collection, limit=limit, with_payload=True,
                                        prefetch=rankings,
                                        query=models.FusionQuery(fusion=self.tuning["fusion"])).points

    @_locked
    def apply_tuning(self) -> None:
        """Give an existing collection the configured graph, compression and disk settings
        (Qdrant rebuilds its index in the background; searches keep working meanwhile)."""
        if not self._exists():
            return
        t = self.tuning
        self.client.update_collection(
            self.collection, hnsw_config=t["hnsw"],
            vectors_config={DENSE: models.VectorParamsDiff(on_disk=t["on_disk"])} if self._named
            else None,
            quantization_config=t["quantization"] or models.Disabled.DISABLED)

    @_locked
    def around(self, document_id: str, first: int, last: int,
               groups: list[str] | None = None) -> list:
        """A document's chunks from position `first` to `last` (context around a match)."""
        return self._range(document_id, "chunk_index", models.Range(gte=first, lte=last), groups)

    @_locked
    def during(self, document_id: str, start: float, end: float,
               groups: list[str] | None = None) -> list:
        """A video's chunks that overlap the time from `start` to `end` seconds."""
        found = self._range(document_id, "start_time", models.Range(lte=end), groups)
        return [p for p in found if (p.payload.get("end_time") or 0) >= start]

    def _range(self, document_id: str, field: str, bounds: models.Range, groups) -> list:
        if not self._exists():
            return []
        must = [_match("document_id", document_id),
                models.FieldCondition(key=field, range=bounds)]
        if groups is not None:
            must.append(models.FieldCondition(key="access", match=models.MatchAny(any=groups)))
        points, _ = self.client.scroll(self.collection, scroll_filter=models.Filter(must=must),
                                       limit=200, with_payload=True, with_vectors=False)
        return points

    @_locked
    def count(self) -> int:
        if not self._exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def iter_points(self, with_vectors: bool = True, source_path: str | None = None,
                    payload: bool | list[str] = True, groups: list[str] | None = None):
        """Stored chunks page by page (all, or one file's), with payload and optionally vectors;
        `groups`: only chunks readable by at least one of them."""
        with self._lock:
            if not _retrying(self._exists):
                return
        must = [_match("source_path", source_path)] if source_path else []
        if groups is not None:
            must.append(models.FieldCondition(key="access", match=models.MatchAny(any=groups)))
        flt = models.Filter(must=must) if must else None
        offset = None
        while True:
            with self._lock:
                points, offset = _retrying(lambda: self.client.scroll(
                    self.collection, limit=256, offset=offset, scroll_filter=flt,
                    with_payload=payload, with_vectors=with_vectors))
            yield from points
            if offset is None:
                return

    def close(self) -> None:
        self.client.close()


def _wait_until_ready(client: QdrantClient, timeout: float = RETRY_SECONDS) -> None:
    """A Qdrant server may still be starting up, or the connection be down for a moment."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            if "401" in str(exc) or "403" in str(exc) or "Unauthorized" in str(exc):
                raise ConnectionError("Qdrant rejected the API key. Check INGEST_QDRANT_API_KEY "
                                      "in your .env file.") from None
            if time.monotonic() > deadline:
                raise ConnectionError(
                    f"Could not reach the Qdrant server for {timeout:.0f} s ({_reason(exc)}). "
                    "Check the internet connection and try again; if it keeps failing, check "
                    "INGEST_QDRANT_URL in .env, or remove it to use the local folder.") from None
            time.sleep(1)
