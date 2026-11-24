"""Shared test fixtures. The fast tests load no AI models: a word-count embedder stands in for
bge-m3 (dense + sparse), the database is Qdrant in memory, and extraction runs in-process."""
import math
import re
import zlib
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from ingest.config import Settings
from ingest.pipeline import Pipeline
from ingest.store import VectorStore


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


class FakeEmbedder:
    """Deterministic stand-in: dense = normalised word counts hashed into 64 slots; sparse =
    one weight per distinct word. Similar texts get similar vectors, exact words match."""
    model_name = "fake-embedder"
    dimension = 64
    has_sparse = True

    def _dense(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in _words(text):
            vector[zlib.crc32(word.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def _sparse(self, text: str) -> dict[int, float]:
        return {zlib.crc32(w.encode()) % 100_000: 1.0 for w in set(_words(text))}

    def embed_documents(self, texts):
        return [self._dense(t) for t in texts], [self._sparse(t) for t in texts]

    def embed_query(self, text):
        return self._dense(text), self._sparse(text)


class FakeBm25:
    """Stand-in for embed.Bm25: term counts per hashed word; Qdrant adds the IDF."""
    def _weights(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for word in _words(text):
            key = zlib.crc32(("bm25:" + word).encode()) % 100_000
            counts[key] = counts.get(key, 0.0) + 1.0
        return counts

    def documents(self, texts):
        return [self._weights(t) for t in texts]

    def query(self, text):
        return {k: 1.0 for k in self._weights(text)}


class ReverseReranker:
    """Stand-in cross-encoder that simply reverses the order, to see that it is applied."""
    def rerank(self, query, hits):
        for i, hit in enumerate(reversed(hits)):
            hit.score = float(len(hits) - i)
        return list(reversed(hits))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, qdrant_url=None, qdrant_path=None,
                    extracts_dir=tmp_path / "extracted", reports_dir=tmp_path / "reports",
                    logs_dir=tmp_path / "logs", uploads_dir=tmp_path / "uploads",
                    exports_dir=tmp_path / "exports", access_file=str(tmp_path / "access.yaml"),
                    isolate_extraction=False, reranker_model="", gpu_backend="local",
                    bm25=False, app_db=tmp_path / "app.db", outputs_dir=tmp_path / "outputs")


@pytest.fixture
def make_pipeline(settings: Settings):
    """A pipeline on an in-memory database; call again with collection= for a second one."""
    client = QdrantClient(":memory:")
    made = []

    def make(collection: str = "test", **overrides) -> Pipeline:
        s = settings.model_copy(update=overrides)
        pipeline = Pipeline(s, embedder=FakeEmbedder(), store=VectorStore(client, collection),
                            count_tokens=lambda text: len(text.split()),  # no tokenizer to load
                            bm25=FakeBm25() if s.bm25 else None)
        made.append(pipeline)
        return pipeline

    yield make
    client.close()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.md").write_text(
        "# Leave Policy\n\nEmployees get 20 days of paid annual leave each year.\n\n"
        "# Expenses\n\nInvoice INV-77821 must be approved by finance before payment.\n",
        encoding="utf-8")
    (folder / "notes.txt").write_text(
        "Quarterly review: revenue grew 12 percent and the Lahore office led growth.",
        encoding="utf-8")
    return folder


@pytest.fixture
def count_extractions(monkeypatch):
    """How many times a file was actually read (not served from a saved extraction)."""
    import ingest.pipeline as pipeline_module
    calls = []
    real = pipeline_module.get_extractor

    def counting(file_type):
        extract = real(file_type)

        def run(path, settings):
            calls.append(Path(path).name)
            return extract(path, settings)
        return run

    monkeypatch.setattr(pipeline_module, "get_extractor", counting)
    return calls
