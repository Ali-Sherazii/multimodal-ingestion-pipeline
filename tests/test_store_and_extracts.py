"""Vector store layouts and saved extractions."""
import pytest
from qdrant_client import QdrantClient, models

from ingest.config import Settings
from ingest.extracts import ExtractStore, extraction_key
from ingest.models import Chunk, Extraction, FileType, Modality, Segment
from ingest.store import VectorStore, dense_vector


def chunk(i: int, text: str, path: str = "C:/a.txt", doc: str = "d1", access=("everyone",)):
    import uuid
    return Chunk(chunk_id=str(uuid.uuid4()), document_id=doc, source_path=path,
                 file_type=FileType.TEXT, content=text, modality=Modality.TEXT,
                 ingested_at="2026-10-13T00:00:00", chunk_index=i, access=list(access),
                 embedding=[1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0],
                 sparse={i + 1: 1.0})


def test_old_single_vector_collection_still_works():
    client = QdrantClient(":memory:")
    client.create_collection("old", vectors_config=models.VectorParams(
        size=3, distance=models.Distance.COSINE))
    store = VectorStore(client, "old")
    store.ensure_collection(3, "fake", sparse=True)
    store.write_document([chunk(0, "alpha"), chunk(1, "beta")])
    assert not store.hybrid
    hits = store.search([1.0, 0.0, 0.0], sparse={1: 1.0}, limit=1)
    assert hits[0].payload["content"] == "alpha"
    assert dense_vector(next(store.iter_points())) in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_new_collection_is_hybrid_and_filters_by_group():
    store = VectorStore(QdrantClient(":memory:"), "new")
    store.ensure_collection(3, "fake", sparse=True)
    store.write_document([chunk(0, "public", access=["everyone"]),
                          chunk(1, "secret", access=["hr"])])
    assert store.hybrid
    hr = store.search([0.0, 1.0, 0.0], sparse={2: 1.0}, limit=5, groups=["hr"])
    assert [h.payload["content"] for h in hr] == ["secret"]
    assert len(dense_vector(next(store.iter_points()))) == 3


def test_model_mismatch_is_refused():
    store = VectorStore(QdrantClient(":memory:"), "c")
    store.ensure_collection(3, "model-a")
    with pytest.raises(ValueError, match="built with model-a"):
        VectorStore(store.client, "c").ensure_collection(3, "model-b")


def test_move_and_access_updates():
    store = VectorStore(QdrantClient(":memory:"), "c")
    store.ensure_collection(3, "fake", sparse=True)
    store.write_document([chunk(0, "x", path="C:/old.txt")])
    store.move_document("d1", "C:/new.txt", ["managers"])
    assert store.paths_for("d1") == {"C:/new.txt"}
    store.set_access("C:/new.txt", ["hr"])
    assert store.stored_paths() == {"C:/new.txt": ["hr"]}


def test_saved_extraction_round_trip(tmp_path):
    settings = Settings(_env_file=None)
    store = ExtractStore(tmp_path)
    extraction = Extraction([Segment("hello", Modality.TRANSCRIPT, "Intro", None, 1.0, 2.5)],
                            ["a warning"])
    key = extraction_key(settings)
    store.save("abc", FileType.VIDEO, "C:/v.mp4", extraction, key, "modal")
    loaded = store.load("abc", key)
    assert loaded == extraction and loaded.warnings == ["a warning"]
    assert store.load("abc", "different-settings") is None
    assert store.load("abc", None) is not None
    store.set_source("abc", "C:/moved.mp4")
    assert [r["source_path"] for r in store.records()] == ["C:/moved.mp4"]


def test_extraction_key_ignores_settings_that_dont_change_extraction():
    base = Settings(_env_file=None)
    assert extraction_key(base) == extraction_key(base.model_copy(update={"chunk_size": 999}))
    assert extraction_key(base) == extraction_key(base.model_copy(update={"gpu_backend": "modal"}))
    assert extraction_key(base) != extraction_key(base.model_copy(update={"pdf_ocr": False}))


# ---- a cloud database over a patchy connection ----

class Dropping:
    """A Qdrant client whose calls fail with a dropped connection `drops` times first."""
    def __init__(self, real: QdrantClient, drops: int):
        self.real, self.drops = real, drops

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if self.drops:
                self.drops -= 1
                from qdrant_client.http.exceptions import ResponseHandlingException
                raise ResponseHandlingException(ConnectionResetError(
                    "[WinError 10054] An existing connection was forcibly closed by the remote host"))
            return attr(*args, **kwargs)
        return call


def test_database_calls_go_on_through_a_dropped_connection(monkeypatch):
    import ingest.store as store_module
    monkeypatch.setattr(store_module.time, "sleep", lambda s: None)
    real = QdrantClient(":memory:")
    store = VectorStore(real, "test")
    store.ensure_collection(3, "fake")
    store.write_document([chunk(0, "Board meeting moved to Friday.")])
    store.client = Dropping(real, drops=3)
    assert store.count() == 1                                     # tried again: same answer
    assert len(list(store.iter_points(with_vectors=False))) == 1


def test_a_connection_that_stays_down_gives_a_plain_message(monkeypatch):
    import ingest.store as store_module
    monkeypatch.setattr(store_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(store_module, "RETRY_SECONDS", 0.0)
    real = QdrantClient(":memory:")
    store = VectorStore(real, "test")
    store.ensure_collection(3, "fake")
    store.client = Dropping(real, drops=99)
    with pytest.raises(ConnectionError, match="connection closed by the server"):
        store.count()


def test_web_page_gets_a_short_answer_when_the_database_is_unreachable(make_pipeline, settings):
    from fastapi.testclient import TestClient
    from ingest.web.app import create_app
    pipeline = make_pipeline()
    def down():
        raise ConnectionError("lost the connection to the database for over 60 s")
    pipeline.store.count = down
    res = TestClient(create_app(pipeline, settings)).get("/api/status")
    assert res.status_code == 503 and "can't be reached" in res.json()["detail"]
