"""BM25 word weights as a fourth ranking (with dense, bge-m3 keywords and exact terms)."""
import logging

from qdrant_client import models


def test_bm25_is_stored_and_searched(make_pipeline, docs, monkeypatch):
    pipeline = make_pipeline(bm25=True)
    pipeline.run([docs])
    assert pipeline.store.has_bm25
    sent = []
    real = pipeline.store.client.query_points

    def spy(*args, **kwargs):
        sent.append(kwargs.get("prefetch") or [])
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline.store.client, "query_points", spy)
    hits = pipeline.search("Lahore office growth", limit=2)
    assert hits[0].payload["source_path"].endswith("notes.txt")
    assert any(p.using == "bm25" for p in sent[0])


def test_an_older_collection_without_bm25_still_works(make_pipeline, docs, caplog):
    older = make_pipeline(collection="old", bm25=False)
    older.run([docs])
    newer = make_pipeline(collection="old", bm25=True)  # same collection, BM25 now switched on
    with caplog.at_level(logging.WARNING):
        newer.ingest_file(docs / "policy.md", force=True)
    assert not newer.store.has_bm25
    assert any("no BM25" in r.getMessage() for r in caplog.records)
    assert newer.search("paid annual leave", limit=1)


def test_tuning_settings_reach_the_collection(make_pipeline):
    from ingest.store import _tuning
    from ingest.config import Settings
    t = _tuning(Settings(_env_file=None, quantization="int8", vectors_on_disk=True,
                         hnsw_m=32, search_ef=128, fusion="dbsf"))
    assert isinstance(t["quantization"], models.ScalarQuantization) and t["on_disk"]
    assert t["hnsw"].m == 32 and t["params"].hnsw_ef == 128
    assert t["params"].quantization.rescore and t["fusion"] == models.Fusion.DBSF
    assert _tuning(Settings(_env_file=None, quantization="none"))["params"].quantization is None
