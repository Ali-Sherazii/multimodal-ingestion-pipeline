"""The pipeline end to end on text files: saved extractions, content-hash identity (moves and
copies), reindex, sync with deletions, permissions, hybrid search and reranking."""
import shutil

from conftest import ReverseReranker


def statuses(report):
    return {f.path.rsplit("/", 1)[-1]: (f.status, f.reason) for f in report.files}


def test_ingest_then_unchanged_is_skipped(make_pipeline, docs):
    pipeline = make_pipeline()
    first = pipeline.run([docs])
    assert first.totals["processed"] == 2 and first.totals["failed"] == 0
    again = statuses(pipeline.run([docs]))
    assert again["policy.md"] == ("skipped", "already ingested, unchanged")


def test_saved_extraction_is_reused_by_force_and_by_a_new_collection(make_pipeline, docs,
                                                                     count_extractions):
    make_pipeline().run([docs])
    assert sorted(count_extractions) == ["notes.txt", "policy.md"]
    make_pipeline().run([docs], force=True)                     # re-index, same collection
    make_pipeline(collection="other").run([docs])               # e.g. a new embedding model
    assert len(count_extractions) == 2, "files must not be read again"
    make_pipeline().run([docs], force=True, re_extract=True)
    assert len(count_extractions) == 4


def test_changed_extraction_settings_read_the_file_again(make_pipeline, docs, count_extractions):
    make_pipeline().run([docs])
    make_pipeline(pdf_ocr=False).run([docs], force=True)
    assert len(count_extractions) == 4


def test_moved_file_keeps_its_chunks_and_is_not_read_again(make_pipeline, docs, count_extractions):
    pipeline = make_pipeline()
    pipeline.run([docs])
    moved = docs / "renamed policy.md"
    (docs / "policy.md").rename(moved)
    result = pipeline.ingest_file(moved)
    assert result.status == "skipped" and result.reason.startswith("moved from")
    assert pipeline.store.stored_paths().keys() >= {moved.resolve().as_posix()}
    assert not any(p.endswith("/policy.md") for p in pipeline.store.stored_paths())
    assert len(count_extractions) == 2


def test_copy_is_not_stored_twice(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.run([docs])
    shutil.copy(docs / "notes.txt", docs / "notes copy.txt")
    result = pipeline.ingest_file(docs / "notes copy.txt")
    assert result.status == "skipped" and result.reason.startswith("same content as")


def test_changed_file_replaces_its_old_version(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.run([docs])
    (docs / "notes.txt").write_text("Completely new text about solar panels.", encoding="utf-8")
    assert pipeline.ingest_file(docs / "notes.txt").status == "processed"
    texts = [p.payload["content"] for p in pipeline.store.iter_points(with_vectors=False)
             if p.payload["source_path"].endswith("/notes.txt")]
    assert texts and all("solar" in t for t in texts)


def test_reindex_rebuilds_a_collection_without_reading_files(make_pipeline, docs, count_extractions):
    make_pipeline().run([docs])
    fresh = make_pipeline(collection="rebuilt")
    report = fresh.reindex()
    assert report.totals["processed"] == 2
    assert fresh.store.count() > 0 and len(count_extractions) == 2
    assert fresh.reindex().totals["processed"] == 0       # already there


def test_reindex_keeps_each_copy_private_to_its_owner(make_pipeline, settings, docs,
                                                      count_extractions):
    """A rebuilt collection must not turn private uploads into shared ones."""
    uploads = settings.uploads_dir
    for name in ("alice", "bob"):
        (uploads / name).mkdir(parents=True)
        shutil.copy(docs / "policy.md", uploads / name / "policy.md")
    pipeline = make_pipeline()
    pipeline.ingest_file(uploads / "alice" / "policy.md", groups=["user:alice"], owner="alice",
                         chat_id="chat-1")
    pipeline.ingest_file(uploads / "bob" / "policy.md", groups=["user:bob", "hr"], owner="bob")
    pipeline.ingest_file(docs / "notes.txt")

    fresh = make_pipeline(collection="rebuilt")
    assert fresh.reindex().totals["processed"] == 3 and len(count_extractions) == 2
    stored = {(p.payload["source_path"].split("/")[-2], p.payload["owner"],
               tuple(p.payload["access"]), p.payload["chat_id"])
              for p in fresh.store.iter_points(with_vectors=False)}
    assert stored == {("alice", "alice", ("user:alice",), "chat-1"),
                      ("bob", "bob", ("user:bob", "hr"), None),
                      ("docs", None, ("everyone",), None)}

    # Deleted copies stay deleted; the other user's copy and the extraction are kept
    fresh.delete(uploads / "alice" / "policy.md")
    newer = make_pipeline(collection="rebuilt-again")
    assert newer.reindex().totals["processed"] == 2
    assert {p.payload["owner"] for p in newer.store.iter_points(with_vectors=False)} == \
        {"bob", None}


def test_reindex_of_an_old_extraction_keeps_uploads_private(make_pipeline, settings, docs):
    """Extractions saved before copies were recorded: a file in a user's own upload
    folder comes back private to them, not shared with everyone."""
    import json
    folder = settings.uploads_dir / "alice"
    folder.mkdir(parents=True)
    shutil.copy(docs / "policy.md", folder / "policy.md")
    make_pipeline().ingest_file(folder / "policy.md", groups=["user:alice"], owner="alice")
    (record_path,) = settings.extracts_dir.glob("*.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    del record["copies"]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    fresh = make_pipeline(collection="rebuilt")
    assert fresh.reindex().totals["processed"] == 1
    assert {(p.payload["owner"], tuple(p.payload["access"]))
            for p in fresh.store.iter_points(with_vectors=False)} == {("alice", ("user:alice",))}


def test_sync_removes_deleted_files_and_applies_access_rules(make_pipeline, docs, settings):
    pipeline = make_pipeline()
    pipeline.sync([docs])
    (docs / "notes.txt").unlink()
    with open(settings.access_file, "w", encoding="utf-8") as f:
        f.write('default: [everyone]\nrules:\n  - match: "*/policy.md"\n    groups: [hr]\n')
    pipeline = make_pipeline()                             # picks up the new rules
    report = pipeline.sync([docs])
    assert report.totals["removed"] == 1
    stored = pipeline.store.stored_paths()
    assert list(stored.values()) == [["hr"]]


def test_search_respects_access_groups(make_pipeline, docs, settings):
    with open(settings.access_file, "w", encoding="utf-8") as f:
        f.write('rules:\n  - match: "*/policy.md"\n    groups: [hr]\n')
    pipeline = make_pipeline()
    pipeline.run([docs])
    anyone = pipeline.search("annual leave days", limit=10, groups=["everyone"])
    assert anyone and all(h.payload["source_path"].endswith("notes.txt") for h in anyone)
    hr = pipeline.search("annual leave days", limit=10, groups=["hr"])
    assert hr and all(h.payload["source_path"].endswith("policy.md") for h in hr)
    admin = pipeline.search("annual leave days", limit=10)          # no groups: everything
    assert {h.payload["source_path"].rsplit("/", 1)[-1] for h in admin} == {"policy.md", "notes.txt"}


def test_upload_groups_override_the_rules(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest_file(docs / "notes.txt", groups=["managers"])
    assert list(pipeline.store.stored_paths().values()) == [["managers"]]


def test_hybrid_search_finds_an_exact_code(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.run([docs])
    assert pipeline.store.hybrid
    top = pipeline.search("INV-77821", limit=1)[0]
    assert "INV-77821" in top.payload["content"]


def test_reranker_reorders_the_candidates(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.run([docs])
    candidates = pipeline.search("revenue growth Lahore", limit=30, rerank=False)
    pipeline._reranker = ReverseReranker()
    reranked = pipeline.search("revenue growth Lahore", limit=2)
    assert len(candidates) >= 3
    assert [h.id for h in reranked] == [h.id for h in reversed(candidates)][:2]
