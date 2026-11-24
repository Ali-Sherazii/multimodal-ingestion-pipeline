"""Server mode end to end with an in-memory Redis: the web API queues the upload, a worker
takes the job off the queue, and the API reports the result."""
import fakeredis
from fastapi.testclient import TestClient

from ingest import jobs
from ingest.web.app import create_app


def test_upload_is_processed_by_a_queue_worker(make_pipeline, settings, monkeypatch):
    redis = fakeredis.FakeRedis()
    monkeypatch.setattr(jobs, "connect", lambda url: redis)
    server = settings.model_copy(update={"queue_url": "redis://in-memory/0"})
    pipeline = make_pipeline()
    client = TestClient(create_app(pipeline, server))
    assert client.get("/api/status").json()["mode"] == "queue"

    job = client.post("/api/upload", data={"groups": "hr"},
                      files={"files": ("memo.txt", b"Payroll closes on the 25th.")}).json()
    assert job["files"][0]["status"] == "queued" and pipeline.store.count() == 0

    jobs.run_worker(pipeline, server.queue_url, burst=True, connection=redis)

    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state["status"] == "done" and state["report"]
    assert state["files"][0]["status"] == "processed" and state["files"][0]["chunks"] == 1
    assert list(pipeline.store.stored_paths().values()) == [["hr"]]


def test_failed_job_reports_its_error(make_pipeline, settings, monkeypatch, tmp_path):
    redis = fakeredis.FakeRedis()
    monkeypatch.setattr(jobs, "connect", lambda url: redis)
    server = settings.model_copy(update={"queue_url": "redis://in-memory/0"})
    pipeline = make_pipeline()
    client = TestClient(create_app(pipeline, server))
    job = client.post("/api/upload", files={"files": ("empty.txt", b"   ")}).json()
    jobs.run_worker(pipeline, server.queue_url, burst=True, connection=redis)
    file = client.get(f"/api/jobs/{job['id']}").json()["files"][0]
    assert file["status"] == "failed" and "no text" in file["reason"]
