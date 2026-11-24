"""The Modal connection on this machine, with Modal itself replaced by stand-ins: a call cut
off by a stopped app is made again on a new one, a stopped app is closed when Modal says so,
and an idle app is closed. No GPU, no network."""
import contextlib
import logging
import time

import pytest

modal = pytest.importorskip("modal")
from ingest import gpu_modal  # noqa: E402
from ingest.config import Settings  # noqa: E402


class FakeSession:
    started = 0

    def __init__(self):
        FakeSession.started += 1
        self.stack = contextlib.ExitStack()
        self.active = 0
        self.closed = False


class FakeWorker:
    """GpuWorker whose info() fails as a stopped app does, `failures` times."""
    failures = 0

    @classmethod
    def with_options(cls, **_):
        return cls

    class info:  # noqa: N801  (stands in for the Modal method of the same name)
        @staticmethod
        def remote():
            if FakeWorker.failures:
                FakeWorker.failures -= 1
                raise modal.exception.ClientClosed("app stopped")
            return {"gpu": "fake"}


@pytest.fixture()
def fake_modal(monkeypatch):
    monkeypatch.setattr(gpu_modal, "_Session", FakeSession)
    monkeypatch.setattr(gpu_modal, "GpuWorker", FakeWorker)
    monkeypatch.setattr(gpu_modal, "_session", None)
    FakeSession.started, FakeWorker.failures = 0, 0
    yield
    gpu_modal._close()


def test_a_call_cut_off_by_a_stopped_app_is_made_again_on_a_new_one(fake_modal):
    FakeWorker.failures = 1
    assert gpu_modal.gpu_info(Settings()) == {"gpu": "fake"}
    assert FakeSession.started == 2


def test_a_second_failure_is_reported(fake_modal):
    FakeWorker.failures = 2
    with pytest.raises(modal.exception.ClientClosed):
        gpu_modal.gpu_info(Settings())


def test_app_closes_when_idle_and_when_modal_stops_it(fake_modal, monkeypatch):
    monkeypatch.setattr(gpu_modal, "IDLE_SECONDS", 0.05)
    gpu_modal.gpu_info(Settings())
    first = gpu_modal._session
    time.sleep(0.3)
    assert first.closed and gpu_modal._session is None

    gpu_modal.gpu_info(Settings())
    second = gpu_modal._session
    try:
        raise modal.exception.ConflictError("App state is APP_STATE_STOPPED")
    except modal.exception.ConflictError:
        # what Modal logs after a failed check-in
        logging.getLogger("modal-utils").exception("Loop attempt for _run_app.<locals>.heartbeat failed")
    for _ in range(50):
        if second.closed:
            break
        time.sleep(0.02)
    assert second.closed and gpu_modal._session is None
