"""Run the slow extraction steps on a Modal cloud GPU (gpu_backend: modal).

Sent to the GPU: PDF conversion (layout, tables, OCR, picture descriptions), Word files when
picture descriptions are on, video transcription (only the compressed audio is uploaded),
tables and equations in video frames and video frame captions (only those frames
are uploaded). Detection,
scene detection, chunking, embedding and storage stay on this machine.

The Modal app starts on first use and stops a few minutes after the last GPU step (or when
this program exits), so the GPU always runs this exact code. Models are downloaded once into the Modal volume "ingest-models".
"""
import atexit
import contextlib
import io
import logging
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import modal

from ingest.config import Settings
from ingest.logs import log
from ingest.models import Extraction, Modality, Segment

_PY = "3.12"
_NVIDIA = f"/usr/local/lib/python{_PY}/site-packages/nvidia"

# Image build step: Docling's OCR engine keeps its models inside the installed package, so
# fetch them into the image. Layout/table models go to a throwaway folder (they live in the
# volume at run time).
_PREFETCH_OCR = (
    "HF_HOME=/tmp/hf-build python -c \"from docling.datamodel.base_models import InputFormat; "
    "from docling.document_converter import DocumentConverter; "
    "DocumentConverter().initialize_pipeline(InputFormat.PDF)\" && rm -rf /tmp/hf-build")

# Everything except this package's code, so other apps (eval/tatdqa_eval.py) can add layers
base_image = (
    modal.Image.debian_slim(python_version=_PY)
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "docling>=2.60", "faster-whisper>=1.1", "transformers",
        "pillow", "numpy", "langchain-text-splitters", "charset-normalizer",
        "pydantic-settings", "pyyaml",
        # faster-whisper (CTranslate2) needs CUDA 12 cuBLAS and cuDNN 9
        "nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
    .env({"LD_LIBRARY_PATH": f"{_NVIDIA}/cublas/lib:{_NVIDIA}/cudnn/lib",
          "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    .run_commands(_PREFETCH_OCR)
    .env({"HF_HOME": "/models/hf"})
)
image = base_image.add_local_python_source("ingest")
models_volume = modal.Volume.from_name("ingest-models", create_if_missing=True)
app = modal.App("ingest-gpu", image=image)


def _remote_settings(values: dict) -> Settings:
    return Settings(_env_file=None, **{
        **values, "gpu_backend": "local", "device": "cuda",
        "whisper_model": values["gpu_whisper_model"],
        "vision_model": values["gpu_vision_model"]})


def _segments_out(segments: list[Segment]) -> list[dict]:
    return [{**asdict(s), "modality": s.modality.value} for s in segments]


@app.cls(gpu="L4", volumes={"/models": models_volume}, scaledown_window=120, timeout=3600)
class GpuWorker:
    @modal.method()
    def info(self) -> dict:
        import ctranslate2
        import torch
        return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "ctranslate2_cuda_devices": ctranslate2.get_cuda_device_count()}

    @modal.method()
    def convert_document(self, filename: str, data: bytes, settings: dict) -> list[dict]:
        from ingest.extractors.docling_doc import extract_document
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            path.write_bytes(data)
            segments = extract_document(path, _remote_settings(settings))
        models_volume.commit()  # keep any model downloaded on first use
        return {"segments": _segments_out(segments),
                "warnings": list(getattr(segments, "warnings", []))}

    @modal.method()
    def transcribe(self, audio: bytes, settings: dict) -> list[dict]:
        from ingest.extractors.video import transcribe
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.ogg"
            path.write_bytes(audio)
            segments = transcribe(path, _remote_settings(settings))
        models_volume.commit()
        return _segments_out(segments)

    @modal.method()
    def read_frames(self, images: list[bytes], maths: list[bool], settings: dict) -> list[dict]:
        """Tables, and equations where `maths`, in video frames (see read_image)."""
        from ingest.extractors.docling_doc import formulas_on, read_image
        remote = _remote_settings(settings)
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (data, math) in enumerate(zip(images, maths, strict=True)):
                path = Path(tmp) / f"frame_{i:04d}.jpg"
                path.write_bytes(data)
                out.append(read_image(path, remote, formulas=math and formulas_on(remote)))
        models_volume.commit()  # keep the formula model after its first download
        return out

    @modal.method()
    def caption(self, images: list[bytes], prompt: str, settings: dict) -> list[str]:
        from PIL import Image
        from ingest.vision import get_captioner
        captioner = get_captioner(_remote_settings(settings))
        captions = [captioner.caption(Image.open(io.BytesIO(b)), prompt) for b in images]
        models_volume.commit()
        return captions


# ---- called on this machine ----
#
# A running Modal app needs this machine to check in with Modal every 15 s. If the laptop
# sleeps or goes offline, Modal stops the app, and a call made on it afterwards waits forever
# (tested). So: the app is open only while GPU work is happening (closed after IDLE_SECONDS
# without a call), a stopped app is noticed from the failed check-ins and closed (which also
# ends a call waiting on it), and a call cut off that way is made again on a new app.

IDLE_SECONDS = 300
_GONE = (modal.exception.ClientClosed, modal.exception.ConflictError)


class _Session:
    """One running Modal app."""
    def __init__(self):
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(modal.enable_output())
        self.stack.enter_context(app.run())
        self.active = 0      # calls in progress
        self.closed = False


_lock = threading.RLock()
_session: _Session | None = None
_idle_timer: threading.Timer | None = None


def _open(settings: Settings) -> _Session:
    """The running app, started if needed (the very first time also builds the image)."""
    global _session
    with _lock:
        if _idle_timer:
            _idle_timer.cancel()
        if _session is None:
            # Modal prints symbols like "✓"; a Windows console redirected to a file can't encode
            # them, which must never stop a run
            for stream in (sys.stdout, sys.stderr):
                if stream and (stream.encoding or "").lower().replace("-", "") != "utf8":
                    stream.reconfigure(errors="replace")
            log.info(f"starting Modal GPU ({settings.modal_gpu}); the very first run builds "
                     "the image and downloads models, which takes several minutes")
            _session = _Session()
        _session.active += 1
        return _session


def _done(session: _Session) -> None:
    global _idle_timer
    with _lock:
        session.active -= 1
        if session is _session and not session.active:
            _idle_timer = threading.Timer(IDLE_SECONDS, _close, (session, True))
            _idle_timer.daemon = True
            _idle_timer.start()


def _close(session: _Session | None = None, only_if_idle: bool = False) -> None:
    """Stop the app (default: the current one). Calls waiting on it end with ClientClosed."""
    global _session
    with _lock:  # held while closing, so a new app never starts before the old one is gone
        session = session or _session
        if session is None or session.closed or (only_if_idle and session.active):
            return
        session.closed = True
        if session is _session:
            _session = None
        try:
            session.stack.close()
        except Exception as exc:  # already stopped by Modal: nothing left to stop
            log.debug(f"closing the Modal app: {type(exc).__name__}: {exc}")


atexit.register(_close)


class _CheckInWatch(logging.Filter):
    """Modal logs every failed check-in with a full traceback, every 15 s, for as long as the
    program runs. Instead: one line, and when Modal answers that the app has been stopped,
    close it, so the next GPU step starts a new one rather than waiting forever."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "heartbeat" not in record.getMessage():
            return True
        error = record.exc_info[1] if record.exc_info else None
        session = _session
        if isinstance(error, modal.exception.ConflictError) and session and not session.closed:
            log.warning("Modal stopped the GPU app (this computer slept or lost its "
                        "connection); the next GPU step starts a new one")
            # not here: this runs on Modal's own event loop, which closing needs
            threading.Thread(target=_close, args=(session,), daemon=True).start()
        return False


# The check-in loop logs to "modal-utils" (its lines say "[modal-client]")
for _name in ("modal-utils", "modal-client"):
    logging.getLogger(_name).addFilter(_CheckInWatch())


def _remote(settings: Settings, method: str, *args):
    """GpuWorker.<method>.remote(*args), made again on a new app if Modal stopped the app."""
    for attempt in (1, 2):
        session = _open(settings)
        try:
            worker = GpuWorker.with_options(gpu=settings.modal_gpu)()
            return getattr(worker, method).remote(*args)
        except _GONE:
            if attempt == 2:
                raise
            log.warning("the Modal app was stopped during a GPU step (this computer slept or "
                        "lost its connection); trying it again on a new one")
            _close(session)
        finally:
            _done(session)


_SETTINGS_SENT = {"chunk_size", "chunk_overlap", "pdf_ocr", "pdf_force_full_page_ocr",
                  "pdf_picture_descriptions", "pdf_page_batch", "pdf_hidden_text_filter",
                  "pdf_watermark_filter",
                  "pdf_garbage_retry", "picture_prompt", "docx_image_ocr",
                  "picture_min_px", "whisper_language", "vision_prompt",
                  "gpu_whisper_model", "gpu_vision_model", "formulas", "model_revisions"}


def _settings_out(settings: Settings) -> dict:
    # Only what extraction needs: never the database URL or API key
    return settings.model_dump(mode="json", include=_SETTINGS_SENT)


def _segments_in(rows: list[dict]) -> list[Segment]:
    return [Segment(**{**r, "modality": Modality(r["modality"])}) for r in rows]


def gpu_info(settings: Settings) -> dict:
    return _remote(settings, "info")


def convert_document(path: Path, settings: Settings) -> Extraction:
    out = _remote(settings, "convert_document", path.name, path.read_bytes(),
                  _settings_out(settings))
    return Extraction(_segments_in(out["segments"]), out["warnings"])


def compress_audio(path: Path) -> bytes:
    """The audio track as 16 kHz mono Opus, about 11 MB per hour: all that is uploaded."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "-"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stdout


def transcribe(path: Path, settings: Settings) -> list[Segment]:
    rows = _remote(settings, "transcribe", compress_audio(path), _settings_out(settings))
    return _segments_in(rows)


def read_frames(images: list[Path], maths: list[bool], settings: Settings) -> list[dict]:
    """Upload only the video frames that look like tables or equations; returns, per frame,
    {"tables": [(markdown, region)], "formulas": [(latex, region)]}."""
    data = [p.read_bytes() for p in images]
    return _remote(settings, "read_frames", data, maths, _settings_out(settings))


def caption_images(images: list[Path], prompt: str, settings: Settings) -> list[str]:
    data = [p.read_bytes() for p in images]
    return _remote(settings, "caption", data, prompt, _settings_out(settings))
