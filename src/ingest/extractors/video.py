"""Video extractor: ffmpeg + Whisper transcript (FR-06), on-screen text and tables per scene,
optional frame captions (FR-07)."""
import json
import re
import subprocess
import tempfile
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ingest.config import Settings
from ingest.logs import log
from ingest.extractors.scenes import Scene, analyse_scenes
from ingest.models import Extraction, Modality, Segment
from ingest.ocr import column_tables, stacked_fractions, texts_inside
from ingest.vision import get_captioner


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr.strip()[-500:]}")
    return result.stdout


def probe(path: Path) -> tuple[float, bool, bool]:
    """Return (duration_seconds, has_audio, has_video)."""
    info = json.loads(_run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:stream=codec_type", "-of", "json", str(path)]))
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    duration = float(info.get("format", {}).get("duration") or 0)
    return duration, "audio" in kinds, "video" in kinds


@lru_cache(maxsize=1)
def _whisper(model_name: str, cuda: bool, revision: str | None = None):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cuda" if cuda else "cpu",
                        compute_type="float16" if cuda else "int8", revision=revision)


def _whisper_revision(model_name: str, settings: Settings) -> str | None:
    """The pinned version of a Whisper size name ("base" is Systran/faster-whisper-base)."""
    from faster_whisper.utils import _MODELS
    return settings.model_revisions.get(_MODELS.get(model_name, model_name))


def load_audio(path: Path):
    """Decode the audio track with ffmpeg to 16 kHz mono float32, the format Whisper expects.

    About 230 MB of memory per hour of audio.
    """
    import numpy as np
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
                             "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def transcribe(path: Path, settings: Settings) -> list[Segment]:
    model = _whisper(settings.whisper_model, settings.use_cuda(),
                     _whisper_revision(settings.whisper_model, settings))
    segments, _info = model.transcribe(load_audio(path), language=settings.whisper_language,
                                       vad_filter=True)
    out = [Segment(text=s.text.strip(), modality=Modality.TRANSCRIPT,
                   start_time=round(s.start, 2), end_time=round(s.end, 2))
           for s in segments if s.text.strip()]
    log.info("transcribed", extra={"file": path.name, "chunks": len(out)})
    return out


def _caption(scenes: list[Scene], settings: Settings) -> None:
    """Describe each scene's keyframe with the vision model (only when frame_captions is on)."""
    frames = [sc.keyframe for sc in scenes]
    if settings.gpu_backend == "modal":  # upload just the keyframes, caption them on the GPU
        from ingest import gpu_modal
        captions = gpu_modal.caption_images(frames, settings.vision_prompt, settings)
    else:
        captioner = get_captioner(settings)
        captions = []
        for frame in frames:
            with Image.open(frame) as img:
                captions.append(captioner.caption(img, settings.vision_prompt))
    for scene, caption in zip(scenes, captions, strict=True):
        scene.caption = "" if _repeats_text(caption, scene.all_lines) else caption.strip()


# Answers made only of these say nothing about the picture ("Black background. White text.")
_GENERIC = {"a", "an", "and", "the", "of", "with", "on", "in", "image", "slide", "screen",
            "black", "white", "dark", "background", "text", "texts", "blackboard", "board",
            "equation", "equations", "formula", "formulas", "shapes", "colours", "colors"}


def _repeats_text(caption: str, lines: list[str]) -> bool:
    """Small vision models often just copy a slide's title ("Training Dta.") or say nothing
    ("Blackboard."); such answers add nothing to the OCR and are dropped."""
    words = lambda text: " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    said = words(caption)
    if not said or all(w in _GENERIC or w.isdigit() for w in said.split()):
        return True
    shown = [words(line) for line in lines] + [words(" ".join(lines))]
    return any(SequenceMatcher(None, said, text).ratio() >= 0.8 for text in shown if text)


def _letters(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _inside_formula(line: str, formulas: list[str]) -> bool:
    """An OCR line that is a piece of an equation already written as LaTeX ("P(Green)",
    "10"), e.g. read from an earlier frame of the scene, where its position isn't known."""
    letters = _letters(line)
    return len(letters) >= 2 and any(letters in _letters(latex) for latex in formulas)


def _rebuild_fractions(scenes: list[Scene]) -> None:
    """Fractions written with words (P(star|Green) = Total Green Stars / Total Green), from
    the OCR positions: no model, so this works on a CPU too. The OCR lines they replace,
    and the same pieces read from earlier frames of the scene, are dropped."""
    for scene in scenes:
        if not scene.boxes:
            continue
        # text_lines: not table cells, and not lines already read inside a formula
        for latex, replaced in stacked_fractions(scene.boxes, keep=set(scene.text_lines)):
            scene.formulas.append(latex)
            scene.lines = [line for line in scene.lines if line not in replaced]
        if scene.formulas:
            scene.earlier_lines = [line for line in scene.earlier_lines
                                   if not _inside_formula(line, scene.formulas)]


def _read_frames(scenes: list[Scene], settings: Settings) -> None:
    """Tables and equations, in the frames that look like they hold them, read with Docling:
    tables with its table model (plus narrow one-column tables it takes for pictures, rebuilt
    from the OCR positions), equations as LaTeX with its formula model (GPU only). The plain
    OCR of each equation is then dropped, so its broken lines don't sit next to the LaTeX."""
    formulas = settings.formulas and (settings.gpu_backend == "modal" or settings.use_cuda())
    todo = [sc for sc in scenes
            if (settings.video_tables and sc.table_like) or (formulas and sc.math_like)]
    if not todo:
        return
    frames = [sc.keyframe for sc in todo]
    maths = [formulas and sc.math_like for sc in todo]
    if settings.gpu_backend == "modal":  # upload just those frames
        from ingest import gpu_modal
        found = gpu_modal.read_frames(frames, maths, settings)
    else:
        from ingest.extractors.docling_doc import read_image, release_image_converter
        found = [read_image(frame, settings, formulas=math) for frame, math in zip(frames, maths)]
        release_image_converter()  # about 1 GB; the embedding model needs the memory next
    counts = {"tables": 0, "narrow": 0, "formulas": 0}
    for scene, result in zip(todo, found, strict=True):
        if settings.video_tables:
            scene.tables = [markdown for markdown, _ in result["tables"]]
            regions = [region for _, region in result["tables"] if region]
            narrow = column_tables(scene.boxes, regions)
            scene.tables += narrow
            counts["tables"] += len(result["tables"])
            counts["narrow"] += len(narrow)
        if result["formulas"]:
            scene.formulas = [latex for latex, _ in result["formulas"]]
            read = texts_inside(scene.boxes, [region for _, region in result["formulas"] if region])
            keep = lambda line: line not in read and not _inside_formula(line, scene.formulas)
            scene.lines = [line for line in scene.lines if keep(line)]
            scene.earlier_lines = [line for line in scene.earlier_lines if keep(line)]
            counts["formulas"] += len(result["formulas"])
    log.info(f"{counts['tables']} table(s) read by Docling, {counts['narrow']} one-column "
             f"table(s) from text positions and {counts['formulas']} equation(s) as LaTeX, "
             f"in {len(todo)} frame(s)")


def scene_segments(transcript: list[Segment], scenes: list[Scene]) -> list[Segment]:
    """One segment per scene for what was on screen; speech labelled with its scene's title.

    The scene's title (a slide's heading) becomes the section heading of everything said
    while it was showing, so a question about either finds both. Each table on screen is a
    segment of its own, a Markdown table, so the chunker splits it by rows and repeats the
    header; its cells are left out of the plain on-screen text.
    """
    out: list[Segment] = []
    showing: dict[str, Segment] = {}  # tables on screen in the previous scene
    for scene in scenes:
        parts = []
        if scene.text_lines:
            parts.append("On screen:\n" + "\n".join(scene.text_lines))
        if scene.formulas:
            parts.append("Formulas:\n" + "\n".join(f"${latex}$" for latex in scene.formulas))
        if scene.caption:
            parts.append("Visual description (approximate): " + scene.caption)
        if parts:
            out.append(Segment(text="\n\n".join(parts), modality=Modality.FRAME_CAPTION,
                               section_heading=scene.title, start_time=scene.start,
                               end_time=scene.end))
        current = {}
        for table in scene.tables:
            if table in showing:  # still on screen: one chunk, stretched to cover every scene
                showing[table].end_time = scene.end
                current[table] = showing[table]
            else:
                current[table] = Segment(text=table, modality=Modality.FRAME_CAPTION,
                                         section_heading=scene.title, start_time=scene.start,
                                         end_time=scene.end)
                out.append(current[table])
        showing = current
    for seg in transcript:
        middle = ((seg.start_time or 0) + (seg.end_time or 0)) / 2
        scene = next((sc for sc in scenes if sc.start <= middle < sc.end), None)
        seg.section_heading = scene.title if scene else None
        out.append(seg)
    return out


def extract_video(path: Path, settings: Settings) -> Extraction:
    duration, has_audio, has_video = probe(path)
    look = has_video and (settings.video_scene_ocr or settings.frame_captions)
    if not has_audio and not look:
        raise ValueError("video has no audio track, and on-screen text and frame captions "
                         "are turned off")

    transcript: list[Segment] = []
    if has_audio and settings.gpu_backend == "modal":
        from ingest import gpu_modal
        transcript = gpu_modal.transcribe(path, settings)
    elif has_audio:
        transcript = transcribe(path, settings)
    if duration:  # Whisper can put the last segment's end past the end of the audio
        for seg in transcript:
            seg.start_time = min(seg.start_time or 0.0, duration)
            seg.end_time = min(seg.end_time or 0.0, round(duration, 2))
    if not look:
        return Extraction(transcript)

    with tempfile.TemporaryDirectory(prefix="ingest-") as tmp:
        scenes, warnings = analyse_scenes(path, Path(tmp), duration, settings)
        _read_frames(scenes, settings)
        _rebuild_fractions(scenes)
        if settings.frame_captions and scenes:
            _caption(scenes, settings)
    return Extraction(scene_segments(transcript, scenes), warnings)
