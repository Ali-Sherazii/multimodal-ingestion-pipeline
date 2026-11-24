"""Video scenes: where the picture changes, one keyframe per scene, its on-screen text.

A frame is captured only when the picture has changed: a slide shown for ten minutes gives one
frame. Both checks below share the single pass PySceneDetect makes over the video.

1. Cuts: PySceneDetect's adaptive detector finds hard cuts and slide changes. Edges count as
   well as colour, because two slides differ mostly in their text, not their colours. It
   compares each frame with its neighbours, so camera motion doesn't fire it.
2. Gradual changes: a dissolve, a fade or scrolling changes each frame too little to look like
   a cut. Twice a second a small greyscale copy of the picture is kept; once the picture has
   settled, it is compared with the scene's own picture, and if it differs, a new scene starts
   where the change began.
3. One keyframe per scene, at its last still moment (a slide that builds up is complete, and
   nothing is mid-scroll or mid-fade), is read with OCR. Consecutive scenes that look the same
   are merged, keeping the later frame plus any lines only the earlier frames showed.
4. Keyframes whose text is laid out in rows and columns, or contains maths signs, are
   flagged; video.py then reads their tables and equations with Docling.
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.config import Settings
from ingest.logs import log
from ingest.ocr import confident_lines, looks_like_table, looks_mathematical, read_text


@dataclass
class Scene:
    start: float
    end: float
    keyframe: Path | None = None
    lines: list[str] = field(default_factory=list)   # on-screen text, top to bottom
    fingerprint: int = 0
    caption: str = ""
    # lines earlier frames of this scene showed that the keyframe no longer does (a caption
    # or a row that came and went while the rest of the slide stayed)
    earlier_lines: list[str] = field(default_factory=list)
    table_like: bool = False   # text laid out in rows and columns: worth Docling's table model
    math_like: bool = False    # text with maths signs: worth Docling's formula model
    boxes: list = field(default_factory=list, repr=False)  # OCR text with positions, if either
    tables: list[str] = field(default_factory=list)  # Markdown tables read from the keyframe
    formulas: list[str] = field(default_factory=list)  # equations as LaTeX

    @property
    def title(self) -> str | None:
        """The first on-screen line long enough to be a title (a slide's heading)."""
        return next((line for line in self.text_lines if len(line) >= 3), None)

    @property
    def all_lines(self) -> list[str]:
        return self.lines + self.earlier_lines

    @property
    def text_lines(self) -> list[str]:
        """On-screen lines that are not already part of a table read from this frame."""
        if not self.tables:
            return self.all_lines
        cells = {w for table in self.tables for w in table.replace("|", " ").lower().split()}
        return [line for line in self.all_lines
                if not all(w in cells for w in line.lower().split())]


def _seconds(timecode) -> float:
    return timecode.seconds if hasattr(timecode, "seconds") else timecode.get_seconds()


# The change check works on small greyscale thumbnails. Measured on test videos with these
# values: compression noise changes 0% of the pixels, scrolling or panning 13-27% between
# samples, a new bullet point 1.4%, two different slides with the same template 1.9%.
SAMPLES_PER_SECOND = 2
THUMB_SIZE = (96, 54)
PIXEL_LEVELS = 16  # a thumbnail pixel has changed if its grey level moved by more than this
STILL = 0.003      # under 0.3% of the picture changing between samples: nothing is moving
CHANGED = 0.008    # over 0.8% different from the scene's own picture: a new view

Sample = tuple[float, np.ndarray]  # (seconds, greyscale thumbnail)


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the picture that differs between two thumbnails."""
    return float(np.mean(np.abs(a.astype(np.int16) - b) > PIXEL_LEVELS))


def _settled(samples: list[Sample], i: int) -> bool:
    """Nothing has moved for the last second (the last three samples are alike)."""
    return i >= 2 and all(_changed(samples[j][1], samples[j - 1][1]) <= STILL for j in (i, i - 1))


def _thumbnail_recorder(fps: float):
    """A PySceneDetect detector that finds no cuts itself, only keeps thumbnails."""
    import cv2
    from scenedetect import SceneDetector

    class ThumbnailRecorder(SceneDetector):
        def __init__(self):
            super().__init__()
            self.step = max(1, round(fps / SAMPLES_PER_SECOND))
            self.samples: list[Sample] = []
            self.frames = 0

        def process_frame(self, timecode, frame_img):
            if self.frames % self.step == 0:
                grey = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                self.samples.append((_seconds(timecode), cv2.resize(
                    grey, THUMB_SIZE, interpolation=cv2.INTER_AREA)))
            self.frames += 1
            return []

    return ThumbnailRecorder()


def detect_cuts(path: Path, settings: Settings) -> tuple[list[tuple[float, float]], list[Sample]]:
    """Scenes split at cuts, plus the thumbnails, from one pass over the video."""
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
    video = open_video(str(path))
    detector = AdaptiveDetector(
        min_content_val=settings.scene_min_change,
        min_scene_len=max(1, round(settings.scene_min_seconds * video.frame_rate)),
        weights=ContentDetector.Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0,
                                           delta_edges=1.0))
    recorder = _thumbnail_recorder(video.frame_rate)
    manager = SceneManager()
    manager.auto_downscale = True
    manager.add_detector(detector)
    manager.add_detector(recorder)
    manager.detect_scenes(video)
    spans = [(_seconds(a), _seconds(b)) for a, b in manager.get_scene_list(start_in_scene=True)]
    return spans, recorder.samples


def _merge_tiny(spans: list[tuple[float, float]], min_len: float) -> list[tuple[float, float]]:
    """Fold scenes shorter than min_len into the one before (e.g. a glitch on the last frame)."""
    out: list[tuple[float, float]] = []
    for start, end in spans:
        if out and end - start < min_len:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    if len(out) > 1 and out[0][1] - out[0][0] < min_len:
        out[1] = (out[0][0], out[1][1])
        out.pop(0)
    return out


def _drop_false_cuts(spans: list[tuple[float, float]],
                     samples: list[Sample]) -> list[tuple[float, float]]:
    """Undo cuts where the picture is the same on both sides.

    On a still, noisy slide the cut detector fires wherever the encoder stores a full frame
    (every few seconds), because the compression noise jumps there; the thumbnails show
    0% change across those points, against at least 1% for a real change.
    """
    out = spans[:1]
    for start, end in spans[1:]:
        before = [s for s in samples if s[0] < start]
        after = [s for s in samples if start + 0.5 <= s[0] < end]
        if before and after and _changed(before[-1][1], after[0][1]) <= STILL:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def _split_gradual(spans: list[tuple[float, float]], samples: list[Sample],
                   min_len: float) -> list[tuple[float, float]]:
    """Start a new scene where the picture changed gradually (dissolve, fade, scrolling).

    Each settled picture is compared with the scene's own picture, not with the frame before,
    so a change spread over seconds still adds up. A moving shot never settles, so it is not
    split.
    """
    out = []
    for start, end in spans:
        inside = [s for s in samples if start <= s[0] < end]
        scene_start, view, change_began = start, None, None
        for i, (t, thumb) in enumerate(inside):
            if view is None:  # the scene's own picture: its first settled moment
                if _settled(inside, i):
                    view = thumb
                continue
            difference = _changed(thumb, view)
            if difference <= STILL:
                change_began = None
            elif change_began is None:
                change_began = t
            if difference > CHANGED and _settled(inside, i):
                if change_began - scene_start >= min_len and end - change_began >= min_len:
                    out.append((scene_start, change_began))
                    scene_start = change_began
                view, change_began = thumb, None
        out.append((scene_start, end))
    return out


def _keyframe_time(start: float, end: float, samples: list[Sample]) -> float:
    """The scene's last still moment; for a shot that never stops moving, two-thirds in."""
    inside = [s for s in samples if start <= s[0] < end]
    still = [inside[i][0] for i in range(len(inside)) if _settled(inside, i)]
    latest = max(end - 0.25, start)
    return min(still[-1], latest) if still else min(start + (end - start) * 2 / 3, latest)


def _fingerprint(image: Image.Image) -> int:
    """64-bit difference hash: near-identical frames get near-identical numbers."""
    small = image.convert("L").resize((9, 8))
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _line_key(line: str) -> str:
    """The same line read from two frames may differ only in spacing or case."""
    return "".join(line.casefold().split())


def _same_view(a: Scene, b: Scene) -> bool:
    """Same slide or shot: pictures nearly identical, and b's text contains a's text."""
    if bin(a.fingerprint ^ b.fingerprint).count("1") > 10:
        return False
    words_a = {w.lower() for line in a.lines for w in line.split()}
    words_b = {w.lower() for line in b.lines for w in line.split()}
    return not words_a or len(words_a & words_b) / len(words_a) >= 0.8


def _keyframe(path: Path, at: float, out: Path) -> Path | None:
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{at:.3f}",
                             "-i", str(path), "-frames:v", "1", "-q:v", "2", str(out)],
                            capture_output=True)
    return out if result.returncode == 0 and out.exists() else None


def analyse_scenes(path: Path, workdir: Path, duration: float,
                   settings: Settings) -> tuple[list[Scene], list[str]]:
    """Scenes with keyframes and on-screen text, plus warnings."""
    warnings: list[str] = []
    samples: list[Sample] = []
    if settings.video_scenes:
        try:
            spans, samples = detect_cuts(path, settings)
        except Exception as exc:  # e.g. a codec OpenCV can't read; the transcript still counts
            warnings.append(f"scene detection failed ({type(exc).__name__}: {str(exc).strip()[:100]}); "
                            "on-screen text was not read")
            return [], warnings
        min_len = settings.scene_min_seconds
        spans = _merge_tiny(_drop_false_cuts(spans, samples), min_len)
        spans = _split_gradual(spans, samples, min_len)
    else:  # scene detection switched off: a frame every frame_interval_seconds
        step = settings.frame_interval_seconds
        spans = [(t, min(t + step, duration)) for t in
                 [i * step for i in range(max(1, int(duration // step) + (duration % step > 0)))]]

    scenes: list[Scene] = []
    for i, (start, end) in enumerate(spans):
        at = _keyframe_time(start, end, samples)
        frame = _keyframe(path, at, workdir / f"scene_{i:05d}.jpg")
        if frame is None:
            continue
        with Image.open(frame) as img:
            found = read_text(img) if settings.video_scene_ocr else []
            table_like = settings.video_tables and looks_like_table(found)
            math_like = looks_mathematical(found)  # formula model (GPU) and fractions
            scene = Scene(start=round(start, 2), end=round(end, 2), keyframe=frame,
                          fingerprint=_fingerprint(img), lines=confident_lines(found),
                          table_like=table_like, math_like=math_like,
                          boxes=found if table_like or math_like else [])
        if scenes and _same_view(scenes[-1], scene):
            # Same slide still showing (or built up further): keep the later, fuller frame,
            # plus any text the earlier frames showed that it no longer does
            previous = scenes[-1]
            scene.start = previous.start
            seen = {_line_key(line) for line in scene.lines}
            for line in previous.earlier_lines + previous.lines:
                if _line_key(line) not in seen:
                    scene.earlier_lines.append(line)
                    seen.add(_line_key(line))
            scenes[-1] = scene
        else:
            scenes.append(scene)
    log.info(f"{len(spans)} scene(s) found, {len(scenes)} after merging repeats",
             extra={"file": path.name})
    return scenes, warnings
