"""Video helpers on synthetic inputs: scene decisions, OCR layout checks, tables, formulas,
captions and how a scene becomes segments."""
import numpy as np

from ingest.extractors import scenes as sc
from ingest.extractors.docling_doc import compact_latex
from ingest.extractors.scenes import Scene
from ingest.extractors.video import _inside_formula, _rebuild_fractions, _repeats_text, scene_segments
from ingest.models import Modality, Segment
from ingest.ocr import (column_tables, looks_like_table, looks_mathematical, stacked_fractions,
                        texts_inside)


def box(text, left, top, width=40, height=20, score=0.99):
    return (text, score, (left, top, left + width, top + height))


# ---- scenes ----

def thumb(value: int, mark: int | None = None) -> np.ndarray:
    image = np.full((54, 96), value, dtype=np.uint8)
    if mark is not None:
        image[10:30, mark:mark + 30] = 255 - value
    return image


def test_cut_with_no_visible_change_is_dropped():
    samples = [(t / 2, thumb(40)) for t in range(40)]           # 20 s of a still slide
    spans = [(0.0, 10.0), (10.0, 20.0)]                          # an encoder "cut" at 10 s
    assert sc._drop_false_cuts(spans, samples) == [(0.0, 20.0)]


def test_gradual_change_starts_a_new_scene_where_it_began():
    still_a = [(t / 2, thumb(40)) for t in range(0, 20)]                      # 0-9.5 s
    fading = [(10 + t / 2, thumb(40 + 10 * (t + 1))) for t in range(4)]       # 10-11.5 s
    still_b = [(12 + t / 2, thumb(80, mark=20)) for t in range(16)]            # 12-19.5 s
    spans = sc._split_gradual([(0.0, 20.0)], still_a + fading + still_b, min_len=2)
    assert len(spans) == 2 and 9.5 <= spans[1][0] <= 12.0


def test_moving_shot_is_not_split():
    moving = [(t / 2, thumb(40, mark=(t * 7) % 60)) for t in range(40)]
    assert sc._split_gradual([(0.0, 20.0)], moving, min_len=2) == [(0.0, 20.0)]


def test_merged_frames_keep_text_only_earlier_frames_showed():
    earlier = Scene(0, 5, lines=["Why is it called random?", "x0 <= 4.3"], earlier_lines=["(Bagging)"])
    later = Scene(5, 9, lines=["x0  <=  4.3"])
    seen = {sc._line_key(line) for line in later.lines}
    for line in earlier.earlier_lines + earlier.lines:
        if sc._line_key(line) not in seen:
            later.earlier_lines.append(line)
            seen.add(sc._line_key(line))
    assert later.all_lines == ["x0  <=  4.3", "(Bagging)", "Why is it called random?"]


# ---- OCR layout ----

def test_table_layout_is_recognised_and_bullets_are_not():
    grid = [box(t, 50 + 60 * c, 50 + 30 * r) for r, row in enumerate(
        [["id", "x0", "y"], ["0", "4.3", "1"], ["1", "3.9", "0"]]) for c, t in enumerate(row)]
    bullets = [box(f"Point number {i}", 50, 50 + 30 * i, width=200) for i in range(6)]
    assert looks_like_table(grid) and not looks_like_table(bullets)


def test_one_column_tables_are_rebuilt_outside_docling_tables():
    main = [box(t, 50 + 60 * c, 50 + 30 * r) for r, row in enumerate(
        [["id", "x0"], ["0", "4.3"], ["1", "3.9"], ["2", "2.7"]]) for c, t in enumerate(row)]
    sample = [box(t, 400, 50 + 30 * r, width=20) for r, t in enumerate(["id", "2", "0", "2", "4"])]
    tables = column_tables(main + sample, skip=[(40, 40, 200, 170)])
    assert tables == ["| id |\n|---|\n| 2 |\n| 0 |\n| 2 |\n| 4 |"]


def test_maths_check_and_regions():
    assert looks_mathematical([box("P(star) :", 0, 0)])           # "=" lost by OCR
    assert looks_mathematical([box("a = b + c", 0, 0)])
    assert not looks_mathematical([box("Training Data", 0, 0)])
    found = [box("P(Green) =", 100, 100), box("Title", 10, 10)]
    assert texts_inside(found, [(90, 90, 300, 200)]) == {"P(Green) ="}


# OCR of the Naive Bayes video at 1:00: P(star|Green) = Total Green Stars / Total Green
WORD_FRACTION = [("Total Green Stars", 1.0, (616, 468, 1216, 539)),
                 ("P(star|Green)", 1.0, (48, 497, 508, 576)),
                 ("Total Green", 1.0, (721, 556, 1112, 622))]


def test_fraction_written_with_words_is_rebuilt_from_positions():
    scene = Scene(60.5, 69.0, lines=[t for t, _, _ in WORD_FRACTION], boxes=WORD_FRACTION,
                  earlier_lines=["Total Green", "Naive Bayes"])
    _rebuild_fractions([scene])
    assert scene.formulas == [r"P(star|Green)=\frac{\text{Total Green Stars}}{\text{Total Green}}"]
    assert scene.lines == [] and scene.earlier_lines == ["Naive Bayes"]


def test_lines_that_are_not_fractions_are_left_alone():
    # a probability already worked out, beside the training-data table's cells
    worked = [box("P(Yes) = 0.625", 50, 100, width=200), box("Outlook", 300, 70, width=90),
              box("Humidity", 300, 130, width=90)]
    # a function name beside a left-aligned list
    listed = [box("f(x)", 50, 100), box("Points measured in each experiment", 120, 70, width=300),
              box("Errors", 120, 130, width=60)]
    # a tiny subscript under a word
    subscript = [box("P(A) =", 50, 100, width=80), box("Total count", 150, 70, width=150),
                 box("i", 215, 130, width=10, height=12)]
    for found in (worked, listed, subscript):
        assert stacked_fractions(found) == []


# ---- formulas and captions ----

def test_latex_is_compacted_but_commands_keep_their_space():
    raw = r"P ( S t a r \cap G r e e n ) = \frac { 4 } { 1 0 } \cdot P ( A )"
    assert compact_latex(raw) == r"P(Star\cap Green)=\frac{4}{10}\cdot P(A)"


def test_formula_fragments_and_empty_captions_are_dropped():
    assert _inside_formula("P(Green)", [r"P(Green)=\frac{4}{10}"])
    assert not _inside_formula("Outlook: Rain", [r"P(Yes)=\frac{1}{2}"])
    assert _repeats_text("Training Dta.", ["Training Data"])
    assert _repeats_text("Black background. White text.", [])
    assert not _repeats_text("Green, red, circle, star.", ["P(Green) ="])


# ---- scene -> segments ----

def test_scene_segments_tables_formulas_and_captions():
    table = "| id |\n|---|\n| 0 |\n| 1 |"
    scenes = [Scene(0, 30, lines=["Intro", "id"], tables=[table], formulas=["a=b"],
                    caption="Red circles."),
              Scene(30, 60, lines=["Next"], tables=[table])]
    speech = [Segment("we start", Modality.TRANSCRIPT, start_time=10, end_time=12)]
    out = scene_segments(speech, scenes)
    on_screen = [s for s in out if s.modality is Modality.FRAME_CAPTION]
    assert on_screen[0].text == ("On screen:\nIntro\n\nFormulas:\n$a=b$\n\n"
                                 "Visual description (approximate): Red circles.")
    tables = [s for s in on_screen if s.text.startswith("|")]
    assert len(tables) == 1 and (tables[0].start_time, tables[0].end_time) == (0, 60)
    assert out[-1].section_heading == "Intro"
