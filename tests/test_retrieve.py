"""Retrieval (retrieve.py): exact ids and figures, passages with context, citations, and the
search API, with the word-count embedder and an in-memory database (no models)."""
from pathlib import Path

from ingest.models import DetectedFile, FileResult, FileType, Modality, Segment
from ingest.retrieve import Retriever, citation, exact_terms


def index(pipeline, name: str, file_type: FileType, segments: list[Segment]) -> None:
    detected = DetectedFile(Path(name), file_type, f"doc-{name}")
    result = FileResult(path=name, status="failed")
    pipeline._index(detected, name, segments, ["everyone"], result)
    assert result.status == "processed", result.reason


def test_ids_figures_quotes_and_references_are_looked_for_as_written():
    assert exact_terms('Who approved AP-317, and was "Table 9" right about 3,663 in FY2026?') == [
        "Table 9", "AP-317", "3,663", "FY2026"]
    assert exact_terms("What was revenue in Q3 2026?") == []  # short or plain: normal search


def test_an_exact_id_is_found_even_when_other_chunks_match_the_words_better(make_pipeline):
    pipeline = make_pipeline()
    rows = [Segment(f"Who approved it and for what amount was settled in meeting {i}.",
                    page_number=1) for i in range(40)]  # nearly the question's own words
    rows.append(Segment("| AP-317 | R. Khattak | 12,300,000 |", page_number=2))
    index(pipeline, "register.pdf", FileType.PDF, rows)
    question = "Who approved AP-317 and for what amount?"
    # meaning only (no keyword ranking), so only the exact-match list can bring the id in
    plain = pipeline.search(question, limit=5, hybrid=False, exact=False)
    exact = pipeline.search(question, limit=5, hybrid=False, exact=True)
    assert not any("AP-317" in h.payload["content"] for h in plain)
    assert any("AP-317" in h.payload["content"] for h in exact)


def test_text_passage_adds_the_chunks_around_it_without_repeating_the_overlap(make_pipeline):
    pipeline = make_pipeline(chunk_size=200, chunk_overlap=40)
    words = [f"word{i}" for i in range(120)]
    words[60] = "Zanzibar"  # a word only the middle of the section has
    index(pipeline, "notes.txt", FileType.TEXT,
          [Segment(" ".join(words), section_heading="Travel")])
    passages = Retriever(pipeline).search("Zanzibar", limit=3)
    first = passages[0]
    assert "Zanzibar" in first.match and len(first.chunk_ids) == 3      # match + before + after
    assert first.text.count("word59") == 1 and first.text.count("word61") == 1
    assert len(first.text) > len(first.match)
    assert first.citation == "notes.txt · Travel"
    shown = [i for p in passages for i in p.chunk_ids[:1]]
    assert len(shown) == len(set(shown))  # no match shown twice


def test_video_passage_has_the_screen_and_the_speech_around_the_moment(make_pipeline):
    pipeline = make_pipeline(chunk_size=60, chunk_overlap=0)  # one chunk per 10 s of speech
    segments = [Segment(f"Speech part {i} about general topics.", Modality.TRANSCRIPT,
                        start_time=10.0 * i, end_time=10.0 * i + 10) for i in range(12)]
    segments[6] = Segment("Bayes theorem flips the conditional probability.", Modality.TRANSCRIPT,
                          start_time=60.0, end_time=70.0)
    segments.append(Segment("On screen:\nP(star|Green) = 2/4", Modality.FRAME_CAPTION,
                            section_heading="Conditional probability", start_time=55.0,
                            end_time=75.0))
    index(pipeline, "lecture.mp4", FileType.VIDEO, segments)
    found = Retriever(pipeline).search("Bayes theorem flips", limit=1)[0]
    assert found.modality == "transcript" and found.citation == "lecture.mp4, 1:00-1:10"
    assert "On screen (0:55-1:15):" in found.text and "P(star|Green)" in found.text
    assert "Speech (0:20-1:50):" in found.text and "Speech part 4" in found.text
    assert "Speech part 11" not in found.text  # more than 30 s after the moment


def test_table_passage_is_the_chunk_itself_cited_by_its_caption(make_pipeline):
    pipeline = make_pipeline()
    table = ("Table 16b: Lahore Central depot bays\n\n| Bay | Free |\n| --- | --- |\n"
             "| B4 | 108 |")
    index(pipeline, "report.pdf", FileType.PDF,
          [Segment(table, Modality.TABLE, "16. Depot detail", page_number=20)])
    found = Retriever(pipeline).search("free pallet spaces bay B4 Lahore Central", limit=1)[0]
    assert found.text == table
    assert found.citation == "report.pdf, p. 20 · Table 16b: Lahore Central depot bays"


def test_citation_formats():
    assert citation({"source_path": "C:/docs/a.pdf", "page_number": 3,
                     "section_heading": "Intro", "content": "x"}) == "a.pdf, p. 3 · Intro"
    assert citation({"source_path": "talk.mp4", "start_time": 3725.0, "end_time": 3731.0,
                     "content": "x"}) == "talk.mp4, 1:02:05-1:02:11"


def test_search_api_returns_passages(make_pipeline, settings):
    from fastapi.testclient import TestClient
    from ingest.web.app import create_app
    pipeline = make_pipeline()
    index(pipeline, "memo.txt", FileType.TEXT, [Segment("Board meeting moved to Friday.")])
    found = TestClient(create_app(pipeline, settings)).get(
        "/api/search", params={"q": "When is the board meeting?"}).json()
    assert found[0]["citation"] == "memo.txt" and "Friday" in found[0]["text"]
