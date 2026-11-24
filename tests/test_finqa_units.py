"""FinQA benchmark: which gold items each retrieval unit holds, and the scoring (eval/finqa)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "finqa"))
from finqa_units import coverage, our_chunks, paper_passages, row_units, score_question  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def test_table_rows_carry_their_headers_and_row_zero_is_a_unit():
    table = [["", "2015", "2014"], ["net revenue", "$ 5829", "$ 5735"], ["2016", "$ 1", ""]]
    units = row_units("ETR/2016/page_23.pdf", table)
    assert [u.items for u in units] == [["table_0"], ["table_1"], ["table_2"]]
    assert units[0].text == "2015 | 2014"
    assert units[1].text == "net revenue | 2015: $ 5829 | 2014: $ 5735"
    assert units[2].text == "2016 | 2015: $ 1"  # empty cells left out


def test_paper_passages_pack_whole_sentences_with_overlap():
    sentences = [f"s{i} " + "w " * 9 for i in range(10)]  # 10 words each
    units = paper_passages("p", sentences, words, size=30, overlap=10)
    assert [u.items for u in units][:2] == [["text_0", "text_1", "text_2"],
                                            ["text_2", "text_3", "text_4"]]  # one carried over
    held = {i for u in units for i in u.items}
    assert held == {f"text_{i}" for i in range(10)}
    assert all(words(u.text) <= 30 for u in units)


def test_our_chunks_map_sentences_exactly_and_long_ones_by_half():
    short = [f"sentence {i} about revenue in year {2000 + i}." for i in range(12)]
    long_one = "a very long sentence " * 30   # 630 characters: longer than a 400 chunk
    units = our_chunks("p", short[:6] + [long_one] + short[6:], size=400, overlap=100)
    held = [i for u in units for i in u.items]
    assert set(held) == {f"text_{i}" for i in range(13)}  # every sentence held by some chunk
    for u in units:  # a short sentence only counts where it is whole
        for item in u.items:
            j = int(item.split("_")[1])
            if j != 6:
                assert (short[:6] + [long_one] + short[6:])[j] in u.text


def test_scoring_any_hit_item_recall_and_mrr():
    ranked = [["text_1"], ["table_3"], [], ["text_1", "table_5"]] + [[]] * 16
    s = score_question(ranked, ["table_3", "table_5"])
    assert s["any@5"] == 1.0 and s["items@5"] == 1.0
    assert score_question(ranked[:2] + [[]] * 18, ["table_3", "table_5"])["items@5"] == 0.5
    assert s["mrr@10"] == 0.5  # first gold item at rank 2
    miss = score_question([[]] * 20, ["text_9"])
    assert miss["any@20"] == 0.0 and miss["mrr@10"] == 0.0


def test_gold_items_no_unit_holds_are_reported():
    units = row_units("p", [["h"], ["a"]])
    report = coverage(units, [{"id": "p-1", "page": "p", "gold": ["table_1", "table_7", "text_-1"]}])
    assert report["missing"] == {"p-1": ["table_7"]} and report["malformed"] == {"p-1": ["text_-1"]}
