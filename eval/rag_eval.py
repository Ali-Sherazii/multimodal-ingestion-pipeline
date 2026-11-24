"""Answer quality: does the whole RAG answer TAT-DQA questions correctly, with citations?

For a fixed sample of TAT-DQA test questions (CC-BY-4.0; real annual-report pages, many
needing arithmetic), each question's passages come from our search inside its own report
(the situation of a user chatting with an uploaded report: run on Modal, with the pipeline's
own extraction, chunks, hybrid search and reranker), then the language model (llm_model)
answers from those passages with the pipeline's answer prompt (answer.py).

Scored against TAT-DQA's gold answers, leniently on wording but strictly on values:
  number answers (arithmetic, counts)  right if a number in the answer equals the gold one
      within 0.5% (or 0.01), the scale being free: "3,663 thousand" and "3.663 million" both
      match 3,663 thousand; percentages match with or without the % sign
  text answers (spans)                 right if every gold span appears in the answer
  cited                                the answer cites at least one passage
  evidence found                       all the gold "facts" are in the passages (the most
                                       the model could do: retrieval's share of the misses)
Answers are cached (eval/cache/rag/), so a re-run asks the model only for what is missing.

    python eval/rag_eval.py --limit 20     # a quick look
    python eval/rag_eval.py                # 200 questions
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CACHE = HERE / "cache" / "rag"
SAMPLE, SEED = 200, 7
SCALES = {"": 1, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1}


def numbers(text: str) -> list[float]:
    """Every number written in the text: 1,234.5, (12) and -12 negatives, 12% as 12."""
    found = []
    for m in re.finditer(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?", text):
        raw = m.group(0)
        negative = raw.startswith("(") and raw.endswith(")") or "-" in raw
        digits = re.sub(r"[^\d.]", "", raw)
        if digits.count(".") > 1 or not digits.strip("."):
            continue
        value = float(digits)
        found.append(-value if negative else value)
    return found


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, 0.005 * abs(b))


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9.%-]+", " ", str(text).lower()).strip()


def correct(answer_text: str, gold, answer_type: str, scale: str) -> bool:
    if isinstance(gold, (int, float)) or answer_type in ("arithmetic", "count"):
        try:
            target = float(str(gold).replace(",", ""))
        except ValueError:
            return norm(gold) in norm(answer_text)
        for value in numbers(answer_text):
            for factor in (1, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 100, 0.01):
                if close(value * factor, target) or close(abs(value) * factor, abs(target)):
                    return True
        return False
    spans = gold if isinstance(gold, list) else [gold]
    return all(norm(s) in norm(answer_text) for s in spans if str(s).strip())


# Words that carry no fact: a reworded answer ("its" for "our", a list for commas) keeps
# the facts but not the exact wording, which the strict score above requires
_FILLER = set("a an the of and or to in on for by with as at from is are was were be been "
              "our we us its it their they which that this these those such".split())


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in norm(str(text).replace("’", "'")).split()
            if w not in _FILLER and len(w) > 1}


def lenient(answer_text: str, gold, answer_type: str, scale: str) -> bool:
    """Text answers: right if each gold span's fact words (80% of them) are in the answer.
    Numbers: as strict as `correct`."""
    if correct(answer_text, gold, answer_type, scale):
        return True  # never stricter than strict
    if isinstance(gold, (int, float)) or answer_type in ("arithmetic", "count"):
        return False
    said = _words(answer_text)
    spans = [s for s in (gold if isinstance(gold, list) else [gold]) if str(s).strip()]
    return all(len(_words(s) & said) >= 0.8 * len(_words(s)) for s in spans)


def load_gold() -> list[dict]:
    from huggingface_hub import hf_hub_download

    import tatdqa_eval as tat
    data = json.loads(Path(hf_hub_download(tat.REPO, tat.GOLD, repo_type="dataset"))
                      .read_text(encoding="utf-8"))
    questions = []
    for d in data:
        for q in d["questions"]:
            questions.append({"uid": q["uid"], "question": q["question"],
                              "doc": f"{d['doc']['uid']}.pdf", "answer": q["answer"],
                              "answer_type": q.get("answer_type"), "scale": q.get("scale") or "",
                              "facts": [f for f in (q.get("facts") or []) if str(f).strip()]})
    return questions


def get_passages(sample: list[dict], values: dict) -> list[list[dict]]:
    """Each question's passages, from Modal (cached here)."""
    import hashlib

    import tatdqa_eval as tat
    key = hashlib.sha256(json.dumps([[q["uid"] for q in sample], values], sort_keys=True)
                         .encode()).hexdigest()[:16]
    saved = CACHE / f"passages-{key}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    _, names = tat.load_questions()
    records = tat.extractions(names, values)
    records_file = tat.upload_records(records)
    asked = [{"question": q["question"], "doc": q["doc"]} for q in sample]
    size = -(-len(asked) // 4)
    batches = [asked[i:i + size] for i in range(0, len(asked), size)]
    parts = tat.run_parts("passages", batches, records_file, values,
                          HERE / "cache" / "tatdqa-parts" / f"passages-{key}")
    found = [p for part in parts for p in part]
    CACHE.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return found


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Urdu etc. on a Windows console
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=SAMPLE)
    parser.add_argument("--passages-only", action="store_true",
                        help="only fetch the passages (Modal), no language model")
    args = parser.parse_args()

    from ingest import gpu_modal
    from ingest.answer import answer
    from ingest.config import load_settings
    from ingest.llm import LLMUnavailable, make_llm, model_label
    from ingest.retrieve import Passage

    settings = load_settings(HERE.parent / "config.yaml", qdrant_url=None, qdrant_api_key=None)
    values = gpu_modal._settings_out(settings) | settings.model_dump(mode="json", include={
        "embedding_model", "embedding_query_prompt", "embedding_batch_size", "hybrid_search",
        "reranker_model", "rerank_candidates", "exact_match", "passage_context_chars",
        "table_chunk_tokens", "chunk_size", "chunk_overlap", "model_revisions",
        "video_context_seconds", "answer_passages"})
    everything = load_gold()
    random.Random(SEED).shuffle(everything)
    sample = everything[:args.limit]
    found = get_passages(sample, values)
    if args.passages_only:
        print(f"passages ready for {len(found)} questions")
        return 0
    try:
        llm = make_llm(settings)
    except LLMUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    CACHE.mkdir(parents=True, exist_ok=True)
    # One file per answer prompt: a changed prompt asks again (the old answers stay)
    import hashlib
    from ingest.answer import SYSTEM
    version = hashlib.sha256(SYSTEM.encode()).hexdigest()[:8]
    label = model_label(settings)  # Gemma on Modal and each Gemini model: own answers
    answers_file = CACHE / f"answers-{re.sub(r'[^\w.-]+', '-', label).strip('-')}-{version}.jsonl"
    answered = {}
    if answers_file.exists():
        for line in answers_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answered[row["uid"]] = row
    rows = []
    with answers_file.open("a", encoding="utf-8") as out:
        for i, (q, passages) in enumerate(zip(sample, found), 1):
            if q["uid"] not in answered:
                objects = [Passage(**p) for p in passages]
                try:
                    reply = answer(q["question"], objects, llm, model=settings.llm_model)
                except LLMUnavailable as exc:
                    print(f"stopped at question {i}: {exc} (answers so far are saved)")
                    break
                answered[q["uid"]] = {"uid": q["uid"], "text": reply.text,
                                      "cited": bool(reply.citations),
                                      "model": getattr(llm, "last_model", None)}
                out.write(json.dumps(answered[q["uid"]], ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(sample)}] {q['answer_type']}: "
                      f"{'right' if correct(reply.text, q['answer'], q['answer_type'], q['scale']) else 'wrong'}",
                      flush=True)
            a = answered[q["uid"]]
            evidence = " ".join(p["text"] for p in passages)
            rows.append({**q, "reply": a["text"], "cited": a["cited"], "model": a.get("model"),
                         "right": correct(a["text"], q["answer"], q["answer_type"], q["scale"]),
                         "lenient": lenient(a["text"], q["answer"], q["answer_type"], q["scale"]),
                         "evidence": bool(q["facts"]) and all(
                             norm(f) in norm(evidence) for f in q["facts"])})

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["answer_type"]].append(r)
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else 0.0
    lines = ["| Questions | n | right (strict) | right (lenient) | cites a passage | "
             "evidence in passages |",
             "|---|---|---|---|---|---|",
             f"| all | {len(rows)} | {pct([r['right'] for r in rows])} | "
             f"{pct([r['lenient'] for r in rows])} | "
             f"{pct([r['cited'] for r in rows])} | {pct([r['evidence'] for r in rows])} |"]
    for kind, rs in sorted(by_type.items()):
        lines.append(f"| {kind} | {len(rs)} | {pct([r['right'] for r in rs])} | "
                     f"{pct([r['lenient'] for r in rs])} | "
                     f"{pct([r['cited'] for r in rs])} | {pct([r['evidence'] for r in rs])} |")
    with_evidence = [r for r in rows if r["evidence"]]
    lines.append(f"\nWhen the evidence was in the passages ({len(with_evidence)} questions), "
                 f"the answer was right {pct([r['right'] for r in with_evidence])}% of the time "
                 f"(lenient: {pct([r['lenient'] for r in with_evidence])}%).\n\nStrict: text "
                 "answers must contain the gold wording exactly; lenient: 80% of each gold span's "
                 "fact words (numbers are scored the same way in both).")
    wrong = [r for r in rows if not r["lenient"]][:12]
    examples = "\n".join(f"- {r['question']} (gold: {r['answer']} {r['scale']}) → "
                         f"{r['reply'][:220].replace(chr(10), ' ')}" for r in wrong)
    models = defaultdict(int)  # a busy model hands over to llm_fallback_models
    for r in rows:
        models[r["model"] or "unknown"] += 1
    by = ", ".join(f"{m} {n}" for m, n in sorted(models.items(), key=lambda x: -x[1]))
    table = "\n".join(lines)
    print(f"\n{table}\nanswered by: {by}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    report = HERE / "results" / f"rag-{stamp}.md"
    report.write_text(f"# Answer quality on TAT-DQA {stamp}\n\n{len(rows)} questions (seed {SEED}), "
                      f"passages from each question's own report (top {settings.answer_passages}, "
                      f"with context); answers by {by} (configured: {label}).\n\n{table}\n\n"
                      f"## Some wrong answers\n\n{examples}\n", encoding="utf-8")
    (report.with_suffix(".json")).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"saved {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
