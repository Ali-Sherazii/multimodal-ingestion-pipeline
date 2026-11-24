"""Question router check: does it read each question the way a person would?

Sixteen chat situations about the files below (the Kestrel stress-test report, a lecture
video, an HR policy), each with what the router must do: the kind of question, what the
stand-alone search queries must mention, and the filters. Runs the real router model
(router_model in config.yaml) once per situation; needs INGEST_GEMINI_API_KEY in .env.

    python eval/router_eval.py
"""
import sys
from datetime import datetime
from pathlib import Path

from ingest import router
from ingest.config import load_settings
from ingest.llm import LLMUnavailable, make_llm, model_label

HERE = Path(__file__).resolve().parent
FILES = ["kestrel_annual_report_2026.pdf", "Random Forest lecture.mp4", "HR_leave_policy.docx"]
ASKED = "User: What was Hyderabad branch revenue in Q3 FY2026?\nAssistant: PKR 3,663 thousand [1]."

# (name, conversation so far, question, expected): expected keys:
#   kind; any: every group must have a word in some query (groups are alternatives);
#   split: at least this many queries; file_type / content / hint (a word of file_hint);
#   terms: in `terms`; calc: needs_calculation; reply: a reply must be written
CASES = [
    ("plain search", "", "What was Hyderabad branch revenue in Q3 FY2026?",
     {"kind": "search", "any": [["hyderabad"], ["revenue"]]}),
    ("follow-up: another branch", ASKED, "And Lahore?",
     {"kind": "follow_up", "any": [["lahore"], ["revenue"], ["q3", "third quarter"]]}),
    ("follow-up: pronoun", "User: Show the headcount in Table 8.\nAssistant: Table 8 lists "
     "headcount at year end by department [1].", "Which footnotes apply to it?",
     {"kind": "follow_up", "any": [["table 8", "headcount"], ["footnote"]]}),
    ("comparison splits", "", "Compare the Karachi Port and Lahore Central depot bays",
     {"kind": "search", "split": 2, "any": [["karachi"], ["lahore"]]}),
    ("thanks", ASKED, "thanks, that's really helpful!", {"kind": "chit_chat", "reply": True}),
    ("about the chat", ASKED, "Summarise what you told me so far",
     {"kind": "chit_chat", "reply": True}),
    ("off topic", "", "What's the weather in Karachi tomorrow?",
     {"kind": "out_of_scope", "reply": True}),
    ("too vague", "", "What's the total?", {"kind": "clarify", "reply": True}),
    ("names a file", "", "In the HR policy, how many leave days do staff get?",
     {"kind": "search", "hint": ["hr", "leave", "policy"], "any": [["leave"]]}),
    ("video speech", "", "What does the lecturer say about bagging in the random forest video?",
     {"kind": "search", "content": "speech", "any": [["bagging"]]}),
    ("exact id", "", "Who approved request AP-317?",
     {"kind": "search", "terms": "AP-317", "any": [["ap-317"]]}),
    ("needs arithmetic", "", "By how much did Punjab's revenue grow from H1 to H2 FY2026?",
     {"kind": "search", "calc": True, "any": [["punjab"], ["revenue"]]}),
    ("abbreviations", "", "rev for KHI port depot in fy26?",
     {"kind": "search", "any": [["revenue"], ["karachi", "khi"]]}),
    ("in Urdu", "", "حیدرآباد برانچ کی تیسری سہ ماہی کی آمدنی کیا تھی؟",
     {"kind": "search", "any": [["hyderabad", "حیدرآباد"]]}),
    ("asks for a table", "", "Show me the budget versus actual table",
     {"kind": "search", "content": "table", "any": [["budget"]]}),
    ("follow-up from memory", "Memory of the earlier conversation:\nSummary: the user is "
     "reviewing Table 13 (monthly RAG status by project) in the Kestrel report.\n\nLatest "
     "messages:\nUser: ok\nAssistant: Anything else?", "Which projects were red in March?",
     {"kind": "follow_up", "any": [["red"], ["table 13", "rag", "project"]]}),
]


def check(found: router.Route, want: dict) -> list[str]:
    """What the route got wrong (empty: right)."""
    wrong = []
    if found.kind != want["kind"] and not (want["kind"] == "follow_up" and found.kind == "search"
                                           and want.get("any")):
        wrong.append(f"kind {found.kind}, wanted {want['kind']}")
    text = " ".join(found.queries).lower()
    for group in want.get("any", []):
        if not any(word in text for word in group):
            wrong.append(f"queries lack {'/'.join(group)}")
    if len(found.queries) < want.get("split", 0):
        wrong.append(f"{len(found.queries)} queries, wanted {want['split']}+")
    for key, field in (("file_type", "file_type"), ("content", "content")):
        if key in want and getattr(found, field) != want[key]:
            wrong.append(f"{field} {getattr(found, field)}, wanted {want[key]}")
    if "hint" in want and not any(w in found.file_hint.lower() for w in want["hint"]):
        wrong.append(f"file_hint {found.file_hint!r}")
    if "terms" in want and want["terms"].lower() not in " ".join(found.terms).lower():
        wrong.append(f"terms {found.terms}")
    if "calc" in want and found.needs_calculation != want["calc"]:
        wrong.append("needs_calculation")
    if want.get("reply") and not found.reply.strip():
        wrong.append("no reply written")
    return wrong


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Urdu etc. on a Windows console
    settings = load_settings(HERE.parent / "config.yaml")
    try:
        llm = make_llm(settings)
    except LLMUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    rows, right = [], 0
    for name, convo, question, want in CASES:
        found, problem = router.route(question, llm, convo, FILES, model=settings.router_model)
        wrong = [f"model unavailable: {problem}"] if problem else check(found, want)
        right += not wrong
        detail = "; ".join(wrong) or "ok"
        rows.append(f"| {name} | {found.kind} | {' · '.join(found.queries) or '-'} | {detail} |")
        print(f"{'ok  ' if not wrong else 'MISS'} {name}: {found.kind} {found.queries} {detail}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    table = "\n".join(["| Situation | Kind | Search queries | Result |", "|---|---|---|---|", *rows])
    out = HERE / "results" / f"router-{stamp}.md"
    # who really answered: a busy model hands over to another (or Gemma to Gemini)
    by = ", ".join(f"{m} {n}" for m, n in llm.usage.by_model.items()) or "no model"
    out.write_text(f"# Question router check {stamp}\n\nModel: {model_label(settings, 'router')} "
                   f"(answered by: {by}). {right} of {len(CASES)} situations handled right."
                   f"\n\n{table}\n", encoding="utf-8")
    print(f"\n{right} of {len(CASES)} right (answered by: {by}); saved {out}")
    return 0 if right == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
