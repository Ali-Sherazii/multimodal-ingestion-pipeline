"""The question router (the supervisor's step 5): reads each question before any search,
improves it and picks the route.

One quick call to a small model (router_model) fills in a form (Route):
  kind      search (needs the documents), follow_up (refers to the conversation: resolved
            into a stand-alone question first), chit_chat (no search), out_of_scope (not
            about the documents), clarify (too vague: ask back)
  queries   1-3 stand-alone search queries: pronouns and "that table" resolved from the chat,
            abbreviations spelled out, a comparison split into one query per side
  filters   file type, content type (tables / speech / text), a file the question names
  mode      hybrid (meaning + keywords, the default) or meaning only (a paraphrase-heavy
            question), plus words to find exactly as written (ids, figures, names)
If the model can't be reached, the question goes to a normal search unchanged.
"""
from typing import Literal

from pydantic import BaseModel, Field

from ingest.llm import LLMUnavailable
from ingest.logs import log

SYSTEM = """You route questions in a chat about the user's own documents (reports, PDFs, Word
files, videos). You never answer the question itself. Fill in the form:

kind:
- "search": needs information from the documents.
- "follow_up": refers to earlier messages ("and Lahore?", "what about 2019?", "explain that").
  Still a search: write the stand-alone question in queries.
- "chit_chat": greetings, thanks, or questions about the conversation itself ("summarise what
  you said") that need no document search. Put a short, friendly reply in `reply`.
- "out_of_scope": clearly unrelated to the documents (weather, sports scores, coding help).
  Put a polite one-sentence reply in `reply` saying you answer from the user's documents.
- "clarify": too vague to search even with the conversation ("what's the total?" with no
  context). Put one short clarifying question in `reply`. A question naming something
  specific (a table, a figure, a person, an id) is a search even if it names no file: all
  the user's files are searched.

queries (search / follow_up): 1 to 3 stand-alone search queries. Resolve pronouns and
references from the conversation; spell out abbreviations; keep ids, figures, names and years
exactly as written; for a comparison of N things write one query per thing. Otherwise [].
file_type: "pdf", "docx", "text", "video", or "any". Only when the question says so.
content: "table" (asks for figures from a table), "speech" (what is said in a video), or "any".
file_hint: words from a file name the question mentions (see the user's files), else "".
mode: "hybrid" normally; "meaning" only for vague, conceptual questions with no keywords.
terms: ids, codes, figures or exact names the answer must contain as written, else [].
needs_calculation: true if the answer needs arithmetic (totals, differences, percentages).
reply: only for chit_chat / out_of_scope / clarify, else "".
Write the reply in the language of the question."""


class Route(BaseModel):
    kind: Literal["search", "follow_up", "chit_chat", "out_of_scope", "clarify"]
    queries: list[str] = Field(description="stand-alone search queries")
    file_type: Literal["any", "pdf", "docx", "text", "video"]
    content: Literal["any", "table", "speech"]
    file_hint: str
    mode: Literal["hybrid", "meaning"]
    terms: list[str]
    needs_calculation: bool
    reply: str

    @property
    def searches(self) -> bool:
        return self.kind in ("search", "follow_up")


def fallback(question: str) -> Route:
    """The route when the model can't be asked: a normal search for the question as written."""
    return Route(kind="search", queries=[question], file_type="any", content="any",
                 file_hint="", mode="hybrid", terms=[], needs_calculation=False, reply="")


def route(question: str, llm, context: str = "", files: list[str] = (),
          model: str | None = None) -> tuple[Route, str | None]:
    """The route for this question, and why the fallback was used (None if it wasn't)."""
    shown = "\n".join(f"- {name}" for name in list(files)[:60]) or "(none listed)"
    prompt = (f"The user's files:\n{shown}\n\nThe conversation so far:\n{context or '(none)'}"
              f"\n\nThe new question:\n{question}")
    try:
        # No thinking: sorting a question needs none (the Lite model, which doesn't think,
        # gets them right), and a thinking model can spend the whole reply on it
        found = llm.generate(prompt, system=SYSTEM, schema=Route, model=model, temperature=0.0,
                             max_tokens=600, think=0)
    except LLMUnavailable as exc:
        log.warning(f"question router unavailable, searching as written: {exc}")
        return fallback(question), str(exc)
    if found.searches:
        queries = [q.strip() for q in found.queries if q.strip()][:3]
        found.queries = queries or [question]
    return found, None


def file_for(hint: str, files: dict[str, str]) -> str | None:
    """The stored path of the file a hint names ({name: path}): all hint words in its name."""
    words = [w for w in hint.lower().replace("_", " ").replace("-", " ").split() if len(w) > 1]
    if not words:
        return None
    matches = [path for name, path in files.items()
               if all(w in name.lower().replace("_", " ").replace("-", " ") for w in words)]
    return matches[0] if len(matches) == 1 else None
