"""Chat memory (the supervisor's step 3): snapshots and the context window.

Sending a whole conversation to the model every turn gets slow, costly and eventually too
long. Instead, every `snapshot_every` messages the conversation so far becomes a snapshot:
not a plain summary but a structured record that keeps what later questions need:

  summary         what the conversation is about and where it stands
  user_goal       what the user is trying to find out or do
  entities        companies, people, tables, files... each with the facts learned about it
  relationships   how they relate ("Kestrel" - "owns" - "Karachi Port depot")
  key_facts       figures and findings, each with its source (file, page)
  open_questions  what is still unanswered
  documents       the files discussed

Each new snapshot merges the previous one with the messages since, so nothing is lost when
the old messages leave the prompt. A prompt is then: the snapshot, the messages after it
(word for word, at most `history_messages`), and the passages found for the question, kept
within `context_budget_tokens` (oldest messages dropped first, then the weakest passages).
"""
import json

from pydantic import BaseModel

from ingest.llm import LLMUnavailable
from ingest.logs import log


class Entity(BaseModel):
    name: str
    kind: str
    facts: list[str]


class Relation(BaseModel):
    subject: str
    relation: str
    object: str


class Snapshot(BaseModel):
    summary: str
    user_goal: str
    entities: list[Entity]
    relationships: list[Relation]
    key_facts: list[str]
    open_questions: list[str]
    documents: list[str]


SYSTEM = """You keep the memory of a conversation between a user and an assistant that answers
from the user's documents. Update the snapshot with the new messages. Keep everything a later
question may need: the user's goal and preferences; every entity discussed (companies, people,
products, branches, tables, files) with the facts learned about it, figures written exactly,
with units and years; relationships between entities; key facts with their source (file and
page or time) as cited in the messages; questions still open; files discussed. Merge with the
previous snapshot: keep what still matters, update what changed, drop small talk. Be compact
(under about 500 words in total). Write in the language of the conversation."""


def tokens(text: str) -> int:
    """A token estimate (about 4 characters each in English) without calling the API."""
    return len(text) // 4 + 1


def render(snapshot: dict | None) -> str:
    """A snapshot as compact text for a prompt."""
    if not snapshot:
        return ""
    s = snapshot
    lines = [f"Summary: {s.get('summary', '')}", f"User's goal: {s.get('user_goal', '')}"]
    for e in s.get("entities") or []:
        lines.append(f"- {e['name']} ({e['kind']}): " + "; ".join(e.get("facts") or []))
    for r in s.get("relationships") or []:
        lines.append(f"- {r['subject']} — {r['relation']} — {r['object']}")
    if s.get("key_facts"):
        lines.append("Key facts: " + " | ".join(s["key_facts"]))
    if s.get("open_questions"):
        lines.append("Open questions: " + " | ".join(s["open_questions"]))
    if s.get("documents"):
        lines.append("Files discussed: " + ", ".join(s["documents"]))
    return "\n".join(lines)


def transcript(messages: list[dict], limit: int = 1500) -> str:
    """Messages as 'User: ...' / 'Assistant: ...' lines, each cut to `limit` characters."""
    out = []
    for m in messages:
        text = m["content"] if len(m["content"]) <= limit else m["content"][:limit] + "…"
        out.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {text}")
    return "\n".join(out)


def context(chat: dict, recent: list[dict], history_messages: int, budget: int) -> str:
    """The conversation for a prompt: the snapshot, then the latest messages after it,
    oldest dropped first until it fits `budget` tokens."""
    memory = render(chat.get("snapshot"))
    recent = recent[-history_messages:] if history_messages else []
    while recent and tokens(memory) + tokens(transcript(recent)) > budget:
        recent = recent[1:]
    parts = [f"Memory of the earlier conversation:\n{memory}"] if memory else []
    if recent:
        parts.append(f"Latest messages:\n{transcript(recent)}")
    return "\n\n".join(parts)


def due(chat: dict, unsnapshotted: list[dict], every: int) -> bool:
    """A new snapshot is due once `every` messages have come after the last one."""
    return every > 0 and len(unsnapshotted) >= every


def take_snapshot(chat: dict, messages: list[dict], llm, model: str | None = None) -> dict | None:
    """The previous snapshot merged with these messages; None if the model can't be asked
    (the messages then stay word for word until the next try)."""
    previous = json.dumps(chat.get("snapshot") or {}, ensure_ascii=False)
    prompt = (f"Previous snapshot (JSON, may be empty):\n{previous}\n\n"
              f"New messages:\n{transcript(messages, limit=4000)}")
    try:
        found = llm.generate(prompt, system=SYSTEM, schema=Snapshot, model=model,
                             temperature=0.0, max_tokens=2000)
    except LLMUnavailable as exc:
        log.warning(f"chat snapshot skipped for now: {exc}")
        return None
    return found.model_dump()
