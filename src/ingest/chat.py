"""Chat with your documents: router -> retrieval -> answer -> memory (the basic RAG).

For each question:
  1. route it (router.py): what kind of question, the stand-alone search queries, filters;
  2. search what this user may read (their own files, shared ones; or only this chat's
     uploads) for each query, merge the passages and keep the best that fit the prompt;
  3. answer from those passages with citations (answer.py);
  4. save both messages; every `snapshot_every` messages, update the chat's memory snapshot
     (memory.py) in the background.
Without the language model (no key, quota spent, offline) it still works: the question is
searched as written and the passages are shown, with the reason.
"""
import threading

from ingest import memory, router
from ingest.answer import Answer, answer
from ingest.chats import Chats
from ingest.llm import LLMUnavailable, make_llm
from ingest.logs import log
from ingest.retrieve import Retriever
from ingest.users import User

MODALITY = {"table": "table", "speech": "transcript", "any": None}


def _overlaps(a, b) -> bool:
    return bool(set(a.chunk_ids) & set(b.chunk_ids))


def merge(lists: list[list]) -> list:
    """Passages from several queries, taking each query's best in turn (so a comparison gets
    both sides), skipping passages that repeat an earlier one."""
    out: list = []
    for rank in range(max((len(x) for x in lists), default=0)):
        for found in lists:
            if rank < len(found) and not any(_overlaps(found[rank], p) for p in out):
                out.append(found[rank])
    return out


class ChatEngine:
    def __init__(self, pipeline, chats: Chats, llm=None):
        self.pipeline = pipeline
        self.settings = pipeline.settings
        self.chats = chats
        self.retriever = Retriever(pipeline)
        self._llm = llm
        self.llm_error: str | None = None
        self._snapshotting: set[str] = set()
        self._lock = threading.Lock()

    @property
    def llm(self):
        """The language model, or None (with llm_error saying why)."""
        if self._llm is None:
            try:
                self._llm = make_llm(self.settings)
                self.llm_error = None
            except LLMUnavailable as exc:
                self.llm_error = str(exc)
        return self._llm

    def files(self, user: User, chat_id: str | None = None) -> dict[str, str]:
        """{file name: stored path} of what this user may read."""
        paths = self.pipeline.store.files(user.reads, chat_id=chat_id)
        return {p.replace("\\", "/").rsplit("/", 1)[-1]: p for p in paths}

    def retrieve(self, found: router.Route, user: User, chat_id: str | None,
                 files: dict[str, str]) -> list:
        s = self.settings
        source = router.file_for(found.file_hint, files) if found.file_hint else None
        filters = {"file_type": None if found.file_type == "any" else found.file_type,
                   "modality": MODALITY[found.content], "source_path": source}
        hybrid = False if found.mode == "meaning" else None

        def search(**f) -> list:
            return [self.retriever.search(q, s.answer_passages, groups=user.reads, chat_id=chat_id,
                                          hybrid=hybrid, terms=found.terms, **f)
                    for q in found.queries]
        lists = search(**filters)
        if not any(lists) and any(filters.values()):  # the router's filters found nothing
            lists = search()
        return merge(lists)[:s.answer_passages]

    def _fit(self, passages: list, used: int) -> list:
        """The passages that fit the prompt's budget after the conversation (`used` tokens)
        and room for the reply; the weakest go first."""
        room = self.settings.context_budget_tokens - used - self.settings.llm_max_output_tokens
        kept, total = [], 0
        for p in passages:
            size = memory.tokens(p.text) + 30
            if kept and total + size > room:
                break
            kept.append(p)
            total += size
        return kept

    def ask(self, chat_id: str, user: User, question: str, only_chat_files: bool = False,
            wait_for_memory: bool = False) -> dict:
        s = self.settings
        chat = self.chats.get(chat_id, owner=user.name)
        if chat is None:
            raise KeyError("No such chat")
        question = question.strip()
        recent = self.chats.messages(chat_id, after=chat["snapshot_upto"])
        convo = memory.context(chat, recent, s.history_messages, s.context_budget_tokens // 3)
        files = self.files(user, chat_id if only_chat_files else None)
        llm = self.llm
        notice = None
        if llm is None:
            found, notice = router.fallback(question), self.llm_error
        else:
            found, notice = router.route(question, llm, convo, list(files), model=s.router_model)

        passages, reply = [], None
        if found.searches:
            passages = self.retrieve(found, user, chat_id if only_chat_files else None, files)
            passages = self._fit(passages, memory.tokens(convo) + memory.tokens(question) + 600)
            if llm is not None:
                try:
                    reply = answer(question, passages, llm, convo, found.queries[0],
                                   found.needs_calculation, model=s.llm_model,
                                   think=s.calculation_thinking_budget
                                   if found.needs_calculation else None)
                except LLMUnavailable as exc:
                    notice = str(exc)
            if reply is None:
                # Why it isn't available is in `notice`, shown under the reply
                text = ("The language model isn't available, so there is no written answer. "
                        "The passages that best match your question are below." if passages else
                        "The language model isn't available, and no passage matches.")
                reply = Answer(text, [{"n": i, "citation": p.citation, "source_path": p.source_path,
                                       "page_number": p.page_number, "start_time": p.start_time,
                                       "end_time": p.end_time, "file_type": p.file_type}
                                      for i, p in enumerate(passages, 1)])
        else:
            reply = Answer(found.reply or "Could you tell me a bit more about what you're "
                                          "looking for?")

        self.chats.add(chat_id, "user", question)
        data = {"citations": reply.citations, "grounded": reply.grounded,
                "route": found.model_dump(), "notice": notice,
                "passages": [{"n": i, "citation": p.citation, "text": p.text,
                              "source_path": p.source_path, "page_number": p.page_number,
                              "start_time": p.start_time, "score": round(p.score, 3)}
                             for i, p in enumerate(passages, 1)]}
        message_id = self.chats.add(chat_id, "assistant", reply.text, data)
        if chat["title"] == "New chat":
            self.chats.rename(chat_id, question[:60] + ("…" if len(question) > 60 else ""))
        self._remember(chat_id, wait_for_memory)
        return {"id": message_id, "role": "assistant", "content": reply.text, **data}

    def _remember(self, chat_id: str, wait: bool) -> None:
        """Update the chat's snapshot once enough messages came after the last one."""
        chat = self.chats.get(chat_id)
        unsnapshotted = self.chats.messages(chat_id, after=chat["snapshot_upto"])
        if self.llm is None or not memory.due(chat, unsnapshotted, self.settings.snapshot_every):
            return
        with self._lock:
            if chat_id in self._snapshotting:
                return
            self._snapshotting.add(chat_id)

        def work():
            try:
                snapshot = memory.take_snapshot(chat, unsnapshotted, self.llm,
                                                model=self.settings.llm_model)
                if snapshot is not None:
                    self.chats.set_snapshot(chat_id, snapshot, unsnapshotted[-1]["id"])
                    log.info(f"chat {chat_id}: memory snapshot up to message "
                             f"{unsnapshotted[-1]['id']}")
            finally:
                with self._lock:
                    self._snapshotting.discard(chat_id)
        if wait:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="chat-snapshot").start()
