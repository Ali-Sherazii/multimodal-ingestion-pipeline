"""The language model: Gemini (Google's google-genai library) or our own copy of Gemma 4 on a
Modal GPU (llm_backend: modal; the server is llm_modal.py).

One small interface (generate: text or a filled-in form; chat_step: one turn with tools) so
the rest of the code doesn't depend on the provider, and tests use FakeLLM. Gemini's free tier
has low per-minute and per-day limits (per model), and its models are often too busy for
free users: a busy, rate-limited or spent model is left alone for a while and the next of
llm_fallback_models answers instead. With llm_backend modal, Gemma answers and Gemini (if a
key is set) steps in while it can't. When nothing can answer, LLMUnavailable carries a plain
message, and callers fall back to showing the passages themselves.

Note: on Gemini's free tier, Google may use what is sent (questions and passages) to improve
its products. Use a paid key, or the Modal model, for confidential documents.
"""
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel

from ingest.config import Settings
from ingest.logs import log


class LLMUnavailable(RuntimeError):
    """No key, the quota is spent, or the service can't be reached: a reason to show users."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)  # replies from each model

    def add(self, meta, model: str | None = None) -> None:
        """One Gemini reply (its usage_metadata)."""
        self.count(model, (meta.prompt_token_count or 0) if meta else 0,
                   ((meta.candidates_token_count or 0) +
                    (getattr(meta, "thoughts_token_count", 0) or 0)) if meta else 0)

    def count(self, model: str | None, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.calls += 1
        if model:
            self.by_model[model] = self.by_model.get(model, 0) + 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class Step:
    """One model turn in a conversation with tools: text, tool calls, or both."""
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    raw: object = None  # the model's own message, to send back with the tool results
    finish: str = ""    # why the model stopped: STOP, MAX_TOKENS, MALFORMED_FUNCTION_CALL...


def _cut_short(response) -> bool:
    """The reply stopped at the token limit with most tokens spent thinking, not replying."""
    candidate = response.candidates[0] if response.candidates else None
    meta = response.usage_metadata
    reason = getattr(getattr(candidate, "finish_reason", None), "name", None)
    return bool(reason == "MAX_TOKENS" and meta and
                (meta.thoughts_token_count or 0) > (meta.candidates_token_count or 0))


def _wait_from(exc) -> float | None:
    """The delay a 429 asks for ("retryDelay": "17s"), if it gives one."""
    match = re.search(r"retry(?:Delay| in)['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


class Gemini:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMUnavailable("No Gemini API key: add INGEST_GEMINI_API_KEY=... to the .env "
                                 "file (a free key: https://aistudio.google.com/apikey)")
        from google import genai
        from google.genai import types
        self.settings = settings
        self.client = genai.Client(  # a time limit: a stalled request must not hang the app
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)))
        self.usage = Usage()
        self.last_model: str | None = None   # the model that gave the last reply
        self._no_thinking: set[str] = set()  # models that refuse a thinking setting
        self._rest: dict[str, float] = {}    # model -> when to try it again (time.monotonic)
        self._fails: dict[str, int] = {}     # model -> busy/silent replies in a row

    def _config(self, model: str, system: str | None, schema, temperature, max_tokens,
                tools=None, think: int | None = None):
        from google.genai import types
        s = self.settings
        budget = s.llm_thinking_budget if think is None else think
        thinking = None
        if budget is not None and model not in self._no_thinking:
            thinking = types.ThinkingConfig(thinking_budget=budget)
        return types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=s.llm_temperature if temperature is None else temperature,
            top_p=s.llm_top_p, max_output_tokens=max_tokens or s.llm_max_output_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=schema, thinking_config=thinking, tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            if tools else None)

    def _try(self, model: str, contents, config_args: tuple):
        from google.genai import errors
        config = self._config(model, *config_args)
        try:
            return self.client.models.generate_content(model=model, contents=contents,
                                                       config=config)
        except errors.APIError as exc:
            # Some models take no thinking setting, or not this one (the Lite ones refuse 0),
            # and may only say "invalid argument": again without it
            if exc.code == 400 and config.thinking_config is not None:
                self._no_thinking.add(model)
                log.info(f"Gemini {model} refused the thinking setting; sending none")
                return self._try(model, contents, config_args)
            raise

    def _failed(self, model: str, exc: Exception) -> tuple[str, float]:
        """Why this model failed and how long to leave it alone; raises LLMUnavailable when
        no other model or later try can help (a bad key, a malformed request)."""
        from google.genai import errors
        if not isinstance(exc, errors.APIError):
            if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or \
                    type(exc).__module__.startswith(("httpx", "httpcore")):
                return self._busy(model, f"no reply ({type(exc).__name__})")
            raise exc
        text = str(exc)
        if exc.code in (401, 403) or "API_KEY_INVALID" in text:
            raise LLMUnavailable("Gemini rejected the API key: check "
                                 "INGEST_GEMINI_API_KEY in .env") from None
        if exc.code == 404:
            return (f"Gemini has no model '{model}' for this key: change llm_model, "
                    "router_model or llm_fallback_models in config.yaml"), 6 * 3600
        if exc.code == 429 and re.search(r"PerDay|per day|daily", text, re.I):
            return ("The Gemini free quota for today is used up; it resets at midnight "
                    "Pacific time"), 3600
        if exc.code == 429:
            return f"rate limit ({model})", min(_wait_from(exc) or 10, 65)
        if (exc.code or 0) >= 500:
            return self._busy(model, f"busy ({exc.code})")
        raise LLMUnavailable(f"Gemini refused the request ({exc.code}: "
                             f"{(exc.message or text)[:200]})") from None

    def _busy(self, model: str, why: str) -> tuple[str, float]:
        """A busy or silent model rests longer each time in a row: 30 s, 1 min ... 10 min
        (seen live: a busy model can take 40 s to say so, on every question)."""
        self._fails[model] = self._fails.get(model, 0) + 1
        return f"Gemini {model} {why}", min(30 * 2 ** (self._fails[model] - 1), 600)

    def _call(self, model: str, contents, config_args: tuple):
        """generate_content on `model`, or on the next of llm_fallback_models while it is
        busy, rate-limited or out of today's quota (free-tier capacity varies by the hour).
        A model that failed is left alone for a while, so later calls don't wait on it."""
        chain = list(dict.fromkeys([model, *self.settings.llm_fallback_models]))
        problem, grown = None, False
        for _ in range(max(1, self.settings.llm_retries) * len(chain) + 1):
            now = time.monotonic()
            ready = [m for m in chain if self._rest.get(m, 0) <= now]
            if not ready:
                busy = [m for m in chain if self._fails.get(m)]
                wait = min(self._rest[m] for m in chain) - now
                if wait > 65 and not busy:
                    break  # out of today's quota, or no such model: waiting won't help
                wait = min(wait, 65)  # all busy: try them again after at most a minute
                log.warning(f"{problem or 'Gemini busy'}; trying again in {wait:.0f} s")
                time.sleep(wait)
                for m in chain:
                    if self._rest[m] <= now + wait or m in busy:
                        self._rest[m] = 0
                continue
            name = ready[0]
            try:
                response = self._try(name, contents, config_args)
            except Exception as exc:
                problem, rest = self._failed(name, exc)
                self._rest[name] = time.monotonic() + rest
                after = next((m for m in chain if self._rest.get(m, 0) <= time.monotonic()), None)
                if after:
                    log.warning(f"{problem}; using {after}")
                continue
            self._fails.pop(name, None)
            self.usage.add(response.usage_metadata, name)
            if _cut_short(response) and not grown:
                # Its thinking used up the tokens meant for the reply (newer models count
                # both): once more, with room for both
                grown = True
                limit = config_args[3] or self.settings.llm_max_output_tokens
                config_args = (*config_args[:3], limit * 2 + 4096, *config_args[4:])
                log.info(f"Gemini {name} ran out of tokens while thinking; asking again with "
                         f"{config_args[3]}")
                continue
            self.last_model = name
            return response
        raise LLMUnavailable(problem or "Gemini gave no answer")

    def generate(self, prompt, system: str | None = None, schema: type[BaseModel] | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, think: int | None = None):
        """The model's reply: text, or (with `schema`) an instance of that pydantic model.
        `think`: this call's thinking budget in tokens (0 = none; None = llm_thinking_budget)."""
        model = model or self.settings.llm_model
        response = self._call(model, prompt, (system, schema, temperature, max_tokens, None,
                                              think))
        if schema is not None:
            if response.parsed is None:
                raise LLMUnavailable("Gemini's reply didn't fit the requested form")
            return response.parsed
        return (response.text or "").strip()

    def chat_step(self, history: list, system: str | None, tools: list[dict],
                  model: str | None = None, temperature: float | None = None) -> Step:
        """One turn of a tool-using conversation. `history` holds google-genai Contents
        (from user_message / tool_results / the previous Step's raw)."""
        from google.genai import types
        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=t["parameters"])
                        for t in tools]
        model = model or self.settings.llm_model
        # Agents write whole files into their tool calls: far longer than a chat answer
        response = self._call(model, history, (system, None, temperature,
                                               self.settings.agent_max_output_tokens,
                                               [types.Tool(function_declarations=declarations)]))
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        parts = (content.parts if content else None) or []
        calls = [ToolCall(p.function_call.name, dict(p.function_call.args or {}),
                          getattr(p.function_call, "id", None))
                 for p in parts if p.function_call]
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
        finish = getattr(getattr(candidate, "finish_reason", None), "name", "") or ""
        return Step(text.strip(), calls, content, finish)

    @staticmethod
    def user_message(text: str):
        from google.genai import types
        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    @staticmethod
    def tool_results(results: list[tuple[ToolCall, dict]]):
        from google.genai import types
        return types.Content(role="user", parts=[
            types.Part.from_function_response(name=call.name, response=result)
            for call, result in results])

    def models(self) -> list[str]:
        """The model names this key may use for text (for `ingest llm`)."""
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list()
                      if "generateContent" in (getattr(m, "supported_actions", None) or []))


def _flat(history: list) -> list[dict]:
    """An agent's memory as OpenAI-format messages (tool results are a list per turn); keys
    starting with "_" are ours and aren't sent."""
    return [{k: v for k, v in m.items() if not k.startswith("_")}
            for item in history for m in (item if isinstance(item, list) else [item])]


def _to_gemini(history: list) -> list:
    """An OpenAI-format conversation as google-genai Contents, so Gemini can take over. Turns
    Gemini made itself go back as they were (with its thought signatures, which it checks);
    another model's tool calls and their results go as plain text, which it accepts."""
    from google.genai import types
    contents: list = []
    gemini_turn, results = False, None   # results: the Content collecting tool results
    for item in history:
        for m in (item if isinstance(item, list) else [item]):
            role = m.get("role")
            if role == "tool":
                name = m.get("_name") or "the tool"
                part = (types.Part.from_function_response(name=name,
                                                          response={"result": m.get("content")})
                        if gemini_turn else
                        types.Part.from_text(text=f"Result of {name}:\n{m.get('content')}"))
                if results is None:
                    results = types.Content(role="user", parts=[])
                    contents.append(results)
                results.parts.append(part)
                continue
            results = None
            if role == "assistant":
                gemini_turn = m.get("_gemini") is not None
                if gemini_turn:
                    contents.append(m["_gemini"])
                    continue
                said = [m.get("content") or ""] + [
                    f"(called {c['function']['name']} with {c['function']['arguments']})"
                    for c in m.get("tool_calls") or []]
                contents.append(types.Content(role="model", parts=[
                    types.Part.from_text(text="\n".join(s for s in said if s) or "(no reply)")]))
            else:
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_text(text=str(m.get("content") or ""))]))
    return contents


def form_instruction(system: str | None, schema) -> str:
    """The system message with the form spelled out. vLLM holds the reply to the JSON schema
    but never shows it to the model, and the server allows no spaces between JSON parts (Gemma
    padded forms with them), so the model is told both. Without this Gemma filled the chat
    memory form's first fields with ": " (seen 2026-11-21); the router form only worked
    because its prompt happens to describe every field."""
    form = json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    return ((system + "\n\n") if system else "") + (
        "Reply with only the JSON object, written compactly on one line with no spaces or "
        f"line breaks between its parts, that fits this JSON schema:\n{form}")


class ModalLLM:
    """Our own copy of an open model (Gemma 4) on a Modal GPU, served by vLLM (llm_modal.py):
    no daily limit, and what is sent stays in your Modal account. The first request after a
    quiet spell waits while the GPU starts (~2.5 minutes). Requests use the OpenAI chat
    format, and so does an agent's memory, so Gemini (`backup`, when a key is set) can answer
    any call, or take over a conversation, while this model can't."""

    REST_SECONDS = 120  # after a failure, the backup answers for this long before we try again

    def __init__(self, settings: Settings, backup: "Gemini | None" = None, server=None):
        from ingest import llm_modal
        self.settings = settings
        spec = llm_modal.spec_for_app(settings.modal_llm_app)
        self.name, self.max_context = spec.served, spec.context
        if server is None:  # nothing is sent until the first request
            import modal
            server = modal.Cls.from_name(settings.modal_llm_app, llm_modal.CLASS_NAME)()
        self.server = server
        self.backup = backup
        self.usage = Usage()
        if backup is not None:
            backup.usage = self.usage  # one count of calls, whoever answered
        self.last_model: str | None = None
        self._rest_until = 0.0
        self._problem = ""

    # --- talking to the server
    def _down(self, problem: str) -> LLMUnavailable:
        self._problem, self._rest_until = problem, time.monotonic() + self.REST_SECONDS
        log.warning(problem + ("; Gemini answers meanwhile" if self.backup else ""))
        return LLMUnavailable(problem)

    def _send(self, body: dict) -> dict:
        """vLLM's chat completion; LLMUnavailable (with a plain reason) when there is none."""
        if time.monotonic() < self._rest_until:
            raise LLMUnavailable(self._problem)
        try:
            reply = self.server.chat.remote(body)
        except Exception as exc:
            kind = type(exc).__name__
            if kind == "NotFoundError":
                problem = (f"the model app '{self.settings.modal_llm_app}' isn't deployed on "
                           "Modal: run `modal deploy src/ingest/llm_modal.py`")
            elif kind == "AuthError":
                problem = "Modal refused the login: run `modal token new`"
            else:
                problem = f"the model on Modal didn't answer ({kind}: {str(exc)[:200]})"
            raise self._down(problem) from None
        status, data = reply.get("status"), reply.get("body") or {}
        if status == 200:
            return data
        error = data.get("error") if isinstance(data.get("error"), dict) else data
        message = str(error.get("message") or data)[:300]
        if (status or 500) >= 500:
            raise self._down(f"the model on Modal failed ({status}: {message})")
        raise LLMUnavailable(f"the model on Modal refused the request ({status}: {message})")

    def _counted(self, data: dict) -> dict:
        usage = data.get("usage") or {}
        self.usage.count(self.name, usage.get("prompt_tokens") or 0,
                         usage.get("completion_tokens") or 0)
        self.last_model = self.name
        return data["choices"][0]

    def _thinking(self, think: int | None) -> tuple[bool, int]:
        """Gemma thinks only when asked to: this call's `think`, else llm_thinking_budget."""
        budget = self.settings.llm_thinking_budget if think is None else think
        return bool(budget and budget > 0), max(budget or 0, 0)

    def _room(self, messages: list, tools: list | None, wanted: int) -> int:
        """The reply's token limit, kept inside the context window (prompt + reply); a
        rough count: about 3 characters per token for prose, tables and JSON."""
        used = len(json.dumps([messages, tools or []], ensure_ascii=False)) // 3 + 64
        return max(256, min(wanted, self.max_context - used))

    def _sampling(self, temperature: float | None, thinking: bool) -> dict:
        s = self.settings
        return {"temperature": s.llm_temperature if temperature is None else temperature,
                "top_p": s.llm_top_p, "chat_template_kwargs": {"enable_thinking": thinking}}

    # --- the interface
    def generate(self, prompt, system: str | None = None, schema: type[BaseModel] | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, think: int | None = None):
        """As Gemini.generate. `model` is the Gemini model the backup would use."""
        try:
            return self._generate(prompt, system, schema, temperature, max_tokens, think)
        except LLMUnavailable as exc:
            if self.backup is None:
                raise
            if time.monotonic() >= self._rest_until:  # not logged already by _down
                log.info(f"{exc}; Gemini answers this one")
            result = self.backup.generate(prompt, system, schema, model, temperature,
                                          max_tokens, think)
            self.last_model = self.backup.last_model
            return result

    def _generate(self, prompt, system, schema, temperature, max_tokens, think):
        text = prompt if isinstance(prompt, str) else "\n\n".join(map(str, prompt))
        if schema is not None:
            system = form_instruction(system, schema)
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": text}]
        thinking, budget = self._thinking(think)
        limit = (max_tokens or self.settings.llm_max_output_tokens) + budget
        body = {"messages": messages, **self._sampling(temperature, thinking)}
        if schema is not None:  # vLLM keeps the reply to this JSON schema as it writes it
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}
        for attempt in range(2):
            body["max_tokens"] = self._room(messages, None, limit)
            choice = self._counted(self._send(body))
            content = (choice["message"].get("content") or "").strip()
            if choice.get("finish_reason") == "length" and attempt == 0 and \
                    (thinking or schema is not None):
                limit = limit * 2 + 4096   # cut off mid-thought or mid-form: once more, longer
                log.info(f"{self.name} ran out of tokens; asking again with {limit}")
                continue
            break
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except ValueError:  # pydantic's ValidationError is a ValueError
            raise LLMUnavailable(f"{self.name}'s reply didn't fit the requested form") from None

    def chat_step(self, history: list, system: str | None, tools: list[dict],
                  model: str | None = None, temperature: float | None = None) -> Step:
        """One turn of a tool-using conversation; `history` holds OpenAI-format messages
        (from user_message / tool_results / the previous Step's raw)."""
        try:
            return self._chat_step(history, system, tools, temperature)
        except LLMUnavailable as exc:
            if self.backup is None:
                raise
            if time.monotonic() >= self._rest_until:
                log.info(f"{exc}; Gemini takes this turn")
            step = self.backup.chat_step(_to_gemini(history), system, tools, model, temperature)
            self.last_model = self.backup.last_model
            calls = [ToolCall(c.name, c.args, c.id or f"call_{uuid.uuid4().hex[:12]}")
                     for c in step.calls]
            return Step(step.text, calls, self._assistant(step.text, calls, step.raw),
                        step.finish)

    @staticmethod
    def _assistant(text: str, calls: list[ToolCall], gemini=None) -> dict:
        """The model's turn as kept in memory (Gemini's own turn is kept with it)."""
        message: dict = {"role": "assistant", "content": text or ""}
        if calls:
            message["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                for c in calls]
        if gemini is not None:
            message["_gemini"] = gemini
        return message

    def _chat_step(self, history, system, tools, temperature) -> Step:
        messages = ([{"role": "system", "content": system}] if system else []) + _flat(history)
        functions = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description") or "",
            "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
            for t in tools]
        thinking, budget = self._thinking(None)
        body = {"messages": messages, "tools": functions, **self._sampling(temperature, thinking),
                # agents write whole files into their tool calls: far longer than an answer
                "max_tokens": self._room(messages, functions,
                                         self.settings.agent_max_output_tokens + budget)}
        choice = self._counted(self._send(body))
        message = choice["message"]
        calls, malformed = [], False
        for c in message.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                malformed = True
                continue
            calls.append(ToolCall(c["function"]["name"], args,
                                  c.get("id") or f"call_{uuid.uuid4().hex[:12]}"))
        reason = choice.get("finish_reason") or ""
        finish = {"stop": "STOP", "tool_calls": "STOP", "length": "MAX_TOKENS"}.get(
            reason, reason.upper())
        if malformed and not calls:
            finish = "MALFORMED_FUNCTION_CALL"
        text = (message.get("content") or "").strip()
        return Step(text, calls, self._assistant(text, calls), finish)

    @staticmethod
    def user_message(text: str) -> dict:
        return {"role": "user", "content": text}

    @staticmethod
    def tool_results(results: list[tuple[ToolCall, dict]]) -> list[dict]:
        out = []
        for call, result in results:
            value = result.get("result") if set(result) == {"result"} else result
            out.append({"role": "tool", "tool_call_id": call.id or "call_0", "_name": call.name,
                        "content": value if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)})
        return out

    def info(self) -> dict:
        """The server's model, GPU and start-up time (starts the GPU if it isn't running)."""
        return self.server.info.remote()

    def models(self) -> list[str]:
        return [self.name]


class FakeLLM:
    """For tests: replies from a script. `replies` are returned in order; a callable is
    called with (prompt, schema) and its result returned."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list = []
        self.thinks: list = []  # each generate call's thinking budget
        self.usage = Usage()

    def _next(self, prompt, schema):
        self.prompts.append(list(prompt) if isinstance(prompt, list) else prompt)  # a copy:
        # agents empty their memory list after reporting
        self.usage.calls += 1
        if not self.replies:
            raise LLMUnavailable("FakeLLM has no more replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(prompt, schema) if callable(reply) else reply

    def generate(self, prompt, system=None, schema=None, model=None, temperature=None,
                 max_tokens=None, think=None):
        self.thinks.append(think)
        reply = self._next(prompt, schema)
        if schema is not None and isinstance(reply, dict):
            return schema.model_validate(reply)
        return reply

    def chat_step(self, history, system, tools, model=None, temperature=None) -> Step:
        reply = self._next(history, None)
        return reply if isinstance(reply, Step) else Step(str(reply))

    @staticmethod
    def user_message(text: str):
        return {"role": "user", "text": text}

    @staticmethod
    def tool_results(results):
        return {"role": "tool", "results": [(c.name, r) for c, r in results]}


def make_llm(settings: Settings):
    """The configured model; raises LLMUnavailable (with the reason) if it can't be used."""
    if settings.llm_backend == "modal":
        return ModalLLM(settings, Gemini(settings) if settings.gemini_api_key else None)
    return Gemini(settings)


def model_label(settings: Settings, role: str = "answers") -> str:
    """The model doing `role` ("answers", "router" or "agents"), as shown to people."""
    if role == "agents":
        settings = agent_settings(settings)
    if settings.llm_backend == "modal":
        from ingest.llm_modal import spec_for_app
        return f"{spec_for_app(settings.modal_llm_app).served} (Modal)"
    return settings.router_model if role == "router" else settings.llm_model


def agent_settings(settings: Settings) -> Settings:
    """The settings the agents' model is made from: the chat's, unless agent_llm_backend or
    agent_modal_llm_app say otherwise."""
    update = {}
    if settings.agent_llm_backend:
        update["llm_backend"] = settings.agent_llm_backend
    if settings.agent_modal_llm_app:
        update["modal_llm_app"] = settings.agent_modal_llm_app
    return settings.model_copy(update=update) if update else settings


def separate_agent_llm(settings: Settings) -> bool:
    """The agents use a model of their own (not the chat's)."""
    mine = agent_settings(settings)
    return (mine.llm_backend, mine.modal_llm_app) != (settings.llm_backend,
                                                     settings.modal_llm_app) and \
        not (mine.llm_backend == settings.llm_backend == "gemini")


def make_agent_llm(settings: Settings):
    return make_llm(agent_settings(settings))
