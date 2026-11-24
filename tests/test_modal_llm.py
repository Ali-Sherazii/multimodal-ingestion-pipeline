"""Our own model on Modal (ModalLLM) against a fake server: the OpenAI-format requests it
sends, the replies it reads (forms, tool calls), and Gemini taking over when it can't."""
import json

from google.genai import types

from ingest.agents.harness import Agent
from ingest.config import Settings
from ingest.llm import LLMUnavailable, ModalLLM, Step, ToolCall, _to_gemini, model_label
from ingest.router import Route, fallback


class FakeServer:
    """Stands in for modal.Cls.from_name("ingest-llm", "LLMServer")()."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.bodies: list[dict] = []
        server = self

        class Chat:
            def remote(self, body):
                server.bodies.append(json.loads(json.dumps(body)))  # fails on anything not JSON
                reply = server.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
        self.chat = Chat()


class Backup:
    """Stands in for Gemini."""
    last_model = "gemini-flash-latest"

    def __init__(self, *steps):
        self.steps, self.calls, self.usage = list(steps), [], None

    def generate(self, prompt, system=None, schema=None, model=None, temperature=None,
                 max_tokens=None, think=None):
        self.calls.append(("generate", model))
        return "from gemini"

    def chat_step(self, history, system, tools, model=None, temperature=None):
        self.calls.append(("chat_step", history))
        return self.steps.pop(0)


def completion(content="", calls=(), finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": f"call-{i}", "type": "function", "function": {
            "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]
    return {"status": 200, "body": {"choices": [{"message": message, "finish_reason": finish}],
                                    "usage": {"prompt_tokens": 50, "completion_tokens": 7}}}


def make(*replies, backup=None, **values):
    server = FakeServer(*replies)
    return ModalLLM(Settings(_env_file=None, llm_backend="modal", **values), backup, server), server


ROUTE = fallback("Lahore branch revenue in Q3 FY2026").model_dump() | {"kind": "follow_up"}
FINISH = {"summary": "ok", "findings": ["20 days (policy.md)"], "files": [], "handover": ""}
TOOLS = [{"name": "search_documents", "description": "Search",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]


def test_answers_and_forms_come_from_our_own_model():
    llm, server = make(completion("Staff get 20 days [1]."), completion(json.dumps(ROUTE)))
    assert llm.generate("How many leave days?", system="Answer from the passages.") == \
        "Staff get 20 days [1]."
    route = llm.generate("And Lahore?", schema=Route, temperature=0.0, max_tokens=600, think=0)
    assert isinstance(route, Route) and route.queries == ["Lahore branch revenue in Q3 FY2026"]
    plain, form = server.bodies
    assert plain["messages"][0] == {"role": "system", "content": "Answer from the passages."}
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}   # unless asked
    assert form["response_format"]["json_schema"]["schema"] == Route.model_json_schema()
    # the model is shown the form too (vLLM only enforces it) and told to write it compactly
    told = form["messages"][0]
    assert told["role"] == "system" and '"needs_calculation"' in told["content"]
    assert "compactly on one line" in told["content"]
    assert form["temperature"] == 0.0 and form["max_tokens"] == 600
    assert llm.last_model == "gemma-4-26b" and llm.usage.by_model == {"gemma-4-26b": 2}
    assert llm.usage.input_tokens == 100


def test_thinking_when_asked_gets_room_and_a_cut_off_form_is_asked_again():
    llm, server = make(completion('{"kind": "fol', finish="length"),
                       completion(json.dumps(ROUTE)), llm_thinking_budget=1024)
    assert llm.generate("And Lahore?", schema=Route, max_tokens=600).kind == "follow_up"
    first, second = server.bodies
    assert first["chat_template_kwargs"] == {"enable_thinking": True}
    assert first["max_tokens"] == 600 + 1024 and second["max_tokens"] > first["max_tokens"]


def test_an_agent_calls_tools_through_our_own_model():
    llm, server = make(completion(calls=[("search_documents", {"query": "leave"})],
                                  finish="tool_calls"),
                       completion(calls=[("finish", FINISH)], finish="tool_calls"))
    report = Agent("a", "system", "goal", TOOLS, lambda name, args: "20 days (policy.md)", llm,
                   5, lambda line: None).run()
    assert report.ok and report.findings == ["20 days (policy.md)"]
    sent = server.bodies[1]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["messages"][2]["tool_calls"][0]["id"] == "call-0"
    assert sent["messages"][3] == {"role": "tool", "tool_call_id": "call-0",
                                   "content": "20 days (policy.md)"}   # our "_name" isn't sent
    names = [t["function"]["name"] for t in sent["tools"]]
    assert names == ["search_documents", "finish"] and sent["tools"][0]["type"] == "function"
    assert sent["max_tokens"] == 16384


def test_cut_off_and_garbled_tool_calls_are_reported_as_such():
    llm, _ = make(completion(calls=[("search_documents", "{not json")], finish="tool_calls"),
                  completion("half a reply", finish="length"))
    garbled = llm.chat_step([llm.user_message("goal")], None, TOOLS)
    assert garbled.calls == [] and garbled.finish == "MALFORMED_FUNCTION_CALL"
    assert llm.chat_step([llm.user_message("goal")], None, TOOLS).finish == "MAX_TOKENS"


def test_a_huge_prompt_keeps_the_reply_inside_the_context_window():
    llm, server = make(completion("ok"))
    llm.chat_step([llm.user_message("x" * 150_000)], None, TOOLS)
    assert server.bodies[0]["max_tokens"] < 16384


def test_gemini_answers_while_our_model_is_down_without_asking_it_each_time():
    backup = Backup()
    llm, server = make(ConnectionError("no route"), backup=backup)
    assert llm.generate("q", model="gemini-flash-lite-latest") == "from gemini"
    assert llm.generate("q2", model="gemini-flash-latest") == "from gemini"
    assert len(server.bodies) == 1              # resting: the second went straight to Gemini
    assert backup.calls == [("generate", "gemini-flash-lite-latest"),
                            ("generate", "gemini-flash-latest")]
    assert llm.last_model == "gemini-flash-latest" and backup.usage is llm.usage


def test_without_a_backup_the_reason_is_plain():
    class NotFoundError(Exception):
        pass
    llm, _ = make(NotFoundError("App 'ingest-llm' not found"))
    try:
        llm.generate("q")
    except LLMUnavailable as exc:
        assert "modal deploy src/ingest/llm_modal.py" in str(exc)
    else:
        raise AssertionError("expected LLMUnavailable")


def test_gemini_takes_over_an_agent_half_way():
    """Gemma made the first tool call, then Modal went down: Gemini gets the conversation
    so far (Gemma's call and its result as text) and finishes the job."""
    gemini_turn = types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="finish", args=FINISH))])
    backup = Backup(Step("", [ToolCall("finish", FINISH)], gemini_turn, "STOP"))
    llm, _ = make(completion(calls=[("search_documents", {"query": "leave"})],
                             finish="tool_calls"), ConnectionError("gone"), backup=backup)
    report = Agent("a", "system", "goal", TOOLS, lambda name, args: "20 days (policy.md)", llm,
                   5, lambda line: None).run()
    assert report.ok and report.findings == ["20 days (policy.md)"]
    contents = backup.calls[0][1]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert "called search_documents" in contents[1].parts[0].text
    assert contents[2].parts[0].text == "Result of search_documents:\n20 days (policy.md)"


def test_gemini_gets_its_own_turns_back_as_they_were():
    own = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
        name="search_documents", args={"query": "leave"}), thought_signature=b"sig")])
    history = [ModalLLM.user_message("goal"),
               ModalLLM._assistant("", [ToolCall("search_documents", {"query": "leave"}, "c1")],
                                   own),
               ModalLLM.tool_results([(ToolCall("search_documents", {}, "c1"),
                                       {"result": "20 days"})])]
    contents = _to_gemini(history)
    assert contents[1] is own                    # with its thought signature
    response = contents[2].parts[0].function_response
    assert response.name == "search_documents" and response.response == {"result": "20 days"}


def test_the_label_names_the_model_in_use():
    gemini = Settings(_env_file=None)
    assert model_label(gemini) == gemini.llm_model
    assert model_label(gemini, "router") == gemini.router_model
    assert model_label(Settings(_env_file=None, llm_backend="modal")) == "gemma-4-26b (Modal)"


def test_agents_can_have_a_model_of_their_own():
    from ingest.llm import agent_settings, separate_agent_llm
    chat = Settings(_env_file=None, llm_backend="modal")
    assert not separate_agent_llm(chat) and agent_settings(chat) is chat
    split = Settings(_env_file=None, llm_backend="modal", agent_modal_llm_app="ingest-llm-qwen")
    assert separate_agent_llm(split)
    assert model_label(split) == "gemma-4-26b (Modal)"
    assert model_label(split, "agents") == "qwen3.8-27b (Modal)"
    llm = ModalLLM(agent_settings(split), None, FakeServer())
    assert (llm.name, llm.max_context) == ("qwen3.8-27b", 131072)
    on_gemini = Settings(_env_file=None, llm_backend="modal", agent_llm_backend="gemini")
    assert separate_agent_llm(on_gemini) and model_label(on_gemini, "agents") == \
        on_gemini.llm_model
    # both on Gemini: the Modal app named for agents doesn't matter
    assert not separate_agent_llm(Settings(_env_file=None, agent_modal_llm_app="x"))
