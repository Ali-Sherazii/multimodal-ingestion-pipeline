"""The Gemini client against a fake Gemini server on this machine: the real google-genai
library builds the requests and reads the replies (forms, tool calls, rate limits), so the
wiring is checked without a key or the internet."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import SecretStr

from ingest.config import Settings
from ingest.llm import Gemini, LLMUnavailable
from ingest.router import Route


class FakeGemini:
    def __init__(self):
        self.replies: list[tuple[int, dict]] = []
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.requests.append({"path": self.path, "body": json.loads(body)})
                status, reply = fake.replies.pop(0)
                if status == "stall":  # say nothing for this long
                    time.sleep(reply)
                    return
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def reply(self, parts: list[dict]):
        self.replies.append((200, {"candidates": [{"content": {"role": "model", "parts": parts},
                                                   "finishReason": "STOP"}],
                                   "usageMetadata": {"promptTokenCount": 12,
                                                     "candidatesTokenCount": 5}}))


def make_gemini(fake, **settings):
    from google import genai
    from google.genai import types
    s = Settings(_env_file=None, gemini_api_key=SecretStr("test-key"), llm_retries=3,
                 **{"llm_fallback_models": [], **settings})
    llm = Gemini(s)
    llm.client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        base_url=fake.url, timeout=int(s.llm_timeout_seconds * 1000)))
    return llm


@pytest.fixture
def gemini():
    fake = FakeGemini()
    yield make_gemini(fake), fake
    fake.server.shutdown()


def test_forms_come_back_filled_in(gemini):
    llm, fake = gemini
    route = {"kind": "follow_up", "queries": ["Lahore branch revenue Q3"], "file_type": "any",
             "content": "table", "file_hint": "", "mode": "hybrid", "terms": [],
             "needs_calculation": False, "reply": ""}
    fake.reply([{"text": json.dumps(route)}])
    found = llm.generate("And Lahore?", system="route it", schema=Route, model="gemini-x")
    assert isinstance(found, Route) and found.queries == ["Lahore branch revenue Q3"]
    sent = fake.requests[0]
    assert "gemini-x:generateContent" in sent["path"]
    config = sent["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert "responseSchema" in config or "responseJsonSchema" in config
    assert sent["body"]["systemInstruction"]["parts"][0]["text"] == "route it"
    assert llm.usage.calls == 1 and llm.usage.input_tokens == 12


def test_tool_calls_and_their_results(gemini):
    llm, fake = gemini
    fake.reply([{"functionCall": {"name": "search_documents", "args": {"query": "leave"}}}])
    tools = [{"name": "search_documents", "description": "search",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"]}}]
    history = [llm.user_message("How many leave days?")]
    step = llm.chat_step(history, "be useful", tools)
    assert step.calls[0].name == "search_documents" and step.calls[0].args == {"query": "leave"}
    history += [step.raw, llm.tool_results([(step.calls[0], {"result": "20 days [1]"})])]
    fake.reply([{"text": "20 days [1]."}])
    assert llm.chat_step(history, "be useful", tools).text == "20 days [1]."
    sent = fake.requests[1]["body"]
    assert sent["tools"][0]["functionDeclarations"][0]["name"] == "search_documents"
    assert sent["contents"][2]["parts"][0]["functionResponse"]["response"] == {"result": "20 days [1]"}


def test_rate_limits_are_waited_out_and_daily_quota_reported(gemini, monkeypatch):
    llm, fake = gemini
    waits = []
    monkeypatch.setattr("ingest.llm.time.sleep", waits.append)
    limit = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                       "message": "Quota exceeded. Please retry in 7s.",
                       "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                    "retryDelay": "7s"}]}}
    fake.replies.append((429, limit))
    fake.reply([{"text": "ok"}])
    assert llm.generate("hi", model="gemini-x") == "ok" and waits == [pytest.approx(7, abs=0.5)]
    daily = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                       "Quota exceeded for metric: generate_content_free_tier_requests, "
                       "limit: 20, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}
    fake.replies.append((429, daily))
    with pytest.raises(LLMUnavailable, match="quota for today"):
        llm.generate("hi", model="gemini-x")
    fake.replies.append((404, {"error": {"code": 404, "status": "NOT_FOUND",
                                         "message": "models/nope is not found"}}))
    with pytest.raises(LLMUnavailable, match="no model 'nope'"):
        llm.generate("hi", model="nope")


def test_a_busy_or_silent_model_hands_over_to_the_next_and_is_left_alone():
    """Free-tier models are often busy (503) or stall: the next model answers, and the
    failed one is skipped for a while instead of slowing every later call."""
    fake = FakeGemini()
    try:
        llm = make_gemini(fake, llm_fallback_models=["backup"], llm_timeout_seconds=1)
        busy = {"error": {"code": 503, "status": "UNAVAILABLE",
                          "message": "This model is currently experiencing high demand."}}
        fake.replies.append((503, busy))
        fake.reply([{"text": "from backup"}])
        assert llm.generate("hi", model="main") == "from backup" and llm.last_model == "backup"
        fake.reply([{"text": "backup again"}])
        assert llm.generate("hi", model="main") == "backup again"   # main is resting
        assert [r["path"].split("/")[-1] for r in fake.requests] == [
            "main:generateContent", "backup:generateContent", "backup:generateContent"]
        assert llm.usage.by_model == {"backup": 2}

        llm._rest.clear()
        fake.replies.append(("stall", 3))  # no reply within the 1 s limit
        fake.reply([{"text": "after a stall"}])
        assert llm.generate("hi", model="main") == "after a stall"
        assert llm.last_model == "backup"
    finally:
        fake.server.shutdown()


def test_a_reply_cut_short_by_thinking_is_asked_again_with_more_room(gemini):
    """Newer models count thinking in the reply's tokens: seen live, the router's 600 went
    on ~570 tokens of thinking and the form was cut off mid-way."""
    llm, fake = gemini
    fake.replies.append((200, {"candidates": [{"content": {"role": "model", "parts": [
        {"text": '{"kind":"follow_up","queries":["Lahore branch'}]}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 12,
                          "thoughtsTokenCount": 572}}))
    route = {"kind": "follow_up", "queries": ["Lahore branch revenue Q3"], "file_type": "any",
             "content": "any", "file_hint": "", "mode": "hybrid", "terms": [],
             "needs_calculation": False, "reply": ""}
    fake.reply([{"text": json.dumps(route)}])
    found = llm.generate("And Lahore?", schema=Route, model="gemini-x", max_tokens=600, think=0)
    assert found.queries == ["Lahore branch revenue Q3"]
    first, second = (r["body"]["generationConfig"] for r in fake.requests)
    assert first["maxOutputTokens"] == 600 and second["maxOutputTokens"] > 600 + 4000
    sent = first["thinkingConfig"]  # the library writes thinking_budget; the API takes both
    assert sent.get("thinkingBudget", sent.get("thinking_budget")) == 0


def test_a_model_refusing_the_thinking_setting_is_asked_without_it(gemini):
    """Seen live: gemini-flash-lite-latest answers thinking_budget=0 with a bare 400
    "Request contains an invalid argument." (no mention of thinking)."""
    llm, fake = gemini
    fake.replies.append((400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                         "message": "Request contains an invalid argument."}}))
    fake.reply([{"text": "ok"}])
    fake.reply([{"text": "ok again"}])
    assert llm.generate("hi", model="lite", think=0) == "ok"
    assert llm.generate("hi", model="lite", think=0) == "ok again"   # remembered: one request
    sent = [r["body"]["generationConfig"] for r in fake.requests]
    assert "thinkingConfig" in sent[0] and "thinkingConfig" not in sent[1]
    assert len(sent) == 3 and "thinkingConfig" not in sent[2]


def test_no_key_says_where_to_get_one():
    with pytest.raises(LLMUnavailable, match="aistudio.google.com"):
        Gemini(Settings(_env_file=None, gemini_api_key=None))
