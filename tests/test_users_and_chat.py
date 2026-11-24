"""Users (private uploads, shared groups), and the chat: router, answers, memory snapshots.
No real language model: FakeLLM replies from a script."""
import time

import pytest

from ingest import memory, router
from ingest.answer import check_citations
from ingest.appdb import Database
from ingest.chat import ChatEngine, merge
from ingest.chats import Chats
from ingest.llm import FakeLLM, LLMUnavailable
from ingest.retrieve import Passage
from ingest.users import LOCAL, Users


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


def route(**fields) -> dict:
    base = {"kind": "search", "queries": [], "file_type": "any", "content": "any",
            "file_hint": "", "mode": "hybrid", "terms": [], "needs_calculation": False,
            "reply": ""}
    return {**base, **fields}


def passage(text: str, ids: list[str], path: str = "/d/a.md") -> Passage:
    return Passage(text=text, citation=f"{path.rsplit('/', 1)[-1]}, p. 1", score=1.0,
                   source_path=path, file_type="text", modality="text", match=text,
                   page_number=1, chunk_ids=ids)


# ---- users ----

def test_passwords_sessions_and_groups(db):
    users = Users(db, session_hours=1)
    assert not users.any()
    alice = users.add("Alice", "correct horse", groups=["hr"])
    assert alice.name == "alice" and users.any()
    assert users.check("alice", "correct horse").name == "alice"
    assert users.check("alice", "wrong password") is None
    assert users.check("nobody", "correct horse") is None
    token = users.session(alice)
    assert users.from_session(token).name == "alice"
    assert users.from_session(token[:-1] + ("0" if token[-1] != "0" else "1")) is None  # tampered
    assert alice.reads == ["everyone", "user:alice", "hr"]
    assert alice.may_share_with("hr") and not alice.may_share_with("finance")
    assert LOCAL.reads is None  # the no-login user reads everything
    with pytest.raises(ValueError):
        users.add("alice", "another password")
    with pytest.raises(ValueError):
        users.add("bad name!", "long enough")
    with pytest.raises(ValueError):
        users.add("bob", "short")


def test_session_expires(db):
    users = Users(db, session_hours=0.0001)  # about a third of a second
    token = users.session(users.add("carol", "password1"))
    time.sleep(0.5)
    assert users.from_session(token) is None


def test_each_user_gets_their_own_private_copy(make_pipeline, docs):
    pipeline = make_pipeline()
    policy = docs / "policy.md"
    a = pipeline.ingest_file(policy, groups=["user:alice"], owner="alice")
    b = pipeline.ingest_file(policy, groups=["user:bob"], owner="bob")
    again = pipeline.ingest_file(policy, groups=["user:alice"], owner="alice")
    assert (a.status, b.status, again.status) == ("processed", "processed", "skipped")
    document = next(pipeline.store.iter_points(with_vectors=False)).payload["document_id"]
    assert {by for _, by in pipeline.store.copies(document)} == {"alice", "bob"}
    alice_sees = pipeline.search("paid annual leave", groups=["everyone", "user:alice"])
    assert alice_sees and {h.payload["owner"] for h in alice_sees} == {"alice"}
    assert pipeline.search("paid annual leave", groups=["everyone", "user:carol"]) == []


# ---- chat pieces ----

def test_citations_are_checked():
    passages = [passage("a", ["1"]), passage("b", ["2"])]
    answer = check_citations("Revenue was 3,663 [2]. Growth [1-2], see [7].", passages)
    assert answer.text == "Revenue was 3,663 [2]. Growth [1][2], see ."
    assert [c["n"] for c in answer.citations] == [2, 1]
    assert answer.grounded
    assert not check_citations("Revenue was high.", passages).grounded
    assert check_citations("The passages don't contain the 2019 figure.", passages).grounded


def test_router_resolves_follow_ups_and_falls_back():
    llm = FakeLLM(route(kind="follow_up", queries=["Lahore branch revenue Q3 FY2026"]),
                  LLMUnavailable("quota"))
    found, problem = router.route("and Lahore?", llm, "User: Hyderabad Q3 revenue?")
    assert found.kind == "follow_up" and found.queries == ["Lahore branch revenue Q3 FY2026"]
    assert problem is None and "Hyderabad" in llm.prompts[0]
    found, problem = router.route("and Lahore?", llm)
    assert found.queries == ["and Lahore?"] and problem == "quota"


def test_file_hints_and_merging():
    files = {"kestrel_annual_report_2026.pdf": "/u/k.pdf", "notes.txt": "/u/n.txt"}
    assert router.file_for("Kestrel report", files) == "/u/k.pdf"
    assert router.file_for("annual", {**files, "annual_2025.pdf": "/u/a.pdf"}) is None
    first = [passage("x", ["1"]), passage("y", ["2"])]
    second = [passage("x again", ["1", "9"]), passage("z", ["3"])]
    assert [p.text for p in merge([first, second])] == ["x", "y", "z"]


def test_context_keeps_the_snapshot_and_latest_messages():
    chat = {"snapshot": {"summary": "Kestrel revenue", "user_goal": "compare branches",
                         "entities": [{"name": "Hyderabad", "kind": "branch",
                                       "facts": ["Q3 revenue 3,663"]}],
                         "relationships": [], "key_facts": [], "open_questions": [],
                         "documents": ["kestrel.pdf"]}}
    messages = [{"role": "user", "content": "x" * 4000}, {"role": "assistant", "content": "ok"}]
    text = memory.context(chat, messages, history_messages=8, budget=200)
    assert "Hyderabad (branch): Q3 revenue 3,663" in text
    assert "x" * 100 not in text and "Assistant: ok" in text  # the long one dropped to fit


def test_chat_answers_with_citations_and_takes_snapshots(make_pipeline, docs, db):
    pipeline = make_pipeline(snapshot_every=4, answer_passages=3)
    pipeline.run([docs])
    chats = Chats(db)
    snapshot = {"summary": "leave policy", "user_goal": "know leave days", "entities": [],
                "relationships": [], "key_facts": ["20 days paid leave (policy.md)"],
                "open_questions": [], "documents": ["policy.md"]}
    llm = FakeLLM(route(queries=["paid annual leave days"]),
                  "Employees get 20 days of paid annual leave [1].",
                  route(kind="chit_chat", reply="You're welcome!"),
                  snapshot)
    engine = ChatEngine(pipeline, chats, llm=llm)
    chat = chats.create("local")
    reply = engine.ask(chat["id"], LOCAL, "How many leave days?", wait_for_memory=True)
    assert reply["content"].endswith("[1].") and reply["citations"][0]["n"] == 1
    assert "policy.md" in reply["citations"][0]["citation"]
    assert chats.get(chat["id"])["title"] == "How many leave days?"
    assert engine.ask(chat["id"], LOCAL, "thanks!", wait_for_memory=True)["content"] == \
        "You're welcome!"
    saved = chats.get(chat["id"])
    assert saved["snapshot"]["key_facts"] == ["20 days paid leave (policy.md)"]
    assert saved["snapshot_upto"] == chats.messages(chat["id"])[-1]["id"]
    assert len(chats.messages(chat["id"], after=saved["snapshot_upto"])) == 0


def test_an_answer_needing_arithmetic_may_think_first(make_pipeline, docs, db):
    pipeline = make_pipeline(calculation_thinking_budget=1024)
    pipeline.run([docs])
    chats = Chats(db)
    llm = FakeLLM(route(queries=["paid annual leave days"]), "20 days [1].",
                  route(queries=["paid annual leave days"], needs_calculation=True),
                  "20 x 2 = 40 days [1].")
    engine = ChatEngine(pipeline, chats, llm=llm)
    chat = chats.create("local")
    engine.ask(chat["id"], LOCAL, "How many leave days?")
    engine.ask(chat["id"], LOCAL, "And over two years?")
    assert llm.thinks == [0, None, 0, 1024]  # router, plain answer, router, calculation


def test_chat_without_the_language_model_shows_passages(make_pipeline, docs, db):
    pipeline = make_pipeline()
    pipeline.run([docs])
    chats = Chats(db)
    engine = ChatEngine(pipeline, chats)  # no key in the test settings
    chat = chats.create("local")
    reply = engine.ask(chat["id"], LOCAL, "Who approves invoice INV-77821?")
    assert "isn't available" in reply["content"] and reply["notice"]
    assert reply["passages"] and "INV-77821" in reply["passages"][0]["text"]


def test_chats_belong_to_their_owner(db):
    chats = Chats(db)
    chat = chats.create("alice")
    assert chats.get(chat["id"], owner="alice") and chats.get(chat["id"], owner="bob") is None
    chats.add(chat["id"], "user", "hi")
    chats.delete(chat["id"])
    assert chats.get(chat["id"]) is None and chats.messages(chat["id"]) == []


# ---- the web app with users ----

def test_web_app_keeps_users_apart(make_pipeline, docs, db, tmp_path):
    from fastapi.testclient import TestClient

    from ingest.web.app import create_app
    pipeline = make_pipeline()
    users = Users(db)
    app = create_app(pipeline, pipeline.settings, db=db, llm=FakeLLM())
    users.add("alice", "password-a")
    users.add("bob", "password-b", groups=["hr"])
    users.add("root", "password-r", admin=True)
    alice, bob, root = TestClient(app), TestClient(app), TestClient(app)
    assert alice.get("/api/documents").status_code == 401
    for client, name in ((alice, "alice"), (bob, "bob"), (root, "root")):
        assert client.post("/api/login", json={"name": name, "password": f"password-{name[0]}"}
                           ).status_code == 200
    assert alice.post("/api/login", json={"name": "alice", "password": "nope"}).status_code == 401

    def upload(client, path, groups=""):
        with open(path, "rb") as f:
            job = client.post("/api/upload", files={"files": (path.name, f)},
                              data={"groups": groups}).json()
        for _ in range(100):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] == "done":
                return job
            time.sleep(0.05)
        raise AssertionError("upload didn't finish")

    assert upload(alice, docs / "policy.md")["files"][0]["status"] == "processed"
    assert upload(bob, docs / "notes.txt", groups="hr")["files"][0]["status"] == "processed"
    assert alice.post("/api/upload", files={"files": ("x.txt", b"x")},
                      data={"groups": "hr"}).status_code == 403  # alice isn't in hr

    names = lambda client: {d["name"] for d in client.get("/api/documents").json()}
    assert names(alice) == {"policy.md"} and names(bob) == {"notes.txt"}
    assert names(root) == {"policy.md", "notes.txt"}
    found = lambda client: {r["citation"].split(",")[0].split(" ·")[0] for r in
                            client.get("/api/search", params={"q": "Lahore office growth"}).json()}
    assert found(alice) == {"policy.md"} and found(bob) == {"notes.txt"}
    policy = next(d for d in root.get("/api/documents").json() if d["name"] == "policy.md")
    assert bob.get("/api/chunks", params={"source": policy["source_path"]}).status_code == 404
    assert bob.delete("/api/documents", params={"source": policy["source_path"]}
                      ).status_code == 404

    chat = alice.post("/api/chats").json()
    assert bob.get(f"/api/chats/{chat['id']}").status_code == 404
    reply = alice.post(f"/api/chats/{chat['id']}/ask",
                       json={"question": "How many days of leave?"}).json()
    assert reply["passages"] and all("policy.md" in p["citation"] for p in reply["passages"])
    assert alice.delete("/api/documents", params={"source": policy["source_path"]}
                        ).json()["deleted"] > 0
