"""Chats and their messages, with each chat's latest memory snapshot (memory.py)."""
import json
import time
import uuid

from ingest.appdb import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
    created REAL NOT NULL, updated REAL NOT NULL,
    snapshot TEXT, snapshot_upto INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS chats_owner ON chats (owner, updated);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat_id, id);
"""


def _chat(row) -> dict:
    return {"id": row["id"], "owner": row["owner"], "title": row["title"],
            "created": row["created"], "updated": row["updated"],
            "snapshot": json.loads(row["snapshot"]) if row["snapshot"] else None,
            "snapshot_upto": row["snapshot_upto"]}


def _message(row) -> dict:
    return {"id": row["id"], "role": row["role"], "content": row["content"],
            "created": row["created"], **json.loads(row["data"])}


class Chats:
    def __init__(self, db: Database):
        self.db = db
        db.script(_SCHEMA)

    def create(self, owner: str, title: str = "New chat") -> dict:
        now = time.time()
        chat_id = uuid.uuid4().hex[:16]
        self.db.run("INSERT INTO chats (id, owner, title, created, updated) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, owner, title[:120], now, now))
        return self.get(chat_id)

    def get(self, chat_id: str, owner: str | None = None) -> dict | None:
        """The chat; with `owner`, only if it is theirs."""
        rows = self.db.run("SELECT * FROM chats WHERE id = ?", (chat_id,))
        if not rows or (owner is not None and rows[0]["owner"] != owner):
            return None
        return _chat(rows[0])

    def of(self, owner: str) -> list[dict]:
        rows = self.db.run("SELECT * FROM chats WHERE owner = ? ORDER BY updated DESC", (owner,))
        return [_chat(r) for r in rows]

    def rename(self, chat_id: str, title: str) -> None:
        self.db.run("UPDATE chats SET title = ? WHERE id = ?", (title[:120], chat_id))

    def delete(self, chat_id: str) -> None:
        self.db.run("DELETE FROM chats WHERE id = ?", (chat_id,))

    def add(self, chat_id: str, role: str, content: str, data: dict | None = None) -> int:
        now = time.time()
        message_id = self.db.insert(
            "INSERT INTO messages (chat_id, role, content, data, created) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, json.dumps(data or {}, ensure_ascii=False), now))
        self.db.run("UPDATE chats SET updated = ? WHERE id = ?", (now, chat_id))
        return message_id

    def messages(self, chat_id: str, after: int = 0) -> list[dict]:
        rows = self.db.run("SELECT * FROM messages WHERE chat_id = ? AND id > ? ORDER BY id",
                           (chat_id, after))
        return [_message(r) for r in rows]

    def set_snapshot(self, chat_id: str, snapshot: dict, upto: int) -> None:
        """The chat's memory up to message `upto`: later messages are kept word for word."""
        self.db.run("UPDATE chats SET snapshot = ?, snapshot_upto = ? WHERE id = ?",
                    (json.dumps(snapshot, ensure_ascii=False), upto, chat_id))
