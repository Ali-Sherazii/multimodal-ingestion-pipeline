"""Users: each one uploads and chats with their own files (the supervisor's step 4).

With no users added, nothing changes: the web app serves one person on this machine without
a login. Once a user exists (`ingest user add NAME`), everyone logs in, and:

  - a user's uploads are private to them: their chunks carry the access group "user:NAME"
    and owner NAME (indexed as a Qdrant tenant), so no one else's search can return them,
    unless they share the upload with a group they belong to (or everyone);
  - a user's searches and chats read only what their groups may read: "everyone", their own
    "user:NAME" and the groups given to them (e.g. hr), enforced by the server, not the page;
  - admins (`--admin`) read and manage everything.

Passwords are kept as salted PBKDF2-SHA256 hashes (Python's hashlib); a login gives a signed
session cookie that expires after session_hours.
"""
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass, field

from ingest.appdb import Database

ITERATIONS = 310_000  # PBKDF2 rounds (OWASP's figure for SHA-256)
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")


@dataclass
class User:
    name: str
    groups: list[str] = field(default_factory=list)
    admin: bool = False

    @property
    def reads(self) -> list[str] | None:
        """The access groups this user may read; None = everything (admins)."""
        if self.admin:
            return None
        return list(dict.fromkeys(["everyone", self.private, *self.groups]))

    @property
    def private(self) -> str:
        """The access group only this user has."""
        return f"user:{self.name}"

    def may_share_with(self, group: str) -> bool:
        return self.admin or group == "everyone" or group in self.groups

    def as_dict(self) -> dict:
        return {"name": self.name, "groups": self.groups, "admin": self.admin}


LOCAL = User("local", admin=True)  # the one person on this machine when no users exist

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY, salt TEXT NOT NULL, hash TEXT NOT NULL,
    groups TEXT NOT NULL DEFAULT '[]', admin INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()


class Users:
    def __init__(self, db: Database, session_hours: float = 72):
        self.db = db
        self.session_seconds = session_hours * 3600
        db.script(_SCHEMA)
        row = db.run("SELECT value FROM app_settings WHERE key = 'session_secret'")
        if row:
            self._secret = bytes.fromhex(row[0]["value"])
        else:  # made once: sessions stay valid across restarts
            self._secret = secrets.token_bytes(32)
            db.run("INSERT INTO app_settings VALUES ('session_secret', ?)", (self._secret.hex(),))

    # ---- accounts ----

    def any(self) -> bool:
        return bool(self.db.run("SELECT 1 FROM users LIMIT 1"))

    def add(self, name: str, password: str, groups: list[str] = (), admin: bool = False) -> User:
        name = name.strip().lower()
        if not NAME.match(name):
            raise ValueError("A user name is 1-40 lower-case letters, digits, '.', '_' or '-'")
        if len(password) < 8:
            raise ValueError("Use a password of at least 8 characters")
        if self.get(name):
            raise ValueError(f"User '{name}' already exists")
        salt = secrets.token_bytes(16)
        groups = [g.strip() for g in groups if g.strip()]
        self.db.run("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                    (name, salt.hex(), _hash(password, salt), json.dumps(groups), int(admin),
                     time.time()))
        return User(name, groups, admin)

    def set_password(self, name: str, password: str) -> None:
        if len(password) < 8:
            raise ValueError("Use a password of at least 8 characters")
        if not self.get(name):
            raise ValueError(f"No user '{name}'")
        salt = secrets.token_bytes(16)
        self.db.run("UPDATE users SET salt = ?, hash = ? WHERE name = ?",
                    (salt.hex(), _hash(password, salt), name))

    def set_groups(self, name: str, groups: list[str]) -> None:
        self.db.run("UPDATE users SET groups = ? WHERE name = ?",
                    (json.dumps([g.strip() for g in groups if g.strip()]), name))

    def remove(self, name: str) -> bool:
        return bool(self.db.run("DELETE FROM users WHERE name = ? RETURNING name", (name,)))

    def get(self, name: str) -> User | None:
        rows = self.db.run("SELECT name, groups, admin FROM users WHERE name = ?", (name,))
        return User(rows[0]["name"], json.loads(rows[0]["groups"]), bool(rows[0]["admin"])) \
            if rows else None

    def all(self) -> list[User]:
        return [User(r["name"], json.loads(r["groups"]), bool(r["admin"]))
                for r in self.db.run("SELECT name, groups, admin FROM users ORDER BY name")]

    def check(self, name: str, password: str) -> User | None:
        """The user, if the password is theirs (same time taken whether or not they exist)."""
        rows = self.db.run("SELECT salt, hash FROM users WHERE name = ?", (name.strip().lower(),))
        salt, stored = (bytes.fromhex(rows[0]["salt"]), rows[0]["hash"]) if rows \
            else (b"\0" * 16, "0" * 64)
        ok = hmac.compare_digest(_hash(password, salt), stored)
        return self.get(name.strip().lower()) if ok and rows else None

    # ---- sessions ----

    def session(self, user: User) -> str:
        """A signed token naming the user, valid for session_hours."""
        body = base64.urlsafe_b64encode(json.dumps(
            [user.name, time.time() + self.session_seconds, secrets.token_hex(4)]).encode())
        return body.decode() + "." + hmac.new(self._secret, body, "sha256").hexdigest()

    def from_session(self, token: str | None) -> User | None:
        if not token or "." not in token:
            return None
        body, signature = token.rsplit(".", 1)
        if not hmac.compare_digest(hmac.new(self._secret, body.encode(), "sha256").hexdigest(),
                                   signature):
            return None
        try:
            name, expires, _ = json.loads(base64.urlsafe_b64decode(body))
        except ValueError:
            return None
        return self.get(name) if expires > time.time() else None
