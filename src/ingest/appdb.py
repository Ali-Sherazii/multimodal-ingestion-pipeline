"""The app database: users, sessions and chats, in one SQLite file (app_db in config.yaml).

Chunks and vectors stay in Qdrant; this holds only what the web app and chat need. SQLite
comes with Python, keeps everything in a single file and is safe for the web app's threads
(one connection, calls serialised by a lock).
"""
import sqlite3
import threading
from pathlib import Path


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")  # readers don't wait for a writer

    def run(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        """One statement, committed."""
        with self._lock, self._db:
            return self._db.execute(sql, args).fetchall()

    def insert(self, sql: str, args: tuple = ()) -> int:
        with self._lock, self._db:
            return self._db.execute(sql, args).lastrowid

    def script(self, sql: str) -> None:
        with self._lock, self._db:
            self._db.executescript(sql)

    def close(self) -> None:
        self._db.close()
