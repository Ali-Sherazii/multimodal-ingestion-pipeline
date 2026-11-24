"""Permissions: which groups may read each document.

Every chunk stores the groups allowed to read its document, and a search can be limited to a
user's groups, so retrieval never hands a user text from a file they couldn't open. The
groups come from rules in access.yaml, matched against the file's path (first match wins):

    default: [everyone]
    rules:
      - match: "*/Data_to_be_ingested/HR/*"
        groups: [hr]
      - match: "*.mp4"
        groups: [everyone, video-team]

Patterns use * (any characters, folders included) and ?, case-insensitive, with / in paths.
The pipeline has no logins of its own: the application that searches passes the user's
groups (`ingest search --groups hr,everyone`, or Pipeline.search(groups=...)).
"""
import fnmatch
from pathlib import Path

import yaml

from ingest.config import Settings


class AccessRules:
    def __init__(self, rules: list[tuple[str, list[str]]], default: list[str]):
        self.rules = rules
        self.default = default

    @classmethod
    def load(cls, settings: Settings) -> "AccessRules":
        path = Path(settings.access_file)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.is_file() else {}
        rules = [(str(rule["match"]).replace("\\", "/").lower(), [str(g) for g in rule["groups"]])
                 for rule in data.get("rules") or []]
        return cls(rules, [str(g) for g in data.get("default") or settings.default_access])

    def groups_for(self, source_path: str) -> list[str]:
        path = source_path.replace("\\", "/").lower()
        return next((groups for pattern, groups in self.rules
                     if fnmatch.fnmatchcase(path, pattern)), self.default)
