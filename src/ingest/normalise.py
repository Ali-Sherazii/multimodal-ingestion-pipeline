"""Normalise: clean every extractor's text the same way and drop empty segments."""
import re
import unicodedata

from ingest.models import Segment

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f­​﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    # NFKC folds look-alike characters (ligatures like "ﬁ", full-width letters) into plain ones
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalise(segments: list[Segment]) -> list[Segment]:
    out = []
    for seg in segments:
        seg.text = clean_text(seg.text)
        if seg.section_heading:
            seg.section_heading = clean_text(seg.section_heading) or None
        if seg.text:
            out.append(seg)
    return out
