"""Text quality checks: garbage text layers and instruction-like text aimed at a chatbot."""
import re
import unicodedata

_SYMBOLS = set("|@#!^~`<>{}\\")
_GARBAGE_SHARE = 0.03   # real text in the test set scores 0%; a garbage OCR layer ~14%
_GARBAGE_MIN = 5

_INSTRUCTIONS = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}?\b(previous|prior|above|earlier|all|any|"
    r"your)\b[^.\n]{0,25}?\b(instructions?|prompts?|rules|directions)\b"
    r"|\bsystem prompt\b"
    r"|\byou are now\b"
    r"|\bdo not (tell|reveal|mention)\b[^.\n]{0,30}\b(user|anyone|human)\b",
    re.IGNORECASE)


def _is_garbage(token: str) -> bool:
    if sum(ch in _SYMBOLS for ch in token) >= 2:  # e.g. "armerl|!##al@IO"
        return True
    # replacement characters, private-use glyphs and control codes come from broken fonts
    return any(ch == "�" or unicodedata.category(ch) in ("Co", "Cc") for ch in token)


def garbage_share(texts: list[str]) -> tuple[int, int]:
    """(garbage tokens, tokens) over plain text; Markdown table lines are left out."""
    lines = [l for t in texts for l in t.splitlines() if not l.lstrip().startswith("|")]
    tokens = [t.strip(".,;:?()[]\"'") for t in " ".join(lines).split()]
    tokens = [t for t in tokens if len(t) >= 3]
    return sum(_is_garbage(t) for t in tokens), len(tokens)


def looks_like_garbage(texts: list[str]) -> bool:
    bad, total = garbage_share(texts)
    return bad >= _GARBAGE_MIN and bad / max(total, 1) >= _GARBAGE_SHARE


def find_instructions(text: str) -> str | None:
    """Text that reads like an instruction to an AI ('ignore previous instructions...')."""
    match = _INSTRUCTIONS.search(text)
    return match.group(0) if match else None
