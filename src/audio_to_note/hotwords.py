"""Editable hotword & domain glossary parser for ASR and LLM stages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HotwordEntry:
    term: str
    aliases: list[str] = field(default_factory=list)
    note: str = ""


_ARROW_RE = re.compile(r"\s*(?:->|=>)\s*")


def parse_hotwords(text: str) -> list[HotwordEntry]:
    """Parse hotword/glossary lines, ignoring blanks and ``#`` comments.

    Supported formats per line:
    - ``term`` (simple hotword)
    - ``wrong1, wrong2 -> correct_term`` (ASR mistake -> canonical term)
    - ``term: brief explanation`` (term with LLM glossary note)
    - ``wrong -> correct_term: brief explanation`` (combined)
    """
    entries: list[HotwordEntry] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        note = ""
        if ":" in line:
            left, right = line.split(":", 1)
            # Only treat ':' as note separator when it isn't e.g. a URL or time
            if left.strip() and right.strip():
                line = left.strip()
                note = right.strip()

        aliases: list[str] = []
        if _ARROW_RE.search(line):
            parts = _ARROW_RE.split(line, maxsplit=1)
            alias_part = parts[0].strip()
            term = parts[1].strip()
            aliases = [item.strip() for item in alias_part.split(",") if item.strip()]
        else:
            term = line

        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        entries.append(HotwordEntry(term=term, aliases=aliases, note=note))
    return entries


def load_hotwords(path: str | Path | None) -> list[HotwordEntry]:
    if not path:
        return []
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return []
    return parse_hotwords(file_path.read_text(encoding="utf-8"))


def format_for_asr(entries: list[HotwordEntry]) -> str:
    """Format hotwords as a clean comma-separated prompt/context string for ASR."""
    if not entries:
        return ""
    return ", ".join(entry.term for entry in entries)


def format_for_llm(entries: list[HotwordEntry]) -> str:
    """Format hotwords & correction rules for LLM polish/summarization prompts."""
    if not entries:
        return "(none)"
    lines: list[str] = []
    for entry in entries:
        line = f"- {entry.term}"
        if entry.aliases:
            line += f" (fix common ASR variants: {', '.join(entry.aliases)})"
        if entry.note:
            line += f" — {entry.note}"
        lines.append(line)
    return "\n".join(lines)
