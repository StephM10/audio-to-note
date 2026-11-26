"""Prompt template discovery, parsing, rendering, and user persistence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

BUILTIN_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class PromptTemplate:
    id: str
    name: str
    description: str
    stage: str  # "summarize" | "polish"
    system_prompt: str
    user_prompt: str
    source_path: str = ""
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(
        self,
        *,
        transcript: str,
        hotwords: str = "(none)",
        title: str = "Recording",
        output_language: str = "auto",
    ) -> tuple[str, str]:
        if not output_language or output_language == "auto":
            lang_instruction = (
                "Write the output in the primary language of the transcript "
                "(use Chinese if the transcript is mostly Chinese, English if mostly English, "
                "or keep bilingual terms intact)."
            )
        else:
            lang_instruction = f"Write the output in {output_language}."

        values = {
            "transcript": transcript,
            "hotwords": hotwords or "(none)",
            "title": title,
            "output_language": output_language,
            "output_language_instruction": lang_instruction,
        }
        return _safe_format(self.system_prompt, values), _safe_format(self.user_prompt, values)


def _safe_format(template: str, values: dict[str, str]) -> str:
    result = template
    for key, val in values.items():
        result = result.replace("{" + key + "}", val)
    return result


def parse_prompt_markdown(text: str, *, fallback_id: str, source_path: str = "", builtin: bool = False) -> PromptTemplate:
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\r\n")

    sys_match = re.search(r"^##\s+System\s*$", body, flags=re.MULTILINE)
    usr_match = re.search(r"^##\s+User\s*$", body, flags=re.MULTILINE)
    if sys_match and usr_match and sys_match.start() < usr_match.start():
        system_prompt = body[sys_match.end() : usr_match.start()].strip()
        user_prompt = body[usr_match.end() :].strip()
    elif usr_match:
        system_prompt = body[: usr_match.start()].strip()
        user_prompt = body[usr_match.end() :].strip()
    else:
        system_prompt = "You are a helpful assistant."
        user_prompt = body.strip()

    pid = str(meta.get("id") or fallback_id).strip()
    return PromptTemplate(
        id=pid,
        name=str(meta.get("name") or pid.replace("_", " ").title()),
        description=str(meta.get("description") or ""),
        stage=str(meta.get("stage") or "summarize"),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_path=source_path,
        builtin=builtin,
    )


def serialize_prompt_markdown(prompt: PromptTemplate) -> str:
    frontmatter = yaml.safe_dump(
        {
            "id": prompt.id,
            "name": prompt.name,
            "description": prompt.description,
            "stage": prompt.stage,
        },
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{frontmatter}\n---\n## System\n{prompt.system_prompt.strip()}\n\n## User\n{prompt.user_prompt.strip()}\n"


def list_prompts(project_root: Path) -> list[PromptTemplate]:
    """Load built-in prompts and apply overrides/custom files from ``<root>/prompts/``."""
    templates: dict[str, PromptTemplate] = {}
    if BUILTIN_PROMPTS_DIR.is_dir():
        for file_path in sorted(BUILTIN_PROMPTS_DIR.glob("*.md")):
            tpl = parse_prompt_markdown(
                file_path.read_text(encoding="utf-8"),
                fallback_id=file_path.stem,
                source_path=str(file_path),
                builtin=True,
            )
            templates[tpl.id] = tpl

    user_dir = project_root / "prompts"
    if user_dir.is_dir():
        for file_path in sorted(user_dir.glob("*.md")):
            tpl = parse_prompt_markdown(
                file_path.read_text(encoding="utf-8"),
                fallback_id=file_path.stem,
                source_path=str(file_path),
                builtin=False,
            )
            templates[tpl.id] = tpl

    return list(templates.values())


def get_prompt(project_root: Path, prompt_id: str) -> PromptTemplate:
    for tpl in list_prompts(project_root):
        if tpl.id == prompt_id:
            return tpl
    raise KeyError(f"Prompt template not found: {prompt_id!r}")


def save_prompt(project_root: Path, prompt: PromptTemplate) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt.id).strip("_") or "custom"
    prompt.id = safe_id
    target_dir = project_root / "prompts"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{safe_id}.md"
    target_path.write_text(serialize_prompt_markdown(prompt), encoding="utf-8")
    prompt.source_path = str(target_path)
    prompt.builtin = False
    return target_path
