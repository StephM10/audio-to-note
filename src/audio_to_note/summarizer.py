"""Transcript polishing and single-pass / map-reduce LLM summarization."""

from __future__ import annotations

import re
from typing import Any, Callable

from .hotwords import HotwordEntry, format_for_llm
from .llm import LLMProvider, LLMResponse, TokenCallback
from .models import Segment, Transcript, format_timestamp, parse_timestamp
from .prompts import PromptTemplate


def estimate_tokens(text: str) -> int:
    """Estimate token count robustly for both English and CJK text."""
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    other_chars = max(0, len(text) - cjk_chars)
    return int(cjk_chars / 1.5 + other_chars / 4.0) + 1


def split_lines_by_tokens(lines: list[str], max_tokens: int, overlap_lines: int = 3) -> list[str]:
    """Group transcript lines into token-bounded chunks with small line overlap."""
    if not lines:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append("\n".join(current))
            tail = current[-overlap_lines:] if overlap_lines > 0 else []
            current = [*tail, line]
            current_tokens = sum(estimate_tokens(item) for item in current)
        else:
            current.append(line)
            current_tokens += line_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\[(?P<ts>\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\]\s+"
    r"(?P<speaker>[^:*\n]+?)(?:\*\*)?\s*:\s*(?P<content>.*)$"
)


def _parse_polished_lines(polished_text: str, original_segments: list[Segment]) -> list[Segment]:
    parsed: list[Segment] = []
    for idx, line in enumerate(polished_text.splitlines()):
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        start = parse_timestamp(match.group("ts"))
        end = original_segments[idx].end if idx < len(original_segments) else start
        parsed.append(
            Segment(
                start=start,
                end=max(start, end),
                speaker=match.group("speaker").strip(),
                text=match.group("content").strip(),
            )
        )
    return parsed


def polish_transcript(
    transcript: Transcript,
    llm: LLMProvider,
    prompt: PromptTemplate,
    hotwords: list[HotwordEntry],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_context_tokens: int = 8000,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[Transcript, dict[str, Any]]:
    """Run an LLM cleanup pass over transcript lines while keeping timestamps and speakers."""
    plain_lines = transcript.to_plain_text(apply_aliases=True).splitlines()
    if not plain_lines:
        return transcript, {"chunks": 0, "tokens": 0}

    chunk_budget = max(1200, int(max_context_tokens * 0.45))
    chunks = split_lines_by_tokens(plain_lines, max_tokens=chunk_budget, overlap_lines=0)
    hotwords_block = format_for_llm(hotwords)

    polished_blocks: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for idx, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(idx, len(chunks))
        sys_p, usr_p = prompt.render(
            transcript=chunk,
            hotwords=hotwords_block,
            title="Transcript Polish",
            output_language="auto",
        )
        resp = llm.complete(
            sys_p,
            usr_p,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        polished_blocks.append(resp.text)
        total_prompt_tokens += resp.prompt_tokens or 0
        total_completion_tokens += resp.completion_tokens or 0

    combined = "\n".join(polished_blocks)
    parsed_segments = _parse_polished_lines(combined, transcript.segments)
    if not parsed_segments:
        # Fallback if the LLM omitted timestamps on a short recording
        parsed_segments = [
            Segment(
                start=0.0,
                end=transcript.duration_seconds,
                speaker="SPEAKER_0",
                text=combined.strip(),
            )
        ]

    polished = Transcript(
        segments=parsed_segments,
        language=transcript.language,
        duration_seconds=transcript.duration_seconds,
        speaker_aliases=dict(transcript.speaker_aliases),
        metadata={**transcript.metadata, "polished": True},
    )
    return polished, {
        "chunks": len(chunks),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


def summarize_transcript(
    transcript_text: str,
    llm: LLMProvider,
    prompt: PromptTemplate,
    hotwords: list[HotwordEntry],
    *,
    title: str = "Recording",
    output_language: str = "auto",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_context_tokens: int = 12000,
    on_token: TokenCallback | None = None,
    on_map_progress: Callable[[int, int], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Summarize transcript in one pass or automatically map-reduce if it exceeds budget."""
    hotwords_block = format_for_llm(hotwords)
    sys_single, usr_single = prompt.render(
        transcript=transcript_text,
        hotwords=hotwords_block,
        title=title,
        output_language=output_language,
    )
    total_est = estimate_tokens(sys_single) + estimate_tokens(usr_single)
    budget = max(2000, max_context_tokens - max_tokens)

    if total_est <= budget:
        resp: LLMResponse = llm.complete(
            sys_single,
            usr_single,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            on_token=on_token,
        )
        return resp.text, {
            "mode": "single_pass",
            "estimated_input_tokens": total_est,
            "model": resp.model,
            "latency_seconds": resp.latency_seconds,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        }

    # Map-reduce for long recordings / small context local models
    lines = transcript_text.splitlines()
    map_budget = max(1500, int(budget * 0.75))
    chunks = split_lines_by_tokens(lines, max_tokens=map_budget, overlap_lines=2)
    partials: list[str] = []
    map_system = (
        "You are summarizing one chronological section of a longer recording. "
        "Extract key points, decisions, action items, technical claims, and speaker attributions "
        "with timestamps. Be thorough and faithful; do not invent anything."
    )
    for idx, chunk in enumerate(chunks, start=1):
        if on_map_progress:
            on_map_progress(idx, len(chunks))
        map_user = (
            f"# Recording: {title} (Part {idx}/{len(chunks)})\n"
            f"# Glossary / Hotwords:\n{hotwords_block}\n\n"
            f"# Transcript Section:\n{chunk}\n\n"
            "Summarize this section concisely in Markdown bullets preserving timestamps and speakers."
        )
        resp = llm.complete(
            map_system,
            map_user,
            model=model,
            temperature=temperature,
            max_tokens=min(2048, max_tokens),
        )
        partials.append(f"### Part {idx}/{len(chunks)}\n{resp.text}")

    combined_notes = "\n\n".join(partials)
    sys_reduce, usr_reduce = prompt.render(
        transcript=combined_notes,
        hotwords=hotwords_block,
        title=title,
        output_language=output_language,
    )
    final_resp = llm.complete(
        sys_reduce,
        usr_reduce,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        on_token=on_token,
    )
    return final_resp.text, {
        "mode": "map_reduce",
        "map_chunks": len(chunks),
        "estimated_input_tokens": total_est,
        "model": final_resp.model,
        "latency_seconds": final_resp.latency_seconds,
    }


def format_transcript_for_llm(transcript: Transcript) -> str:
    if transcript.segments:
        return "\n".join(
            f"[{format_timestamp(seg.start)}] {transcript.resolve_speaker(seg.speaker)}: {seg.text}"
            for seg in transcript.segments
        )
    return ""
