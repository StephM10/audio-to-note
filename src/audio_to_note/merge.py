"""Normalize, stitch overlapping ASR chunks, and align speaker turns."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .models import Segment, SpeakerTurn, parse_timestamp


def _value(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _speaker(value: Any) -> str:
    if value is None:
        return "SPEAKER_UNKNOWN"
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return f"SPEAKER_{int(value)}"
    text = str(value).strip().upper().replace(" ", "_")
    return text if text.startswith("SPEAKER_") else f"SPEAKER_{text}"


def normalize_segment(data: dict[str, Any]) -> Segment:
    start = parse_timestamp(_value(data, "start", "Start", "start_time"))
    end = parse_timestamp(_value(data, "end", "End", "end_time"))
    text = str(_value(data, "text", "Content", "content", default="")).strip()
    if end < start:
        raise ValueError(f"segment end precedes start: {data!r}")
    return Segment(start, end, _speaker(_value(data, "speaker", "Speaker", "speaker_id")), text)


def _overlap(left: Segment | SpeakerTurn, right: Segment | SpeakerTurn) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.lower().split()) == " ".join(right.lower().split())


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _is_mostly_cjk(text: str) -> bool:
    stripped = "".join(text.split())
    if not stripped:
        return False
    cjk_count = len(_CJK_RE.findall(stripped))
    return cjk_count / len(stripped) >= 0.35


def _trim_text_overlap(previous: str, current: str, minimum_tokens: int = 2, minimum_chars: int = 4) -> str:
    """Trim a repeated suffix/prefix across overlapping chunk boundaries (Latin & CJK)."""
    if _is_mostly_cjk(previous) or _is_mostly_cjk(current):
        prev_clean = previous.strip()
        curr_clean = current.strip()
        if len(prev_clean) < minimum_chars or len(curr_clean) < minimum_chars:
            return current
        limit = min(len(prev_clean), len(curr_clean))
        for count in range(limit, minimum_chars - 1, -1):
            if prev_clean[-count:].casefold() == curr_clean[:count].casefold():
                return curr_clean[count:].lstrip(" ，。、；：！？,.;:!?")
        return current

    previous_tokens = previous.split()
    current_tokens = current.split()
    if len(previous_tokens) < minimum_tokens or len(current_tokens) < minimum_tokens:
        return current
    limit = min(len(previous_tokens), len(current_tokens))
    for count in range(limit, minimum_tokens - 1, -1):
        if [token.casefold() for token in previous_tokens[-count:]] == [
            token.casefold() for token in current_tokens[:count]
        ]:
            return " ".join(current_tokens[count:]).strip()
    return current


def merge_chunk_segments(
    chunks: Iterable[tuple[float, Iterable[dict[str, Any] | Segment]]],
) -> list[Segment]:
    """Merge chunk-local segments into one timeline and speaker namespace."""
    merged: list[Segment] = []
    next_speaker = 0
    for offset, raw_segments in sorted(chunks, key=lambda item: item[0]):
        local_to_global: dict[str, str] = {}
        normalized: list[Segment] = []
        for raw in raw_segments:
            segment = raw if isinstance(raw, Segment) else normalize_segment(raw)
            normalized.append(
                Segment(segment.start + offset, segment.end + offset, segment.speaker, segment.text)
            )
        for segment in sorted(normalized, key=lambda item: (item.start, item.end)):
            votes: defaultdict[str, float] = defaultdict(float)
            for previous in merged:
                amount = _overlap(segment, previous)
                if amount:
                    votes[previous.speaker] += amount
            if segment.speaker == "SPEAKER_UNKNOWN":
                speaker = segment.speaker
                local_to_global[segment.speaker] = speaker
            elif segment.speaker in local_to_global:
                speaker = local_to_global[segment.speaker]
            elif votes:
                speaker = max(votes, key=votes.get)
                local_to_global[segment.speaker] = speaker
            else:
                speaker = f"SPEAKER_{next_speaker}"
                next_speaker += 1
                local_to_global[segment.speaker] = speaker
            candidate = Segment(segment.start, segment.end, speaker, segment.text)
            if candidate.text:
                overlaps = [previous for previous in merged if _overlap(candidate, previous) > 0 and previous.text]
                if overlaps:
                    previous = max(overlaps, key=lambda item: item.end)
                    candidate = Segment(
                        candidate.start,
                        candidate.end,
                        candidate.speaker,
                        _trim_text_overlap(previous.text, candidate.text),
                    )
            if not candidate.text:
                continue
            duplicate = any(
                _same_text(candidate.text, previous.text)
                and _overlap(candidate, previous)
                >= 0.5 * min(candidate.end - candidate.start, previous.end - previous.start)
                for previous in merged
                if candidate.text and previous.text
            )
            if not duplicate:
                merged.append(candidate)
    return sorted(merged, key=lambda item: (item.start, item.end))


def align_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Attach speaker labels from diarization ``turns`` to transcript ``segments`` by max overlap."""
    if not turns:
        return segments
    result: list[Segment] = []
    for seg in segments:
        votes: defaultdict[str, float] = defaultdict(float)
        for turn in turns:
            ov = _overlap(seg, turn)
            if ov > 0:
                votes[turn.speaker] += ov
        if votes:
            best_speaker = max(votes, key=votes.get)
            result.append(Segment(seg.start, seg.end, best_speaker, seg.text))
        else:
            result.append(seg)
    return result
