"""Shared data models for transcripts, providers, and job state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(value: Any) -> float:
    """Parse float seconds or ``HH:MM:SS(.mmm)`` / ``MM:SS`` strings."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"invalid timestamp: {value!r}")
    text = value.strip().replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid timestamp: {value!r}")
    seconds = float(parts[-1])
    minutes = float(parts[-2])
    hours = float(parts[-3]) if len(parts) == 3 else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None = None
    duration_seconds: float = 0.0
    speaker_aliases: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_speaker(self, speaker: str, apply_aliases: bool = True) -> str:
        if not apply_aliases:
            return speaker
        return self.speaker_aliases.get(speaker) or speaker

    def unique_speakers(self) -> list[str]:
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    def to_markdown(self, *, title: str = "Transcript", apply_aliases: bool = True) -> str:
        lines = [f"# {title}", ""]
        for seg in self.segments:
            ts = format_timestamp(seg.start)
            spk = self.resolve_speaker(seg.speaker, apply_aliases=apply_aliases)
            lines.append(f"**[{ts}] {spk}**: {seg.text}  ")
        return "\n".join(lines) + "\n"

    def to_plain_text(self, *, apply_aliases: bool = True) -> str:
        lines = []
        for seg in self.segments:
            ts = format_timestamp(seg.start)
            spk = self.resolve_speaker(seg.speaker, apply_aliases=apply_aliases)
            lines.append(f"[{ts}] {spk}: {seg.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "speaker_aliases": dict(self.speaker_aliases),
            "metadata": dict(self.metadata),
            "segments": [asdict(seg) for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | list[dict[str, Any]]) -> "Transcript":
        if isinstance(payload, list):
            segments = [
                Segment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker=str(item.get("speaker", "SPEAKER_UNKNOWN")),
                    text=str(item.get("text", "")).strip(),
                )
                for item in payload
            ]
            duration = max((seg.end for seg in segments), default=0.0)
            return cls(segments=segments, duration_seconds=duration)
        raw_segments = payload.get("segments", [])
        segments = [
            Segment(
                start=parse_timestamp(item.get("start", item.get("Start", 0.0))),
                end=parse_timestamp(item.get("end", item.get("End", item.get("start", 0.0)))),
                speaker=str(item.get("speaker", item.get("Speaker", "SPEAKER_UNKNOWN"))),
                text=str(item.get("text", item.get("Content", ""))).strip(),
            )
            for item in raw_segments
        ]
        duration = float(payload.get("duration_seconds") or max((seg.end for seg in segments), default=0.0))
        return cls(
            segments=segments,
            language=payload.get("language"),
            duration_seconds=duration,
            speaker_aliases=dict(payload.get("speaker_aliases", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_markdown(cls, text: str) -> "Transcript":
        """Parse a Markdown or plain-text transcript back into segments."""
        pattern = re.compile(
            r"^\s*(?:\*\*)?\[(?P<ts>\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\]\s+"
            r"(?P<speaker>[^:*\n]+?)(?:\*\*)?\s*:\s*(?P<content>.*)$"
        )
        segments: list[Segment] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = pattern.match(stripped)
            if match:
                start = parse_timestamp(match.group("ts"))
                segments.append(
                    Segment(
                        start=start,
                        end=start,
                        speaker=match.group("speaker").strip(),
                        text=match.group("content").strip(),
                    )
                )
            elif segments:
                prev = segments[-1]
                segments[-1] = Segment(prev.start, prev.end, prev.speaker, f"{prev.text} {stripped}".strip())
            else:
                segments.append(Segment(0.0, 0.0, "SPEAKER_UNKNOWN", stripped))
        return cls(segments=segments)


@dataclass
class ProviderStatus:
    id: str
    name: str
    kind: str  # "asr" | "diarizer" | "llm" | "tool"
    available: bool
    status: str  # "ready" | "needs_config" | "missing_dependency" | "missing_model" | "unavailable"
    detail: str
    hint: str = ""
    detected_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    status: str  # "queued" | "running" | "completed" | "failed"
    stage: str
    progress: float
    message: str = ""
    input_file: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    speaker_aliases: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", data["id"])),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            status=str(data.get("status", "unknown")),
            stage=str(data.get("stage", "")),
            progress=float(data.get("progress", 0.0)),
            message=str(data.get("message", "")),
            input_file=str(data.get("input_file", "")),
            config_snapshot=dict(data.get("config_snapshot", {})),
            artifacts=dict(data.get("artifacts", {})),
            metrics=dict(data.get("metrics", {})),
            speaker_aliases=dict(data.get("speaker_aliases", {})),
            error=str(data.get("error", "")),
        )
