"""Base protocol and registry for pluggable ASR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hotwords import HotwordEntry
from ..models import Segment, SpeakerTurn


@dataclass(frozen=True)
class ASRResult:
    segments: list[Segment]
    turns: list[SpeakerTurn] = field(default_factory=list)
    raw_text: str = ""
    command: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ASRProvider(ABC):
    id: str
    label: str

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self.provider_config: dict[str, Any] = (
            config.get("asr", {}).get("providers", {}).get(self.id, {})
        )

    @property
    def sample_rate(self) -> int:
        return int(self.provider_config.get("sample_rate", 24000))

    @property
    def long_audio_settings(self) -> dict[str, float]:
        defaults = self.config.get("pipeline", {}).get("long_audio", {})
        settings = self.provider_config.get("long_audio", defaults)
        return {
            "single_pass_seconds": float(settings.get("single_pass_max_seconds", 3300)),
            "chunk_seconds": float(settings.get("chunk_seconds", 3000)),
            "overlap_seconds": float(settings.get("overlap_seconds", 45)),
        }

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_wav: Path,
        workdir: Path,
        *,
        language: str = "auto",
        hotwords: list[HotwordEntry] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ASRResult:
        """Transcribe a single prepared mono PCM WAV file."""


def get_asr_provider(provider_id: str, project_root: Path, config: dict[str, Any]) -> ASRProvider:
    from .audio_cpp import AudioCppASRProvider
    from .command import CommandASRProvider
    from .crispasr import CrispASRProvider
    from .faster_whisper import FasterWhisperASRProvider
    from .openai_compat import OpenAICompatASRProvider
    from .vibeasr_cpp import VibeASRCppProvider

    registry: dict[str, type[ASRProvider]] = {
        "crispasr": CrispASRProvider,
        "audio_cpp": AudioCppASRProvider,
        "vibeasr_cpp": VibeASRCppProvider,
        "openai_compat": OpenAICompatASRProvider,
        "faster_whisper": FasterWhisperASRProvider,
        "command": CommandASRProvider,
    }
    cls = registry.get(provider_id)
    if cls is None:
        raise ValueError(f"Unsupported ASR provider: {provider_id!r}. Available: {sorted(registry)}")
    return cls(project_root, config)
