"""Base protocol and registry for speaker diarization providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import SpeakerTurn


class DiarizerProvider(ABC):
    id: str
    label: str

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self.provider_config: dict[str, Any] = (
            config.get("diarization", {}).get("providers", {}).get(self.id, {})
        )

    @abstractmethod
    def diarize(
        self,
        audio_wav_16k: Path,
        workdir: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[SpeakerTurn]:
        """Return speaker turns for a 16 kHz mono WAV file."""


class NoneDiarizerProvider(DiarizerProvider):
    id = "none"
    label = "Off (Keep ASR Speaker Labels)"

    def diarize(
        self,
        audio_wav_16k: Path,
        workdir: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[SpeakerTurn]:
        return []


def get_diarizer_provider(
    provider_id: str, project_root: Path, config: dict[str, Any]
) -> DiarizerProvider:
    from .command import CommandDiarizerProvider
    from .pyannote import PyannoteDiarizerProvider
    from .sherpa_onnx import SherpaOnnxDiarizerProvider

    registry: dict[str, type[DiarizerProvider]] = {
        "none": NoneDiarizerProvider,
        "sherpa_onnx": SherpaOnnxDiarizerProvider,
        "pyannote": PyannoteDiarizerProvider,
        "command": CommandDiarizerProvider,
    }
    cls = registry.get(provider_id)
    if cls is None:
        raise ValueError(
            f"Unsupported diarization provider: {provider_id!r}. Available: {sorted(registry)}"
        )
    return cls(project_root, config)
