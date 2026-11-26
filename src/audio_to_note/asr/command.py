"""Generic external CLI command ASR adapter."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..audio import wav_duration
from ..hotwords import HotwordEntry, format_for_asr
from ..merge import normalize_segment
from ..models import Segment
from . import ASRProvider, ASRResult
from ._common import extract_record_list, read_json_safe


class CommandASRProvider(ASRProvider):
    id = "command"
    label = "Custom CLI Command"

    def transcribe_chunk(
        self,
        audio_wav: Path,
        workdir: Path,
        *,
        language: str = "auto",
        hotwords: list[HotwordEntry] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ASRResult:
        opts = {**self.provider_config, **(options or {})}
        template = opts.get("command", "")
        if not template:
            raise ValueError("Custom CLI ASR command is not configured (asr.providers.command.command).")

        workdir.mkdir(parents=True, exist_ok=True)
        output_json = workdir / "output.json"
        hotwords_str = format_for_asr(hotwords or [])
        mapping = {
            "audio": str(audio_wav),
            "output_json": str(output_json),
            "language": language,
            "hotwords": hotwords_str,
            "workdir": str(workdir),
        }

        if isinstance(template, str):
            raw_parts = shlex.split(template)
        else:
            raw_parts = [str(part) for part in template]
        command = [part.format(**mapping) for part in raw_parts]

        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        payload: Any = read_json_safe(output_json) if output_json.exists() else None
        if payload is None and completed.stdout.strip().startswith(("[", "{")):
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = None

        records = extract_record_list(payload, "segments", "transcription", "results")
        if records:
            segments = [normalize_segment(item) for item in records]
        elif completed.stdout.strip():
            segments = [
                Segment(
                    start=0.0,
                    end=wav_duration(audio_wav),
                    speaker="SPEAKER_UNKNOWN",
                    text=completed.stdout.strip(),
                )
            ]
        else:
            segments = []

        return ASRResult(
            segments=segments,
            turns=[],
            raw_text=" ".join(seg.text for seg in segments),
            command=command,
        )
