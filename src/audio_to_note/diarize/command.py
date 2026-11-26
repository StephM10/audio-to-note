"""Generic CLI command speaker diarizer adapter."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..merge import _speaker
from ..models import SpeakerTurn, parse_timestamp
from . import DiarizerProvider


class CommandDiarizerProvider(DiarizerProvider):
    id = "command"
    label = "Custom Diarizer CLI"

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
        opts = {**self.provider_config, **(options or {})}
        template = opts.get("command", "")
        if not template:
            raise ValueError("Custom diarizer command is not configured.")

        workdir.mkdir(parents=True, exist_ok=True)
        output_json = workdir / "turns.json"
        mapping = {
            "audio": str(audio_wav_16k),
            "output_json": str(output_json),
            "num_speakers": str(num_speakers or ""),
            "min_speakers": str(min_speakers or ""),
            "max_speakers": str(max_speakers or ""),
            "workdir": str(workdir),
        }
        parts = shlex.split(template) if isinstance(template, str) else [str(p) for p in template]
        command = [p.format(**mapping) for p in parts]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)

        raw = output_json.read_text(encoding="utf-8") if output_json.exists() else completed.stdout
        payload = json.loads(raw or "[]")
        records = payload.get("turns", payload) if isinstance(payload, dict) else payload
        turns: list[SpeakerTurn] = []
        for item in records:
            start = parse_timestamp(item.get("start", item.get("Start", 0.0)))
            end = parse_timestamp(item.get("end", item.get("End", start)))
            spk = _speaker(item.get("speaker", item.get("Speaker", "UNKNOWN")))
            turns.append(SpeakerTurn(start=start, end=max(start, end), speaker=spk))
        return turns
