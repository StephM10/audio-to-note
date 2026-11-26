"""Microsoft VibeASR.cpp BitNet CPU adapter."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ..audio import wav_duration
from ..config import resolve_path
from ..discovery import find_binary
from ..hotwords import HotwordEntry, format_for_asr
from ..merge import normalize_segment
from . import ASRProvider, ASRResult
from ._common import extract_record_list


def parse_vibeasr_segments(stdout: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*\[\s*(?P<start>[^\]]+?)\s*-\s*(?P<end>[^\]]+?)\s*\]\s*"
        r"(?:Speaker\s+(?P<speaker>[^:]+):\s*)?(?P<text>.*)$"
    )
    for line in stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            start_value = float(match.group("start").strip())
            end_value = float(match.group("end").strip())
        except ValueError:
            continue
        speaker = match.group("speaker")
        speaker_value: int | str = (
            int(speaker) if speaker and speaker.strip().isdigit() else (speaker or "0")
        )
        text = match.group("text").strip()
        if text:
            segments.append(
                {"Start": start_value, "End": end_value, "Speaker": speaker_value, "Content": text}
            )
    if segments:
        return segments

    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        parsed = extract_record_list(payload, "segments", "results")
        if parsed:
            return parsed
    return []


class VibeASRCppProvider(ASRProvider):
    id = "vibeasr_cpp"
    label = "VibeVoice BitNet CPU (VibeASR.cpp)"

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
        runtime_bin = find_binary(
            self.project_root,
            opts.get("runtime_bin"),
            ["vendor/VibeASR.cpp/build/bin/asr_infer", "asr_infer"],
        )
        if not runtime_bin:
            raise FileNotFoundError(f"VibeASR.cpp binary not found ({opts.get('runtime_bin')}).")
        vae_model = resolve_path(self.project_root, opts.get("vae_model"))
        lm_model = resolve_path(self.project_root, opts.get("lm_model"))
        if not vae_model or not vae_model.exists() or not lm_model or not lm_model.exists():
            raise FileNotFoundError("VibeASR.cpp BitNet model files not found.")

        workdir.mkdir(parents=True, exist_ok=True)
        command = [
            str(runtime_bin),
            "--vae-model",
            str(vae_model),
            "--lm-model",
            str(lm_model),
            "--audio",
            str(audio_wav),
            "-t",
            str(opts.get("threads", 16)),
            "-c",
            str(opts.get("context_size", 16384)),
            "--prompt-format",
            str(opts.get("prompt_format", "json")),
            "--sample-rate",
            str(opts.get("sample_rate", 24000)),
            "--max-tokens",
            str(opts.get("max_tokens", 4096)),
        ]
        if opts.get("greedy", True):
            command.append("--greedy")
        context = format_for_asr(hotwords or [])
        if context:
            command.extend(["--context", context])

        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        raw_segments = parse_vibeasr_segments(completed.stdout)
        if not raw_segments and completed.stdout.strip():
            raw_segments = [
                {
                    "Start": 0.0,
                    "End": wav_duration(audio_wav),
                    "Speaker": "UNKNOWN",
                    "Content": completed.stdout.strip(),
                }
            ]
        segments = [normalize_segment(item) for item in raw_segments]
        raw_text = " ".join(seg.text for seg in segments)
        return ASRResult(segments=segments, turns=[], raw_text=raw_text, command=command)
