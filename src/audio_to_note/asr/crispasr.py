"""CrispASR CUDA/CPU adapter (recommended for VibeVoice-ASR Q4_K_M GGUF)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..discovery import find_binary
from ..hotwords import HotwordEntry, format_for_asr
from ..merge import normalize_segment
from ..models import parse_timestamp
from . import ASRProvider, ASRResult
from ._common import extract_record_list, read_json_safe


def _parse_crispasr_speaker(value: Any) -> int | str:
    if value is None:
        return "UNKNOWN"
    speaker = str(value).strip().strip("()")
    speaker = re.sub(r"^Speaker\s+", "", speaker, flags=re.IGNORECASE).strip()
    return int(speaker) if speaker.isdigit() else (speaker or "UNKNOWN")


def parse_crispasr_json(path: Path) -> list[dict[str, Any]]:
    payload = read_json_safe(path)
    records = extract_record_list(payload, "transcription", "segments", "results")
    normalized: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", record.get("Content", record.get("content", "")))).strip()
        if not text:
            continue
        offsets = record.get("offsets")
        timestamps = record.get("timestamps")
        start: float | None = None
        end: float | None = None
        if isinstance(offsets, dict):
            try:
                start = float(offsets.get("from")) / 1000.0
                end = float(offsets.get("to")) / 1000.0
            except (TypeError, ValueError):
                start = end = None
        if (start is None or end is None) and isinstance(timestamps, dict):
            try:
                start = parse_timestamp(timestamps.get("from"))
                end = parse_timestamp(timestamps.get("to"))
            except ValueError:
                start = end = None
        if start is None:
            raw_start = record.get("Start", record.get("start", 0.0))
            start = parse_timestamp(raw_start) if raw_start is not None else 0.0
        if end is None:
            raw_end = record.get("End", record.get("end", start))
            end = parse_timestamp(raw_end) if raw_end is not None else start
        normalized.append(
            {
                "Start": start,
                "End": max(start, end),
                "Speaker": _parse_crispasr_speaker(record.get("speaker", record.get("Speaker"))),
                "Content": text,
            }
        )
    return normalized


class CrispASRProvider(ASRProvider):
    id = "crispasr"
    label = "VibeVoice Q4 (CrispASR)"

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
            ["vendor/CrispASR/build-cuda129/bin/crispasr", "crispasr"],
        )
        if not runtime_bin:
            raise FileNotFoundError(
                f"CrispASR binary not found ({opts.get('runtime_bin')}). Build CrispASR or choose another ASR provider."
            )
        model_path = resolve_path(self.project_root, opts.get("model"))
        if not model_path or not model_path.exists():
            raise FileNotFoundError(
                f"CrispASR model file not found: {model_path}. Run scripts/download_models.sh vibevoice-q4 or pick a .gguf model."
            )

        workdir.mkdir(parents=True, exist_ok=True)
        output_base = workdir / "crispasr"
        effective_lang = language if language and language != "auto" else opts.get("language", "auto")
        command = [
            str(runtime_bin),
            "--backend",
            str(opts.get("backend", "vibevoice")),
            "--gpu-backend",
            str(opts.get("gpu_backend", "cuda")),
            "--model",
            str(model_path),
            "--file",
            str(audio_wav),
            "--language",
            str(effective_lang),
            "--threads",
            str(opts.get("threads", 4)),
            "--output-json",
            "--output-file",
            str(output_base),
            "--no-prints",
        ]
        if opts.get("max_new_tokens") is not None:
            command.extend(["--max-new-tokens", str(opts["max_new_tokens"])])
        context = format_for_asr(hotwords or [])
        if context:
            command.extend(["--context", context])

        # Prevent stale CRISPASR_N_GPU_LAYERS from spilling layers to CPU memory.
        env = os.environ.copy()
        env.pop("CRISPASR_N_GPU_LAYERS", None)
        env["CRISPASR_SUPPRESS_ENV_DEPRECATION"] = "1"
        completed = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
        (workdir / "runtime.stderr.log").write_text(completed.stderr or "", encoding="utf-8")

        raw_segments = parse_crispasr_json(output_base.with_suffix(".json"))
        segments = [normalize_segment(item) for item in raw_segments]
        raw_text = " ".join(seg.text for seg in segments)
        return ASRResult(segments=segments, turns=[], raw_text=raw_text, command=command)
