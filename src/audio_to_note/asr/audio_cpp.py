"""audio.cpp CUDA/CPU adapter for VibeVoice-ASR Q8 GGUF."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..discovery import find_binary
from ..hotwords import HotwordEntry, format_for_asr
from ..merge import normalize_segment
from ..models import SpeakerTurn
from . import ASRProvider, ASRResult
from ._common import extract_record_list, read_json_safe


def normalize_audio_cpp_records(records: list[dict[str, Any]], sample_rate: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if "Start" not in item and "start" not in item and "start_time" not in item:
            if "start_sample" in item:
                item["Start"] = float(item["start_sample"]) / sample_rate
        if "End" not in item and "end" not in item and "end_time" not in item:
            if "end_sample" in item:
                item["End"] = float(item["end_sample"]) / sample_rate
        if "Speaker" not in item and "speaker" not in item and "speaker_id" in item:
            item["Speaker"] = item["speaker_id"]
        if "Content" not in item and "text" in item:
            item["Content"] = item["text"]
        normalized.append(item)
    return normalized


class AudioCppASRProvider(ASRProvider):
    id = "audio_cpp"
    label = "VibeVoice Q8 (audio.cpp)"

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
            ["vendor/audio.cpp/build/bin/audiocpp_cli", "audiocpp_cli"],
        )
        if not runtime_bin:
            raise FileNotFoundError(f"audio.cpp binary not found ({opts.get('runtime_bin')}).")
        model_path = resolve_path(self.project_root, opts.get("model"))
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"audio.cpp model file not found: {model_path}.")

        workdir.mkdir(parents=True, exist_ok=True)
        segments_path = workdir / "segments.json"
        turns_path = workdir / "turns.json"
        text_path = workdir / "text.txt"
        sample_rate = int(opts.get("sample_rate", 24000))
        decode = opts.get("decode", {})
        effective_lang = language if language and language != "auto" else opts.get("language", "auto")

        command = [
            str(runtime_bin),
            "--task",
            "asr",
            "--family",
            str(opts.get("family", "vibevoice_asr")),
            "--model",
            str(model_path),
            "--backend",
            str(opts.get("backend", "cuda")),
            "--audio",
            str(audio_wav),
            "--language",
            str(effective_lang),
            "--temperature",
            str(decode.get("temperature", 0.0)),
            "--num-beams",
            str(decode.get("num_beams", 1)),
            "--max-tokens",
            str(decode.get("max_tokens", 8192)),
            "--audio-chunk-mode",
            "none",
            "--text-out",
            str(text_path),
            "--segments-out",
            str(segments_path),
            "--turns-out",
            str(turns_path),
        ]
        context = format_for_asr(hotwords or [])
        if context:
            command.extend(["--text", context])
        for key, value in opts.get("session_options", {}).items():
            command.extend(["--session-option", f"{key}={value}"])

        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        raw_text = text_path.read_text(encoding="utf-8") if text_path.exists() else completed.stdout
        raw_segments = normalize_audio_cpp_records(
            extract_record_list(read_json_safe(segments_path), "segments", "results"),
            sample_rate,
        )
        raw_turns = normalize_audio_cpp_records(
            extract_record_list(read_json_safe(turns_path), "turns", "segments"),
            sample_rate,
        )
        segments = [normalize_segment(item) for item in raw_segments]
        turns = [
            SpeakerTurn(s.start, s.end, s.speaker)
            for s in (normalize_segment({**item, "Content": ""}) for item in raw_turns)
        ]
        return ASRResult(segments=segments, turns=turns, raw_text=raw_text.strip(), command=command)
