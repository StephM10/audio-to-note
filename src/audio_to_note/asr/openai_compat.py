"""OpenAI-compatible /v1/audio/transcriptions ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..audio import wav_duration
from ..hotwords import HotwordEntry, format_for_asr
from ..models import Segment
from . import ASRProvider, ASRResult


class OpenAICompatASRProvider(ASRProvider):
    id = "openai_compat"
    label = "OpenAI-Compatible Audio API"

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
        base_url = str(opts.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        api_key = str(opts.get("api_key", "")).strip()
        model = str(opts.get("model", "whisper-1")).strip() or "whisper-1"
        timeout = float(opts.get("timeout_seconds", 600))

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
        }
        if language and language != "auto":
            data["language"] = language
        prompt = format_for_asr(hotwords or [])
        if prompt:
            data["prompt"] = prompt

        url = f"{base_url}/audio/transcriptions"
        with audio_wav.open("rb") as audio_handle:
            files = {"file": (audio_wav.name, audio_handle, "audio/wav")}
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()
                payload = response.json()

        raw_text = str(payload.get("text", "")).strip()
        raw_segments = payload.get("segments") or []
        segments: list[Segment] = []
        for item in raw_segments:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0.0))
            end = max(start, float(item.get("end", start)))
            speaker = str(item.get("speaker") or "SPEAKER_UNKNOWN")
            segments.append(Segment(start=start, end=end, speaker=speaker, text=text))

        if not segments and raw_text:
            duration = float(payload.get("duration") or wav_duration(audio_wav))
            segments = [Segment(start=0.0, end=duration, speaker="SPEAKER_UNKNOWN", text=raw_text)]

        return ASRResult(
            segments=segments,
            turns=[],
            raw_text=raw_text or " ".join(seg.text for seg in segments),
            metadata={"model": model, "base_url": base_url},
        )
