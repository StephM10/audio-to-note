"""Optional local faster-whisper Python ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hotwords import HotwordEntry, format_for_asr
from ..models import Segment
from . import ASRProvider, ASRResult

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


class FasterWhisperASRProvider(ASRProvider):
    id = "faster_whisper"
    label = "faster-whisper (Local Python)"

    def transcribe_chunk(
        self,
        audio_wav: Path,
        workdir: Path,
        *,
        language: str = "auto",
        hotwords: list[HotwordEntry] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ASRResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                'faster-whisper is not installed. Run: pip install -e ".[whisper]"'
            ) from exc

        opts = {**self.provider_config, **(options or {})}
        model_name = str(opts.get("model", "large-v3-turbo"))
        device = str(opts.get("device", "auto"))
        compute_type = str(opts.get("compute_type", "auto"))
        if device == "auto":
            device = "cuda" if self._has_cuda() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        cache_key = (model_name, device, compute_type)
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _MODEL_CACHE[cache_key] = model

        initial_prompt = format_for_asr(hotwords or []) or None
        fw_lang = None if not language or language == "auto" else language
        raw_segments, info = model.transcribe(
            str(audio_wav),
            language=fw_lang,
            beam_size=int(opts.get("beam_size", 5)),
            vad_filter=bool(opts.get("vad_filter", True)),
            initial_prompt=initial_prompt,
        )
        segments: list[Segment] = []
        for seg in raw_segments:
            text = str(getattr(seg, "text", "")).strip()
            if not text:
                continue
            start = float(getattr(seg, "start", 0.0))
            end = max(start, float(getattr(seg, "end", start)))
            segments.append(Segment(start=start, end=end, speaker="SPEAKER_UNKNOWN", text=text))

        return ASRResult(
            segments=segments,
            turns=[],
            raw_text=" ".join(seg.text for seg in segments),
            metadata={"model": model_name, "detected_language": getattr(info, "language", None)},
        )

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False
