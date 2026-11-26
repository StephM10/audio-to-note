"""Local pyannote.audio high-accuracy speaker diarization."""

from __future__ import annotations

import inspect
import wave
from array import array
from pathlib import Path
from typing import Any

from ..merge import _speaker
from ..models import SpeakerTurn
from . import DiarizerProvider

_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


def _resolve_hf_token(configured: Any) -> str | None:
    token = str(configured or "").strip()
    if token:
        return token
    cached = Path("~/.cache/huggingface/token").expanduser()
    if cached.is_file():
        val = cached.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def _load_wav_tensor(wav_path: Path) -> dict[str, Any]:
    """Load PCM WAV via stdlib wave to bypass torchcodec/torchaudio ABI issues."""
    import torch

    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV for pyannote diarization, got {sampwidth * 8}-bit.")
    samples = array("h")
    samples.frombytes(frames)
    waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
    if channels > 1:
        waveform = waveform.view(-1, channels).mean(dim=1)
    waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


class PyannoteDiarizerProvider(DiarizerProvider):
    id = "pyannote"
    label = "pyannote.audio (Local High-Accuracy)"

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
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                'pyannote.audio is not installed. Run: pip install -e ".[diarize-pyannote]"'
            ) from exc

        opts = {**self.provider_config, **(options or {})}
        model_id = str(opts.get("model", "pyannote/speaker-diarization-3.1"))
        token = _resolve_hf_token(opts.get("hf_token"))
        device_name = str(opts.get("device", "auto"))
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = (model_id, device_name)
        pipeline = _PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            kwargs: dict[str, Any] = {}
            if token:
                sig_params = inspect.signature(Pipeline.from_pretrained).parameters
                token_kw = "token" if "token" in sig_params else "use_auth_token"
                kwargs[token_kw] = token
            try:
                pipeline = Pipeline.from_pretrained(model_id, **kwargs)
            except Exception as exc:
                msg = str(exc)
                if any(code in msg for code in ("401", "403", "gated", "Unauthorized", "authenticate")):
                    raise RuntimeError(
                        "Hugging Face authentication failed for pyannote gated models. "
                        "1) Accept terms on https://huggingface.co/pyannote/speaker-diarization-3.1, "
                        "https://huggingface.co/pyannote/segmentation-3.0, and "
                        "https://huggingface.co/pyannote/speaker-diarization-community-1; "
                        "2) Set HF_TOKEN=hf_... in .env (or run `hf auth login`).\n"
                        f"Original error: {msg}"
                    ) from exc
                raise
            pipeline.to(torch.device(device_name))
            _PIPELINE_CACHE[cache_key] = pipeline

        call_kwargs: dict[str, Any] = {}
        if num_speakers:
            call_kwargs["num_speakers"] = int(num_speakers)
        if min_speakers:
            call_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            call_kwargs["max_speakers"] = int(max_speakers)

        audio_input = _load_wav_tensor(audio_wav_16k)
        output = pipeline(audio_input, **call_kwargs)
        # pyannote.audio 4.x returns DiarizeOutput(speaker_diarization=Annotation); 3.x returns Annotation directly.
        annotation = getattr(output, "speaker_diarization", output)
        turns: list[SpeakerTurn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=_speaker(label),
                )
            )
        return turns
