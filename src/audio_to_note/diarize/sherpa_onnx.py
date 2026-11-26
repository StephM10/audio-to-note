"""Local sherpa-onnx offline speaker diarization (no PyTorch or HF token required)."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..models import SpeakerTurn
from . import DiarizerProvider


def _read_pcm16_float32(wav_path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


class SherpaOnnxDiarizerProvider(DiarizerProvider):
    id = "sherpa_onnx"
    label = "sherpa-onnx (Local, No Token)"

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
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                'sherpa-onnx is not installed. Run: pip install -e ".[diarize-onnx]"'
            ) from exc

        opts = {**self.provider_config, **(options or {})}
        seg_model = resolve_path(self.project_root, opts.get("segmentation_model"))
        emb_model = resolve_path(self.project_root, opts.get("embedding_model"))
        if not seg_model or not seg_model.exists():
            raise FileNotFoundError(
                f"sherpa-onnx segmentation model not found: {seg_model}. "
                "Run: scripts/download_models.sh diarization-onnx"
            )
        if not emb_model or not emb_model.exists():
            raise FileNotFoundError(
                f"sherpa-onnx embedding model not found: {emb_model}. "
                "Run: scripts/download_models.sh diarization-onnx"
            )

        num_clusters = int(num_speakers) if num_speakers and num_speakers > 0 else -1
        clustering_cfg = sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=float(opts.get("threshold", 0.5)),
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model)
                ),
                num_threads=int(opts.get("num_threads", 4)),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model),
                num_threads=int(opts.get("num_threads", 4)),
            ),
            clustering=clustering_cfg,
            min_duration_on=float(opts.get("min_duration_on", 0.3)),
            min_duration_off=float(opts.get("min_duration_off", 0.5)),
        )
        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        samples, sample_rate = _read_pcm16_float32(audio_wav_16k)
        if sample_rate != sd.sample_rate:
            raise ValueError(
                f"Expected {sd.sample_rate} Hz audio for sherpa-onnx diarization, got {sample_rate} Hz."
            )
        result = sd.process(samples).sort_by_start_time()
        return [
            SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=f"SPEAKER_{seg.speaker}")
            for seg in result
        ]
