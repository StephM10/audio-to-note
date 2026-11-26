"""FFmpeg audio probing, conversion, and deterministic long-audio chunk planning."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Any


def probe_duration(input_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def probe_media_info(input_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    fmt = payload.get("format", {})
    return {
        "duration_seconds": float(fmt.get("duration", 0.0) or 0.0),
        "format_name": fmt.get("format_name", ""),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "streams": len(payload.get("streams", [])),
    }


def wav_duration(audio: Path) -> float:
    with wave.open(str(audio), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / rate if rate else 0.0


def extract_wav(
    input_path: Path,
    output_path: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    sample_rate: int = 24000,
) -> Path:
    """Convert any audio/video file (or slice) into mono 16-bit PCM WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        command.extend(["-ss", str(start)])
    if end is not None:
        duration = end - (start or 0.0)
        command.extend(["-t", str(duration)])
    command.extend(
        [
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)
    return output_path


def plan_chunks(
    duration_seconds: float,
    *,
    single_pass_seconds: float = 55 * 60,
    chunk_seconds: float = 50 * 60,
    overlap_seconds: float = 45.0,
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` windows, bounded and overlapping.

    A recording shorter than ``single_pass_seconds`` is kept intact so the
    ASR model can use its global speaker context. Longer recordings use
    bounded windows with a small overlap for stitching.
    """
    duration_seconds = float(duration_seconds)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if chunk_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("chunk_seconds must be positive and overlap non-negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")
    if duration_seconds <= single_pass_seconds:
        return [(0.0, duration_seconds)]

    step = chunk_seconds - overlap_seconds
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        chunks.append((round(start, 6), round(end, 6)))
        if end >= duration_seconds:
            break
        start += step
    return chunks
