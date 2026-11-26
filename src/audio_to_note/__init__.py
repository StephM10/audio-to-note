"""audio-to-note: Modular audio/video transcription, diarization, and AI note-taking."""

from __future__ import annotations

__version__ = "0.1.0"

from .models import JobRecord, ProviderStatus, Segment, SpeakerTurn, Transcript

__all__ = [
    "__version__",
    "JobRecord",
    "ProviderStatus",
    "Segment",
    "SpeakerTurn",
    "Transcript",
]
