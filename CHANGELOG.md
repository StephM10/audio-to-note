# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-25

### Added
- Modular 5-stage pipeline (`audio-to-note`): Audio Prep -> ASR -> Speaker Diarization -> Optional LLM Polish -> LLM Summarization.
- Pluggable ASR providers: `openai_compat`, `crispasr` (VibeVoice Q4), `audio_cpp` (VibeVoice Q8), `vibeasr_cpp` (BitNet CPU), `faster_whisper` (optional), and custom `command`.
- Pluggable speaker diarization: `none` (ASR labels), `sherpa_onnx` (local ONNX, no HF token required), `pyannote` (local high-accuracy), and custom `command`.
- Universal OpenAI-compatible LLM provider (works with vLLM, llama-server, Ollama, LM Studio, OpenAI, DeepSeek, Gemini, OpenRouter) plus optional local `llama-cli`.
- Automatic map-reduce summarization for long recordings exceeding context budget.
- Editable hotwords/glossary (`#` comments ignored, supports `wrong -> right` rules and notes) shared across ASR and LLM stages.
- Built-in editable prompt presets (`meeting_minutes`, `research_discussion`, `lecture_notes`, `transcript_polish`).
- Click + Rich CLI (`run`, `transcribe`, `summarize`, `serve`, `doctor`, `init`).
- FastAPI + zero-build local Web GUI with drag-and-drop upload, local file browser, live SSE progress/streaming, inline speaker renaming, per-segment speaker/text & Markdown transcript editing, and one-click re-summarization.
- GitHub Actions CI (`ruff` + `pytest`) and tag-triggered Release workflow.
