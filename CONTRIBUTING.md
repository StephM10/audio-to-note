# Contributing to audio-to-note

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/<your-username>/audio-to-note.git
cd audio-to-note
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

Run tests and linter before submitting a pull request:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -v
```

## Adding a New Provider

- **ASR Provider**: Subclass `ASRProvider` in `src/audio_to_note/asr/`, implement `transcribe_chunk()`, register it in `asr/__init__.py`, and add detection status in `discovery.py`.
- **Diarization Provider**: Subclass `DiarizerProvider` in `src/audio_to_note/diarize/`, implement `diarize()`, and register it in `diarize/__init__.py`.
- **LLM Provider**: Subclass `LLMProvider` in `src/audio_to_note/llm/`, implement `complete()`, and register it in `llm/__init__.py`.

Please include mocked unit tests so CI runs without requiring GPU hardware, model weights, or external API keys.
