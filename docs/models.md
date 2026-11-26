# Recommended Models & Setup Guide

`audio-to-note` never bundles model weights in Git. You can mix and match local models and OpenAI-compatible APIs freely.

---

## 1. Speech Recognition (ASR)

### Option A: `faster-whisper` (Easiest local Python setup)
```bash
pip install -e ".[whisper]"
```
Models download automatically on first run:
- `large-v3-turbo`: Best accuracy/speed balance on GPU (~3 GB VRAM in `float16`) or CPU (`int8`).
- `small` / `medium`: Fast on CPU-only laptops.

### Option B: `VibeVoice-ASR` GGUF (Long-context 60-minute meetings + built-in speaker turns)
Download weights into `models/` using the helper script:
```bash
scripts/download_models.sh vibevoice-q4      # ~4.5 GB (Recommended for 16 GB GPU via CrispASR)
scripts/download_models.sh vibevoice-q8      # ~9.2 GB (via audio.cpp)
scripts/download_models.sh vibevoice-bitnet  # ~1.6 GB (CPU BitNet via VibeASR.cpp)
```
See [docs/local-runtimes.md](local-runtimes.md) for building the C++ runtimes.

### Option C: OpenAI-Compatible Audio API
Works with OpenAI (`whisper-1`, `gpt-4o-mini-transcribe`), Groq (`whisper-large-v3-turbo`), or local servers like `speaches` / `faster-whisper-server` / `whisper.cpp` server.

---

## 2. Speaker Diarization (Optional, Stage 3)

### Option A: `sherpa-onnx` (Recommended lightweight local diarizer — NO HF Token)
```bash
pip install -e ".[diarize-onnx]"
scripts/download_models.sh diarization-onnx
```
Downloads ~35 MB of `.onnx` models (`pyannote-segmentation-3-0` + `3dspeaker_speech_eres2net_base`) into `models/diarization/`. Works fast on both CPU and GPU without PyTorch.

### Option B: `pyannote.audio` (Highest accuracy)
```bash
pip install -e ".[diarize-pyannote]"
```
Accept the user conditions on Hugging Face for `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1` (required by `pyannote.audio` 4.x), then set `HF_TOKEN` in `.env` or Settings.

---

## 3. Summarization & Polish LLM (Stages 4 & 5)

Any OpenAI-compatible `/v1/chat/completions` endpoint works:

| Backend | Default `base_url` | Example Model |
|---|---|---|
| **Ollama** | `http://127.0.0.1:11434/v1` | `qwen2.5:7b`, `qwen2.5:14b`, `llama3.1:8b` |
| **vLLM** | `http://127.0.0.1:8000/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| **llama.cpp (`llama-server`)** | `http://127.0.0.1:8080/v1` | Any local `.gguf` instruct model |
| **LM Studio** | `http://127.0.0.1:1234/v1` | Loaded local model |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` |

### Tips for Long Meetings
- If using a local model with an 8K–16K context window, keep `llm.max_context_tokens` around `8000`–`12000`. `audio-to-note` automatically switches to **Map-Reduce** chunked summarization so nothing gets silently truncated.
