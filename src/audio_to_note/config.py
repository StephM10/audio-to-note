"""Configuration loading, environment variable expansion, and persistence."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "outputs_dir": "outputs",
        "hotwords_file": "",
        "default_prompt": "meeting_minutes",
        "polish_enabled": False,
        "summarize_enabled": True,
        "output_language": "auto",
        "sample_rate": 24000,
        "long_audio": {
            "single_pass_max_seconds": 3300,
            "chunk_seconds": 3000,
            "overlap_seconds": 45,
        },
    },
    "asr": {
        "provider": "crispasr",
        "language": "auto",
        "providers": {
            "crispasr": {
                "runtime_bin": "vendor/CrispASR/build-cuda129/bin/crispasr",
                "model": "models/vibevoice-asr-q4_k.gguf",
                "backend": "vibevoice",
                "gpu_backend": "cuda",
                "threads": 4,
                "sample_rate": 24000,
                "long_audio": {
                    "single_pass_max_seconds": 3300,
                    "chunk_seconds": 3000,
                    "overlap_seconds": 45,
                },
            },
            "audio_cpp": {
                "runtime_bin": "vendor/audio.cpp/build/bin/audiocpp_cli",
                "model": "models/vibevoice-asr-q8_0.gguf",
                "backend": "cuda",
                "family": "vibevoice_asr",
                "sample_rate": 24000,
                "decode": {"temperature": 0.0, "num_beams": 1, "max_tokens": 8192},
                "session_options": {
                    "vibevoice_asr.weight_type": "q8_0",
                    "vibevoice_asr.tokenizer_weight_type": "q8_0",
                    "vibevoice_asr.connector_weight_type": "q8_0",
                    "vibevoice_asr.decoder_weight_type": "q8_0",
                    "vibevoice_asr.kv_cache_type": "bf16",
                },
                "long_audio": {
                    "single_pass_max_seconds": 3300,
                    "chunk_seconds": 3000,
                    "overlap_seconds": 45,
                },
            },
            "vibeasr_cpp": {
                "runtime_bin": "vendor/VibeASR.cpp/build/bin/asr_infer",
                "vae_model": "models/vibevoice-bitnet/vibeasr-vae-encoder-i8_s.gguf",
                "lm_model": "models/vibevoice-bitnet/vibeasr-lm-i2_s-embed-q6_k.gguf",
                "backend": "cpu",
                "threads": 16,
                "sample_rate": 24000,
                "prompt_format": "json",
                "greedy": True,
                "context_size": 16384,
                "max_tokens": 4096,
                "long_audio": {
                    "single_pass_max_seconds": 120,
                    "chunk_seconds": 120,
                    "overlap_seconds": 15,
                },
            },
            "openai_compat": {
                "base_url": "${A2N_ASR_BASE_URL:-https://api.openai.com/v1}",
                "api_key": "${A2N_ASR_API_KEY:-${OPENAI_API_KEY:-}}",
                "model": "${A2N_ASR_MODEL:-whisper-1}",
                "sample_rate": 16000,
                "timeout_seconds": 600,
                "long_audio": {
                    "single_pass_max_seconds": 600,
                    "chunk_seconds": 600,
                    "overlap_seconds": 15,
                },
            },
            "faster_whisper": {
                "model": "large-v3-turbo",
                "device": "auto",
                "compute_type": "auto",
                "beam_size": 5,
                "vad_filter": True,
                "sample_rate": 16000,
                "long_audio": {
                    "single_pass_max_seconds": 3600,
                    "chunk_seconds": 3000,
                    "overlap_seconds": 30,
                },
            },
            "command": {
                "command": "",
                "sample_rate": 16000,
                "long_audio": {
                    "single_pass_max_seconds": 3300,
                    "chunk_seconds": 3000,
                    "overlap_seconds": 45,
                },
            },
        },
    },
    "diarization": {
        "provider": "none",
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "providers": {
            "sherpa_onnx": {
                "segmentation_model": "models/diarization/pyannote-segmentation-3-0/model.onnx",
                "embedding_model": "models/diarization/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                "threshold": 0.5,
                "num_threads": 4,
            },
            "pyannote": {
                "model": "pyannote/speaker-diarization-3.1",
                "hf_token": "${HF_TOKEN:-}",
                "device": "auto",
            },
            "command": {
                "command": "",
            },
        },
    },
    "llm": {
        "provider": "openai_compat",
        "preset": "ollama",
        "model": "${A2N_LLM_MODEL:-qwen2.5:7b}",
        "temperature": 0.2,
        "max_tokens": 4096,
        "max_context_tokens": 12000,
        "providers": {
            "openai_compat": {
                "base_url": "${A2N_LLM_BASE_URL:-http://127.0.0.1:11434/v1}",
                "api_key": "${A2N_LLM_API_KEY:-${OPENAI_API_KEY:-ollama}}",
                "timeout_seconds": 300,
            },
            "llama_cli": {
                "runtime_bin": "llama-cli",
                "model": "",
                "n_gpu_layers": -1,
                "ctx_size": 16384,
                "threads": 8,
            },
        },
    },
}

LLM_PRESETS: dict[str, dict[str, str]] = {
    "ollama": {"label": "Ollama (Local)", "base_url": "http://127.0.0.1:11434/v1", "default_model": "qwen2.5:7b"},
    "vllm": {"label": "vLLM (Local / Server)", "base_url": "http://127.0.0.1:8000/v1", "default_model": ""},
    "llama_server": {"label": "llama.cpp server", "base_url": "http://127.0.0.1:8080/v1", "default_model": ""},
    "lm_studio": {"label": "LM Studio", "base_url": "http://127.0.0.1:1234/v1", "default_model": ""},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat"},
    "gemini": {
        "label": "Google Gemini (OpenAI-compat)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-flash",
    },
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "default_model": ""},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_dotenv_file(env_path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not env_path.exists() or not env_path.is_file():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            loaded[key] = val
            os.environ.setdefault(key, val)
    return loaded


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None and env_val != "":
            return env_val
        if default is not None:
            return expand_env_vars(default)
        return ""

    return _ENV_PATTERN.sub(_replace, value)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "config").is_dir():
            return candidate
    return current


def resolve_path(project_root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (project_root / p)


def resolve_hotwords_path(project_root: Path, configured: str | None = None) -> Path | None:
    if configured:
        p = resolve_path(project_root, configured)
        if p and p.exists():
            return p
    for candidate in (
        project_root / "config" / "hotwords.local.txt",
        project_root / "config" / "hotwords.txt",
        project_root / "config" / "hotwords.example.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def load_config(
    config_path: str | Path | None = None,
    *,
    project_root: Path | None = None,
    expand_env: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve() if project_root else find_project_root()
    load_dotenv_file(root / ".env")

    selected: Path | None = None
    if config_path is not None:
        selected = Path(config_path).expanduser().resolve()
    else:
        for candidate in (root / "config" / "config.yaml", root / "config" / "config.example.yaml"):
            if candidate.exists():
                selected = candidate
                break

    loaded: dict[str, Any] = {}
    if selected and selected.exists():
        text = selected.read_text(encoding="utf-8")
        if selected.suffix.lower() == ".json":
            loaded = json.loads(text)
        else:
            loaded = yaml.safe_load(text) or {}

    merged = deep_merge(DEFAULT_CONFIG, loaded)
    if expand_env:
        merged = expand_env_vars(merged)
    merged["_meta"] = {
        "project_root": str(root),
        "config_path": str(selected) if selected else "",
    }
    return merged


def save_user_config(config_data: dict[str, Any], target_path: Path) -> Path:
    clean = deepcopy(config_data)
    clean.pop("_meta", None)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(clean, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target_path


def mask_secrets(config_data: dict[str, Any]) -> dict[str, Any]:
    masked = deepcopy(config_data)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if any(tok in k.lower() for tok in ("api_key", "token", "secret")) and isinstance(v, str) and v:
                    if v.startswith("${"):
                        out[k] = v
                    elif len(v) <= 8:
                        out[k] = "****"
                    else:
                        out[k] = f"{v[:4]}****{v[-4:]}"
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(masked)
