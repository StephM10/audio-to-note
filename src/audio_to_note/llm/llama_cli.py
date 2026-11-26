"""Optional local llama.cpp (llama-cli) GGUF LLM runner."""

from __future__ import annotations

import subprocess
import time
from typing import Any

from ..config import resolve_path
from ..discovery import find_binary
from . import LLMProvider, LLMResponse, TokenCallback


class LlamaCliLLMProvider(LLMProvider):
    id = "llama_cli"
    label = "llama-cli (Local GGUF)"

    def list_models(self, options: dict[str, Any] | None = None) -> list[str]:
        models_dir = self.project_root / "models"
        if not models_dir.is_dir():
            return []
        return sorted(str(p) for p in models_dir.rglob("*.gguf") if "vibe" not in p.name.lower())

    def test_connection(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = {**self.provider_config, **(options or {})}
        binary = find_binary(self.project_root, opts.get("runtime_bin"), ["llama-cli"])
        model_path = resolve_path(self.project_root, opts.get("model"))
        ok = bool(binary and model_path and model_path.exists())
        return {
            "ok": ok,
            "models": [str(model_path)] if model_path and model_path.exists() else [],
            "detail": "Ready" if ok else f"bin={binary or 'missing'}, model={model_path or 'unset'}",
        }

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        on_token: TokenCallback | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        opts = {**self.provider_config, **(options or {})}
        binary = find_binary(
            self.project_root,
            opts.get("runtime_bin"),
            ["llama-cli", "vendor/llama.cpp/build/bin/llama-cli"],
        )
        if not binary:
            raise FileNotFoundError("llama-cli binary not found. Install llama.cpp or use openai_compat.")
        model_path = resolve_path(self.project_root, model or opts.get("model"))
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"GGUF model file not found for llama-cli: {model_path}")

        command = [
            str(binary),
            "-m",
            str(model_path),
            "-sys",
            system_prompt,
            "-p",
            user_prompt,
            "-n",
            str(max_tokens),
            "-c",
            str(opts.get("ctx_size", 16384)),
            "-t",
            str(opts.get("threads", 8)),
            "-ngl",
            str(opts.get("n_gpu_layers", -1)),
            "--temp",
            str(temperature),
            "--no-display-prompt",
            "-no-cnv",
        ]
        started = time.monotonic()
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        text = completed.stdout.strip()
        if on_token and text:
            on_token(text)
        return LLMResponse(
            text=text,
            model=model_path.name,
            latency_seconds=time.monotonic() - started,
        )
