"""Universal OpenAI-compatible /v1/chat/completions LLM client (vLLM, llama-server, Ollama, OpenAI, etc.)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import LLMProvider, LLMResponse, TokenCallback


class OpenAICompatLLMProvider(LLMProvider):
    id = "openai_compat"
    label = "OpenAI-Compatible LLM"

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _effective_opts(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {**self.provider_config, **(options or {})}

    def list_models(self, options: dict[str, Any] | None = None) -> list[str]:
        opts = self._effective_opts(options)
        base_url = str(opts.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/")
        api_key = str(opts.get("api_key", "")).strip()
        timeout = float(opts.get("timeout_seconds", 15))
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url}/models", headers=self._headers(api_key))
            resp.raise_for_status()
            payload = resp.json()
        models = [str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
        return sorted(models)

    def test_connection(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            models = self.list_models(options=options)
            return {
                "ok": True,
                "models": models,
                "detail": f"Connected ({len(models)} models discovered)" if models else "Connected",
            }
        except httpx.HTTPStatusError as exc:
            return {"ok": False, "models": [], "detail": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"ok": False, "models": [], "detail": str(exc)}

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
        opts = self._effective_opts(options)
        base_url = str(opts.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/")
        api_key = str(opts.get("api_key", "")).strip()
        timeout = float(opts.get("timeout_seconds", 300))
        model_name = (model or str(self.llm_config.get("model", "")) or str(opts.get("model", ""))).strip()
        if not model_name:
            raise ValueError("LLM model name is empty. Set llm.model in config or select a model in the UI.")

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        use_stream = on_token is not None
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": use_stream,
        }

        started = time.monotonic()
        url = f"{base_url}/chat/completions"
        headers = self._headers(api_key)

        if not use_stream:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = str((choice.get("message") or {}).get("content") or "").strip()
            usage = data.get("usage") or {}
            return LLMResponse(
                text=text,
                model=str(data.get("model") or model_name),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_seconds=time.monotonic() - started,
            )

        chunks: list[str] = []
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        raw_data = line[5:].strip()
                    else:
                        continue
                    if raw_data == "[DONE]":
                        break
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    delta = ((event.get("choices") or [{}])[0]).get("delta") or {}
                    token = delta.get("content")
                    if token:
                        chunks.append(token)
                        on_token(token)

        return LLMResponse(
            text="".join(chunks).strip(),
            model=model_name,
            latency_seconds=time.monotonic() - started,
        )
