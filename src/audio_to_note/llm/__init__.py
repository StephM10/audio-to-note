"""Base protocol and registry for pluggable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float = 0.0


TokenCallback = Callable[[str], None]


class LLMProvider(ABC):
    id: str
    label: str

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self.llm_config: dict[str, Any] = config.get("llm", {})
        self.provider_config: dict[str, Any] = (
            self.llm_config.get("providers", {}).get(self.id, {})
        )

    @abstractmethod
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
        """Run a chat completion and return the full response."""

    def list_models(self, options: dict[str, Any] | None = None) -> list[str]:
        return []

    def test_connection(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        models = self.list_models(options=options)
        return {"ok": True, "models": models}


def get_llm_provider(provider_id: str, project_root: Path, config: dict[str, Any]) -> LLMProvider:
    from .llama_cli import LlamaCliLLMProvider
    from .openai_compat import OpenAICompatLLMProvider

    registry: dict[str, type[LLMProvider]] = {
        "openai_compat": OpenAICompatLLMProvider,
        "llama_cli": LlamaCliLLMProvider,
    }
    cls = registry.get(provider_id)
    if cls is None:
        raise ValueError(f"Unsupported LLM provider: {provider_id!r}. Available: {sorted(registry)}")
    return cls(project_root, config)
