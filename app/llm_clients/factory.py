"""LLM client factory."""

from __future__ import annotations

from .base import BaseLLMClient, LLMClientError
from .ollama_client import OllamaClient
from .vllm_client import VLLMClient
from ..config import LLM_BACKEND


def get_llm_client(
    backend: str | None = None,
    *,
    timeout_seconds: int | None = None,
) -> BaseLLMClient:
    selected_backend = (backend or LLM_BACKEND or "ollama").strip().lower()
    if selected_backend == "ollama":
        return OllamaClient(timeout_seconds=timeout_seconds or 180)
    if selected_backend == "vllm":
        return VLLMClient(timeout_seconds=timeout_seconds)
    raise LLMClientError(
        f"Invalid LLM backend {selected_backend!r}. Expected 'ollama' or 'vllm'."
    )
