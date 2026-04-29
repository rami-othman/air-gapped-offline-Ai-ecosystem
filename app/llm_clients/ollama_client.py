"""Ollama generation client."""

from __future__ import annotations

import time
from typing import Any

import requests

from ..config import (
    GENERAL_GENERATION_MODEL,
    MODEL_NUM_BATCH,
    MODEL_NUM_CTX,
    MODEL_NUM_PREDICT,
    MODEL_NUM_THREAD,
    MODEL_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
)
from .base import BaseLLMClient, LLMClientError


DEFAULT_OLLAMA_OPTIONS = {
    "num_ctx": MODEL_NUM_CTX,
    "num_batch": MODEL_NUM_BATCH,
    "num_thread": MODEL_NUM_THREAD,
    "temperature": MODEL_TEMPERATURE,
    "num_predict": MODEL_NUM_PREDICT,
}


class OllamaClient(BaseLLMClient):
    backend = "ollama"

    def __init__(self, timeout_seconds: int = 180) -> None:
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_model = model or GENERAL_GENERATION_MODEL
        generation_options = {**DEFAULT_OLLAMA_OPTIONS, **(options or {})}
        payload: dict[str, Any] = {
            "model": active_model,
            "prompt": prompt,
            "stream": False,
            "options": generation_options,
        }
        if OLLAMA_KEEP_ALIVE:
            payload["keep_alive"] = OLLAMA_KEEP_ALIVE

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise LLMClientError(f"Ollama generation failed: {exc}") from exc

        generation_time_sec = time.perf_counter() - started
        return {
            "text": data.get("response", ""),
            "model": active_model,
            "backend": self.backend,
            "raw": data,
            "generation_time_sec": generation_time_sec,
        }
