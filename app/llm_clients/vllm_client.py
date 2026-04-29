"""vLLM OpenAI-compatible benchmark client."""

from __future__ import annotations

import time
from typing import Any

import requests

from ..config import (
    MODEL_NUM_PREDICT,
    MODEL_TEMPERATURE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TIMEOUT_SECONDS,
)
from .base import BaseLLMClient, LLMClientError


class VLLMClient(BaseLLMClient):
    backend = "vllm"

    def __init__(self, timeout_seconds: int | None = None) -> None:
        self.timeout_seconds = timeout_seconds or VLLM_TIMEOUT_SECONDS

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_model = model or VLLM_MODEL
        if not active_model:
            raise LLMClientError("VLLM_MODEL is required when using the vLLM backend.")

        active_options = options or {}
        payload = {
            "model": active_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": active_options.get("temperature", MODEL_TEMPERATURE),
            "max_tokens": active_options.get("max_tokens", MODEL_NUM_PREDICT),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {VLLM_API_KEY}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise LLMClientError(f"vLLM generation failed: {exc}") from exc

        generation_time_sec = time.perf_counter() - started
        choices = data.get("choices", [])
        text = ""
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "") or ""

        return {
            "text": text,
            "model": active_model,
            "backend": self.backend,
            "raw": data,
            "generation_time_sec": generation_time_sec,
        }
