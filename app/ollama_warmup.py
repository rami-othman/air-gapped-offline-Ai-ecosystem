"""Ollama model warm-up helpers.

Warm-up only talks to Ollama. It does not call ChromaDB, run retrieval, or
write chat logs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import requests

from .config import (
    EMBEDDING_MODEL,
    GENERAL_GENERATION_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    WARMUP_EMBEDDING_MODEL,
    WARMUP_GENERATION_MODEL,
    WARMUP_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class OllamaWarmupError(RuntimeError):
    """Raised when one or more Ollama warm-up calls fail."""


@dataclass(frozen=True)
class WarmupResult:
    model_name: str
    model_type: str
    elapsed_sec: float
    status: str = "success"


def _post_ollama_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}{path}",
        json=payload,
        timeout=WARMUP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def warmup_generation_model(model_name: str = GENERAL_GENERATION_MODEL) -> WarmupResult:
    logger.info("Starting Ollama generation model warm-up. model=%s", model_name)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": "warmup",
        "stream": False,
        "options": {
            "num_predict": 1,
        },
    }
    if OLLAMA_KEEP_ALIVE:
        payload["keep_alive"] = OLLAMA_KEEP_ALIVE

    try:
        _post_ollama_json("/api/generate", payload)
    except requests.RequestException as exc:
        logger.warning("Ollama generation model warm-up failed. model=%s error=%s", model_name, exc)
        raise OllamaWarmupError(f"Generation model warm-up failed for {model_name}.") from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "Ollama generation model warm-up succeeded. model=%s elapsed_sec=%.4f",
        model_name,
        elapsed,
    )
    return WarmupResult(model_name=model_name, model_type="generation", elapsed_sec=elapsed)


def warmup_embedding_model(model_name: str = EMBEDDING_MODEL) -> WarmupResult:
    logger.info("Starting Ollama embedding model warm-up. model=%s", model_name)
    started = time.perf_counter()
    payload = {
        "model": model_name,
        "prompt": "warmup",
    }

    try:
        data = _post_ollama_json("/api/embeddings", payload)
    except requests.RequestException as exc:
        logger.warning("Ollama embedding model warm-up failed. model=%s error=%s", model_name, exc)
        raise OllamaWarmupError(f"Embedding model warm-up failed for {model_name}.") from exc

    if not data.get("embedding"):
        logger.warning("Ollama embedding model warm-up returned no embedding. model=%s", model_name)
        raise OllamaWarmupError(f"Embedding model warm-up returned no embedding for {model_name}.")

    elapsed = time.perf_counter() - started
    logger.info(
        "Ollama embedding model warm-up succeeded. model=%s elapsed_sec=%.4f",
        model_name,
        elapsed,
    )
    return WarmupResult(model_name=model_name, model_type="embedding", elapsed_sec=elapsed)


def warmup_all_models() -> list[WarmupResult]:
    results: list[WarmupResult] = []

    if WARMUP_GENERATION_MODEL:
        results.append(warmup_generation_model())
    else:
        logger.info("Ollama generation model warm-up skipped by config.")

    if WARMUP_EMBEDDING_MODEL:
        results.append(warmup_embedding_model())
    else:
        logger.info("Ollama embedding model warm-up skipped by config.")

    return results
