"""Shared LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClientError(RuntimeError):
    """Raised for backend-specific LLM client failures."""


class BaseLLMClient(ABC):
    backend: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text and return a normalized result dictionary."""
