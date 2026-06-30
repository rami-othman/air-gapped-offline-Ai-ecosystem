"""Small process-local TTL cache used by the RAG pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

try:
    from .config import RAG_CACHE_MAX_ITEMS, RAG_CACHE_TTL_SECONDS
except ImportError:  # pragma: no cover - script execution fallback
    from config import RAG_CACHE_MAX_ITEMS, RAG_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float
    created_at: float


class TTLCache:
    def __init__(self, *, ttl_seconds: int, max_items: int, name: str = "cache") -> None:
        self.name = name
        self.ttl_seconds = max(0, ttl_seconds)
        self.max_items = max(0, max_items)
        self._items: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.expires_at <= now:
                self._items.pop(key, None)
                self._misses += 1
                return None

            self._hits += 1
            return deepcopy(entry.value)

    def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0 or self.max_items <= 0:
            return

        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            while len(self._items) >= self.max_items:
                oldest_key = min(self._items, key=lambda item_key: self._items[item_key].created_at)
                self._items.pop(oldest_key, None)

            self._items[key] = _CacheEntry(
                value=deepcopy(value),
                expires_at=now + self.ttl_seconds,
                created_at=now,
            )

    def clear(self) -> None:
        with self._lock:
            item_count = len(self._items)
            self._items.clear()
        logger.info("%s cleared. items_removed=%d", self.name, item_count)

    def stats(self) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            return {
                "items": len(self._items),
                "hits": self._hits,
                "misses": self._misses,
                "ttl_seconds": self.ttl_seconds,
                "max_items": self.max_items,
            }

    def _prune_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, entry in self._items.items()
            if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._items.pop(key, None)


retrieval_cache = TTLCache(
    ttl_seconds=RAG_CACHE_TTL_SECONDS,
    max_items=RAG_CACHE_MAX_ITEMS,
    name="Retrieval cache",
)

response_cache = TTLCache(
    ttl_seconds=RAG_CACHE_TTL_SECONDS,
    max_items=RAG_CACHE_MAX_ITEMS,
    name="Response cache",
)
