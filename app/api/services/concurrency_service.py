"""Process-local concurrency limiter for API RAG queries.

This limiter is intentionally in-memory and process-local. If the API runs
with multiple worker processes, each process has its own independent limiter.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
import time
from collections.abc import Iterator

from ...config import (
    MAX_CONCURRENT_LLM_REQUESTS,
    MAX_QUEUE_WAIT_SECONDS,
    MAX_WAITING_RAG_REQUESTS,
)

logger = logging.getLogger(__name__)


class RagLimiterRejectedError(Exception):
    """Raised when the bounded waiting queue is already full."""

    def __init__(
        self,
        message: str,
        *,
        active_requests: int,
        waiting_requests: int,
        max_active_requests: int,
        max_waiting_requests: int,
        max_queue_wait_seconds: int,
    ) -> None:
        super().__init__(message)
        self.active_requests = active_requests
        self.waiting_requests = waiting_requests
        self.max_active_requests = max_active_requests
        self.max_waiting_requests = max_waiting_requests
        self.max_queue_wait_seconds = max_queue_wait_seconds


class RagLimiterTimeoutError(Exception):
    """Raised when a request waits too long for a RAG execution slot."""

    def __init__(
        self,
        message: str,
        *,
        queue_wait_time_sec: float,
        max_queue_wait_seconds: int,
        active_requests: int,
        waiting_requests: int,
    ) -> None:
        super().__init__(message)
        self.queue_wait_time_sec = queue_wait_time_sec
        self.max_queue_wait_seconds = max_queue_wait_seconds
        self.active_requests = active_requests
        self.waiting_requests = waiting_requests


@dataclass(frozen=True)
class RagLimiterMetrics:
    queue_wait_time_sec: float
    active_llm_requests: int
    waiting_rag_requests: int


class RagRequestLimiter:
    def __init__(
        self,
        *,
        max_active_requests: int,
        max_waiting_requests: int,
        queue_wait_timeout_sec: int,
    ) -> None:
        self.max_active_requests = max(1, max_active_requests)
        self.max_waiting_requests = max(0, max_waiting_requests)
        self.queue_wait_timeout_sec = max(0, queue_wait_timeout_sec)
        self._semaphore = threading.BoundedSemaphore(self.max_active_requests)
        self._lock = threading.Lock()
        self._active_requests = 0
        self._waiting_requests = 0

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active_requests

    @property
    def waiting_requests(self) -> int:
        with self._lock:
            return self._waiting_requests

    @contextmanager
    def acquire(self) -> Iterator[RagLimiterMetrics]:
        wait_started = time.monotonic()
        acquired = self._semaphore.acquire(blocking=False)
        was_waiting = False

        if acquired:
            metrics = self._mark_active(wait_started)
            logger.info(
                "RAG limiter allowed request immediately. active=%d waiting=%d",
                metrics.active_llm_requests,
                metrics.waiting_rag_requests,
            )
        else:
            with self._lock:
                if self._waiting_requests >= self.max_waiting_requests:
                    active_requests = self._active_requests
                    waiting_requests = self._waiting_requests
                    logger.warning(
                        "RAG limiter rejected request; queue full. active=%d waiting=%d max_waiting=%d",
                        active_requests,
                        waiting_requests,
                        self.max_waiting_requests,
                    )
                    raise RagLimiterRejectedError(
                        "RAG request queue is full.",
                        active_requests=active_requests,
                        waiting_requests=waiting_requests,
                        max_active_requests=self.max_active_requests,
                        max_waiting_requests=self.max_waiting_requests,
                        max_queue_wait_seconds=self.queue_wait_timeout_sec,
                    )
                self._waiting_requests += 1
                was_waiting = True
                waiting_count = self._waiting_requests
                active_count = self._active_requests

            logger.info(
                "RAG limiter queued request. active=%d waiting=%d timeout_sec=%d",
                active_count,
                waiting_count,
                self.queue_wait_timeout_sec,
            )

            acquired = self._semaphore.acquire(timeout=self.queue_wait_timeout_sec)
            with self._lock:
                self._waiting_requests -= 1

            if not acquired:
                wait_time = time.monotonic() - wait_started
                active_requests = self.active_requests
                waiting_requests = self.waiting_requests
                logger.warning(
                    "RAG limiter timed out request. wait_time_sec=%.4f timeout_sec=%d active=%d waiting=%d",
                    wait_time,
                    self.queue_wait_timeout_sec,
                    active_requests,
                    waiting_requests,
                )
                raise RagLimiterTimeoutError(
                    "RAG request timed out waiting for an execution slot.",
                    queue_wait_time_sec=wait_time,
                    max_queue_wait_seconds=self.queue_wait_timeout_sec,
                    active_requests=active_requests,
                    waiting_requests=waiting_requests,
                )

            metrics = self._mark_active(wait_started)
            logger.info(
                "RAG limiter allowed queued request. queue_wait_time_sec=%.4f active=%d waiting=%d",
                metrics.queue_wait_time_sec,
                metrics.active_llm_requests,
                metrics.waiting_rag_requests,
            )

        try:
            yield metrics
        finally:
            if acquired:
                self._release()
                logger.info(
                    "RAG limiter released request slot. active=%d waiting=%d was_waiting=%s",
                    self.active_requests,
                    self.waiting_requests,
                    was_waiting,
                )

    def _mark_active(self, wait_started: float) -> RagLimiterMetrics:
        queue_wait_time_sec = time.monotonic() - wait_started
        with self._lock:
            self._active_requests += 1
            active_requests = self._active_requests
            waiting_requests = self._waiting_requests

        return RagLimiterMetrics(
            queue_wait_time_sec=queue_wait_time_sec,
            active_llm_requests=active_requests,
            waiting_rag_requests=waiting_requests,
        )

    def _release(self) -> None:
        with self._lock:
            self._active_requests -= 1
            if self._active_requests < 0:
                self._active_requests = 0
        self._semaphore.release()


rag_request_limiter = RagRequestLimiter(
    max_active_requests=MAX_CONCURRENT_LLM_REQUESTS,
    max_waiting_requests=MAX_WAITING_RAG_REQUESTS,
    queue_wait_timeout_sec=MAX_QUEUE_WAIT_SECONDS,
)
