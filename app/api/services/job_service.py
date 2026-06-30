"""In-process background job manager for heavy admin tasks.

This is intentionally process-local. If the API runs with multiple worker
processes, each process has its own job queue and job history.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from ...config import (
    BACKGROUND_JOB_MAX_QUEUE,
    BACKGROUND_JOB_RETENTION_SECONDS,
    BACKGROUND_JOB_WORKERS,
    BACKGROUND_JOBS_ENABLED,
)
from .rag_service import RAGServiceError

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"success", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    name: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    result: Any | None = None
    error: str | None = None
    progress_message: str | None = None
    metadata: dict[str, Any] | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    finished_monotonic: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "job_status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "result": self.result,
            "error": self.error,
            "progress_message": self.progress_message,
            "metadata": self.metadata,
        }


class BackgroundJobManager:
    def __init__(
        self,
        *,
        enabled: bool,
        workers: int,
        max_queue: int,
        retention_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.workers = max(1, workers)
        self.max_queue = max(1, max_queue)
        self.retention_seconds = max(0, retention_seconds)
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="admin-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create_job(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> JobRecord:
        self.cleanup_old_jobs()
        if not self.enabled:
            raise RAGServiceError(
                "Background jobs are disabled.",
                error_code="background_jobs_disabled",
                status_code=503,
            )

        with self._lock:
            active_count = self._active_job_count_locked()
            if active_count >= self.max_queue:
                logger.warning(
                    "Background job queue full. active_jobs=%d max_queue=%d name=%s",
                    active_count,
                    self.max_queue,
                    name,
                )
                raise RAGServiceError(
                    "Background job queue is full. Please try again shortly.",
                    error_code="background_queue_full",
                    status_code=503,
                    details={
                        "active_jobs": active_count,
                        "max_queue": self.max_queue,
                    },
                )

            job = JobRecord(
                job_id=str(uuid.uuid4()),
                name=name,
                status="queued",
                created_at=_utc_now(),
                metadata=metadata or {},
            )
            self._jobs[job.job_id] = job

        try:
            self._executor.submit(self._run_job, job.job_id, func, args, kwargs)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._jobs.pop(job.job_id, None)
            logger.exception("Failed to submit background job. name=%s", name)
            raise RAGServiceError(
                "Could not queue background job.",
                error_code="background_job_submit_error",
                status_code=500,
            ) from exc

        logger.info("Background job queued. job_id=%s name=%s", job.job_id, name)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.cleanup_old_jobs()
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        self.cleanup_old_jobs()
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_monotonic,
                reverse=True,
            )
            return [job.to_dict() for job in jobs[:safe_limit]]

    def cleanup_old_jobs(self) -> None:
        if self.retention_seconds <= 0:
            return

        now = time.monotonic()
        with self._lock:
            expired_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in TERMINAL_STATUSES
                and job.finished_monotonic is not None
                and now - job.finished_monotonic > self.retention_seconds
            ]
            for job_id in expired_ids:
                self._jobs.pop(job_id, None)

    def get_queue_stats(self) -> dict[str, Any]:
        self.cleanup_old_jobs()
        with self._lock:
            queued_count = sum(1 for job in self._jobs.values() if job.status == "queued")
            running_count = sum(1 for job in self._jobs.values() if job.status == "running")
            success_count = sum(1 for job in self._jobs.values() if job.status == "success")
            failed_count = sum(1 for job in self._jobs.values() if job.status == "failed")
            return {
                "enabled": self.enabled,
                "workers": self.workers,
                "max_queue": self.max_queue,
                "retention_seconds": self.retention_seconds,
                "queued_count": queued_count,
                "running_count": running_count,
                "active_count": queued_count + running_count,
                "success_count": success_count,
                "failed_count": failed_count,
                "total_tracked": len(self._jobs),
            }

    def _run_job(
        self,
        job_id: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        started_monotonic = time.monotonic()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = _utc_now()
            job.progress_message = "Job started."

        logger.info("Background job started. job_id=%s name=%s", job.job_id, job.name)
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            duration_sec = time.monotonic() - started_monotonic
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "failed"
                    job.finished_at = _utc_now()
                    job.finished_monotonic = time.monotonic()
                    job.duration_sec = round(duration_sec, 4)
                    job.error = str(exc) or exc.__class__.__name__
                    job.progress_message = "Job failed."
            logger.exception("Background job failed. job_id=%s", job_id)
            return

        duration_sec = time.monotonic() - started_monotonic
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "success"
                job.finished_at = _utc_now()
                job.finished_monotonic = time.monotonic()
                job.duration_sec = round(duration_sec, 4)
                job.result = result
                job.progress_message = "Job completed."
        logger.info("Background job succeeded. job_id=%s duration_sec=%.4f", job_id, duration_sec)

    def _active_job_count_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status in {"queued", "running"})


job_manager = BackgroundJobManager(
    enabled=BACKGROUND_JOBS_ENABLED,
    workers=BACKGROUND_JOB_WORKERS,
    max_queue=BACKGROUND_JOB_MAX_QUEUE,
    retention_seconds=BACKGROUND_JOB_RETENTION_SECONDS,
)


def create_job(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> JobRecord:
    return job_manager.create_job(name, func, *args, metadata=metadata, **kwargs)


def get_job(job_id: str) -> dict[str, Any] | None:
    return job_manager.get_job(job_id)


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    return job_manager.list_jobs(limit=limit)


def cleanup_old_jobs() -> None:
    job_manager.cleanup_old_jobs()


def get_queue_stats() -> dict[str, Any]:
    return job_manager.get_queue_stats()
