from fastapi import APIRouter, Body, Depends, Header, Query

from ..config import ADMIN_API_KEY_HEADER
from ..schemas.admin import (
    ChatHistoryIngestRequest,
    ChatHistoryIngestResponse,
    ChatHistoryMigrationRequest,
    ChatHistoryMigrationResponse,
    JobCreateResponse,
    JobListResponse,
    JobStatsResponse,
    JobStatusResponse,
)
from ..services import admin_service, job_service
from ..services.auth_service import ensure_admin_access
from ..services.rag_service import RAGServiceError

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _require_admin_access(
    api_key: str | None = Header(default=None, alias=ADMIN_API_KEY_HEADER),
) -> None:
    ensure_admin_access(provided_api_key=api_key)


def _job_create_response(job) -> JobCreateResponse:
    return JobCreateResponse(
        status="queued",
        job_id=job.job_id,
        job_status=job.status,
        message="Job queued successfully.",
    )


@router.get("/jobs", response_model=JobListResponse)
def list_background_jobs(
    limit: int = Query(default=50, ge=1, le=500),
    _: None = Depends(_require_admin_access),
) -> JobListResponse:
    jobs = job_service.list_jobs(limit=limit)
    return JobListResponse(status="success", count=len(jobs), jobs=jobs)


@router.get("/jobs-stats", response_model=JobStatsResponse)
def get_background_job_stats(
    _: None = Depends(_require_admin_access),
) -> JobStatsResponse:
    return JobStatsResponse(status="success", stats=job_service.get_queue_stats())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_background_job(
    job_id: str,
    _: None = Depends(_require_admin_access),
) -> JobStatusResponse:
    job = job_service.get_job(job_id)
    if job is None:
        raise RAGServiceError(
            "Background job not found.",
            error_code="job_not_found",
            status_code=404,
        )
    return JobStatusResponse(status="success", job=job)


@router.post("/migrate-chat-history", response_model=ChatHistoryMigrationResponse)
def migrate_chat_history(
    payload: ChatHistoryMigrationRequest = Body(default_factory=ChatHistoryMigrationRequest),
    _: None = Depends(_require_admin_access),
) -> ChatHistoryMigrationResponse:
    result = admin_service.run_chat_history_migration(
        output_dir=payload.output_dir,
        write_latest=payload.write_latest,
    )
    return ChatHistoryMigrationResponse(**result)


@router.post("/migrate-chat-history/async", response_model=JobCreateResponse)
def migrate_chat_history_async(
    payload: ChatHistoryMigrationRequest = Body(default_factory=ChatHistoryMigrationRequest),
    _: None = Depends(_require_admin_access),
) -> JobCreateResponse:
    job = job_service.create_job(
        "migrate_chat_history",
        admin_service.run_chat_history_migration,
        metadata={
            "output_dir": payload.output_dir,
            "write_latest": payload.write_latest,
        },
        output_dir=payload.output_dir,
        write_latest=payload.write_latest,
    )
    return _job_create_response(job)


@router.post("/ingest-chat-history", response_model=ChatHistoryIngestResponse)
def ingest_chat_history(
    payload: ChatHistoryIngestRequest = Body(default_factory=ChatHistoryIngestRequest),
    _: None = Depends(_require_admin_access),
) -> ChatHistoryIngestResponse:
    result = admin_service.run_chat_history_ingestion(
        input_file=payload.input_file,
        dry_run=payload.dry_run,
    )
    return ChatHistoryIngestResponse(**result)


@router.post("/ingest-chat-history/async", response_model=JobCreateResponse)
def ingest_chat_history_async(
    payload: ChatHistoryIngestRequest = Body(default_factory=ChatHistoryIngestRequest),
    _: None = Depends(_require_admin_access),
) -> JobCreateResponse:
    job = job_service.create_job(
        "ingest_chat_history",
        admin_service.run_chat_history_ingestion,
        metadata={
            "input_file": payload.input_file,
            "dry_run": payload.dry_run,
        },
        input_file=payload.input_file,
        dry_run=payload.dry_run,
    )
    return _job_create_response(job)
