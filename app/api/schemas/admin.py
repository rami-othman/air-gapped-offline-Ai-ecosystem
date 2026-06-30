from typing import Any

from pydantic import BaseModel, StrictBool


class ChatHistoryMigrationRequest(BaseModel):
    output_dir: str | None = None
    write_latest: StrictBool = True


class ChatHistoryMigrationResponse(BaseModel):
    status: str
    operation: str
    items_migrated: int
    output_file: str
    latest_file: str | None = None


class ChatHistoryIngestRequest(BaseModel):
    input_file: str | None = None
    dry_run: StrictBool = False


class ChatHistoryIngestResponse(BaseModel):
    status: str
    operation: str
    records_loaded: int
    records_upserted: int
    records_skipped: int
    collection: str


class JobCreateResponse(BaseModel):
    status: str
    job_id: str
    job_status: str
    message: str


class JobSummary(BaseModel):
    job_id: str
    name: str
    job_status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    result: Any | None = None
    error: str | None = None
    progress_message: str | None = None
    metadata: dict[str, Any] | None = None


class JobStatusResponse(BaseModel):
    status: str
    job: JobSummary


class JobListResponse(BaseModel):
    status: str
    count: int
    jobs: list[JobSummary]


class JobStatsResponse(BaseModel):
    status: str
    stats: dict[str, Any]
