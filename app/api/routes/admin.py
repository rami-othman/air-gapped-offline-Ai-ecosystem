from fastapi import APIRouter, Body, Depends, Header

from ..config import ADMIN_API_KEY_HEADER
from ..schemas.admin import (
    ChatHistoryIngestRequest,
    ChatHistoryIngestResponse,
    ChatHistoryMigrationRequest,
    ChatHistoryMigrationResponse,
)
from ..services import admin_service
from ..services.auth_service import ensure_admin_access

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _require_admin_access(
    api_key: str | None = Header(default=None, alias=ADMIN_API_KEY_HEADER),
) -> None:
    ensure_admin_access(provided_api_key=api_key)


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
