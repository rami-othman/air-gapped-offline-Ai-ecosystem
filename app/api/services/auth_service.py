import logging

from fastapi import HTTPException

from ..config import (
    ADMIN_API_KEY,
    ADMIN_PROTECTION_ENABLED,
    INGEST_API_KEY,
    INGEST_PROTECTION_ENABLED,
)

logger = logging.getLogger(__name__)


def _ensure_api_key_access(
    *,
    provided_api_key: str | None,
    expected_api_key: str | None,
    protection_enabled: bool,
    endpoint_name: str,
) -> None:
    if not protection_enabled:
        return

    if not expected_api_key:
        logger.error("%s protection is enabled but the API key is empty.", endpoint_name)
        raise HTTPException(
            status_code=503,
            detail=f"{endpoint_name.title()} endpoint is not available due to server configuration.",
        )

    if provided_api_key != expected_api_key:
        logger.warning("Unauthorized %s request rejected.", endpoint_name)
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized for {endpoint_name} endpoint.",
        )


def ensure_ingest_access(provided_api_key: str | None) -> None:
    _ensure_api_key_access(
        provided_api_key=provided_api_key,
        expected_api_key=INGEST_API_KEY,
        protection_enabled=INGEST_PROTECTION_ENABLED,
        endpoint_name="ingest",
    )


def ensure_admin_access(provided_api_key: str | None) -> None:
    _ensure_api_key_access(
        provided_api_key=provided_api_key,
        expected_api_key=ADMIN_API_KEY,
        protection_enabled=ADMIN_PROTECTION_ENABLED,
        endpoint_name="admin",
    )
