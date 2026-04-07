import logging

from fastapi import HTTPException

from ..config import INGEST_API_KEY, INGEST_PROTECTION_ENABLED

logger = logging.getLogger(__name__)


def ensure_ingest_access(provided_api_key: str | None) -> None:
    if not INGEST_PROTECTION_ENABLED:
        return

    if not INGEST_API_KEY:
        logger.error("Ingest protection is enabled but INGEST_API_KEY is empty.")
        raise HTTPException(
            status_code=503,
            detail="Ingest endpoint is not available due to server configuration.",
        )

    if provided_api_key != INGEST_API_KEY:
        logger.warning("Unauthorized ingest request rejected.")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized for ingest endpoint.",
        )
