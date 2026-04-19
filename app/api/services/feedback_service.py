import logging

from ...chat_log_store import update_interaction_helpful
from .rag_service import RAGServiceError

logger = logging.getLogger(__name__)


def _normalize_interaction_id(interaction_id: str) -> str:
    normalized = (interaction_id or "").strip()
    if not normalized:
        raise RAGServiceError(
            "interaction_id must not be empty.",
            error_code="validation_error",
            status_code=422,
        )
    return normalized


def update_feedback(interaction_id: str, helpful: bool) -> dict:
    normalized_interaction_id = _normalize_interaction_id(interaction_id)

    try:
        updated = update_interaction_helpful(
            interaction_id=normalized_interaction_id,
            helpful=helpful,
        )
    except ValueError as exc:
        raise RAGServiceError(
            "helpful must be true or false.",
            error_code="validation_error",
            status_code=422,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Feedback update failed. interaction_id=%s",
            normalized_interaction_id,
        )
        raise RAGServiceError(
            "Could not update chat feedback.",
            error_code="feedback_update_error",
            status_code=500,
        ) from exc

    if not updated:
        raise RAGServiceError(
            "Interaction not found.",
            error_code="not_found",
            status_code=404,
        )

    return {
        "status": "success",
        "interaction_id": normalized_interaction_id,
        "helpful": helpful,
    }
