import logging

logger = logging.getLogger(__name__)


def log_chat_interaction(question: str, answer: str, sources: list[str]) -> dict | None:
    """
    Keep API logging compatible with existing CLI chat log format.
    """
    try:
        from ...full_rag import save_interaction

        return save_interaction(question=question, answer=answer, sources=sources)
    except Exception as exc:  # pragma: no cover - logging failures should not fail requests
        logger.warning("Could not persist chat interaction: %s", exc)
        return None
