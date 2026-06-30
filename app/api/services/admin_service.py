import logging

import requests

from .rag_service import RAGServiceError

logger = logging.getLogger(__name__)


def run_chat_history_migration(output_dir: str | None = None, write_latest: bool = True) -> dict:
    try:
        from scripts.migrate_chat_history import run_migration

        return run_migration(output_dir=output_dir, write_latest=write_latest)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat history migration failed. output_dir=%s", output_dir)
        raise RAGServiceError(
            "Could not migrate chat history.",
            error_code="chat_history_migration_error",
            status_code=500,
        ) from exc


def run_chat_history_ingestion(input_file: str | None = None, dry_run: bool = False) -> dict:
    try:
        from scripts.ingest_chat_history import run_ingestion

        return run_ingestion(input_file=input_file, dry_run=dry_run)
    except FileNotFoundError as exc:
        logger.warning("Migrated chat history file not found. input_file=%s", input_file)
        raise RAGServiceError(
            str(exc),
            error_code="migration_file_not_found",
            status_code=404,
        ) from exc
    except requests.exceptions.RequestException as exc:
        logger.exception("Chat history ingestion failed during Ollama request. input_file=%s", input_file)
        raise RAGServiceError(
            "Ollama request failed during chat history ingestion.",
            error_code="ollama_error",
            status_code=503,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat history ingestion failed. input_file=%s", input_file)
        message = str(exc).lower()
        if "chroma" in message:
            raise RAGServiceError(
                "ChromaDB operation failed during chat history ingestion.",
                error_code="chroma_error",
                status_code=503,
            ) from exc
        raise RAGServiceError(
            "Could not ingest chat history.",
            error_code="chat_history_ingestion_error",
            status_code=500,
        ) from exc
