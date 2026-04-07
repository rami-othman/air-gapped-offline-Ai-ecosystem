import logging

import chromadb
import requests

from ..config import (
    API_CHROMA_HOST,
    API_CHROMA_PORT,
    API_OLLAMA_BASE_URL,
    HEALTH_INCLUDE_ERROR_DETAILS,
)

logger = logging.getLogger(__name__)


def _service_status(is_ok: bool, error: str | None = None) -> dict:
    payload = {"status": "up" if is_ok else "down"}
    if HEALTH_INCLUDE_ERROR_DETAILS and error:
        payload["error"] = error
    return payload


def check_ollama() -> dict:
    try:
        response = requests.get(f"{API_OLLAMA_BASE_URL}/api/tags", timeout=4)
        response.raise_for_status()
        return _service_status(True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ollama health check failed.")
        return _service_status(False, error=str(exc))


def check_chroma() -> dict:
    try:
        client = chromadb.HttpClient(host=API_CHROMA_HOST, port=API_CHROMA_PORT)
        client.list_collections()
        return _service_status(True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ChromaDB health check failed.")
        return _service_status(False, error=str(exc))


def get_health_payload() -> dict:
    ollama = check_ollama()
    chroma = check_chroma()
    overall_status = "ok" if ollama["status"] == "up" and chroma["status"] == "up" else "degraded"

    return {
        "status": overall_status,
        "ollama": ollama,
        "chroma": chroma,
    }
