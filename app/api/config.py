import os

from ..config import CHROMA_HOST, CHROMA_PORT, OLLAMA_BASE_URL


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _get_int("API_PORT", 8001)
API_DEV_MODE = _get_bool("API_DEV_MODE", True)
API_RELOAD = _get_bool("API_RELOAD", API_DEV_MODE)
HEALTH_INCLUDE_ERROR_DETAILS = _get_bool("HEALTH_INCLUDE_ERROR_DETAILS", API_DEV_MODE)

# /api/v1/rag/ingest protection settings
INGEST_PROTECTION_ENABLED = _get_bool("INGEST_PROTECTION_ENABLED", True)
INGEST_API_KEY_HEADER = os.getenv("INGEST_API_KEY_HEADER", "X-API-Key")
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "dev-ingest-key")

# /api/v1/admin/* protection settings
ADMIN_PROTECTION_ENABLED = _get_bool("ADMIN_PROTECTION_ENABLED", INGEST_PROTECTION_ENABLED)
ADMIN_API_KEY_HEADER = os.getenv("ADMIN_API_KEY_HEADER", INGEST_API_KEY_HEADER)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", INGEST_API_KEY)

# Re-export dependency endpoints for API modules.
API_OLLAMA_BASE_URL = OLLAMA_BASE_URL
API_CHROMA_HOST = CHROMA_HOST
API_CHROMA_PORT = CHROMA_PORT
