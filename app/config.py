"""Centralized runtime configuration for the local RAG project."""

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_llm_backend(name: str, default: str) -> str:
    backend = os.getenv(name, default).strip().lower()
    if not backend:
        backend = default
    if backend not in {"ollama", "vllm"}:
        raise ValueError(f"{name} must be either 'ollama' or 'vllm'. Got: {backend!r}")
    return backend


# Service endpoints
# Host mode defaults:
# - OLLAMA_BASE_URL=http://localhost:11434
# - CHROMA_HOST=localhost
# Docker mode:
# - OLLAMA_BASE_URL=http://ollama:11434
# - CHROMA_HOST=chromadb
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = _get_int("CHROMA_PORT", 8000)

# LLM backend selection. Ollama remains the default production backend.
LLM_BACKEND = _get_llm_backend("LLM_BACKEND", "ollama")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "")
VLLM_TIMEOUT_SECONDS = _get_int("VLLM_TIMEOUT_SECONDS", 300)

# Vector database
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# Models
GENERAL_GENERATION_MODEL = os.getenv("GENERAL_GENERATION_MODEL", "gemma3:12b")
TECHNICAL_GENERATION_MODEL = os.getenv("TECHNICAL_GENERATION_MODEL", "qwen2.5-coder:14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Retrieval / ingestion defaults.
# After changing chunk size, overlap, or chunking logic, re-ingest PDFs so
# Chroma contains chunks built with the active configuration.
TOP_K = _get_int("TOP_K", 5)
CHUNK_SIZE = _get_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 180)
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")

# Performance / concurrency foundation. Enforcement is intentionally left to
# later tasks; these values are exposed now so API metadata and future runtime
# controls use one source of truth.
MAX_CONCURRENT_LLM_REQUESTS = _get_int("MAX_CONCURRENT_LLM_REQUESTS", 2)
MAX_WAITING_RAG_REQUESTS = _get_int("MAX_WAITING_RAG_REQUESTS", 20)
MAX_QUEUE_WAIT_SECONDS = _get_int("MAX_QUEUE_WAIT_SECONDS", 45)

# Ollama runtime defaults
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
MODEL_NUM_CTX = _get_int("MODEL_NUM_CTX", 4096)
MODEL_NUM_BATCH = _get_int("MODEL_NUM_BATCH", 64)
MODEL_NUM_THREAD = _get_int("MODEL_NUM_THREAD", 8)
MODEL_TEMPERATURE = _get_float("MODEL_TEMPERATURE", 0.2)
MODEL_NUM_PREDICT = _get_int("MODEL_NUM_PREDICT", 512)

# Ollama warm-up
WARMUP_ON_STARTUP = _get_bool("WARMUP_ON_STARTUP", False)
WARMUP_TIMEOUT_SECONDS = _get_int("WARMUP_TIMEOUT_SECONDS", 60)
WARMUP_GENERATION_MODEL = _get_bool("WARMUP_GENERATION_MODEL", True)
WARMUP_EMBEDDING_MODEL = _get_bool("WARMUP_EMBEDDING_MODEL", True)

# Cache metadata foundation. Cache behavior is not implemented in Task 9.
RAG_RETRIEVAL_CACHE_ENABLED = _get_bool("RAG_RETRIEVAL_CACHE_ENABLED", False)
RAG_RESPONSE_CACHE_ENABLED = _get_bool("RAG_RESPONSE_CACHE_ENABLED", False)
RAG_CACHE_TTL_SECONDS = _get_int("RAG_CACHE_TTL_SECONDS", 600)
RAG_CACHE_MAX_ITEMS = _get_int("RAG_CACHE_MAX_ITEMS", 500)
RAG_INDEX_VERSION = os.getenv("RAG_INDEX_VERSION", "dev")
RAG_PROMPT_VERSION = os.getenv("RAG_PROMPT_VERSION", "v1")

# Background jobs for heavy admin/maintenance operations.
BACKGROUND_JOBS_ENABLED = _get_bool("BACKGROUND_JOBS_ENABLED", True)
BACKGROUND_JOB_WORKERS = _get_int("BACKGROUND_JOB_WORKERS", 1)
BACKGROUND_JOB_MAX_QUEUE = _get_int("BACKGROUND_JOB_MAX_QUEUE", 20)
BACKGROUND_JOB_RETENTION_SECONDS = _get_int("BACKGROUND_JOB_RETENTION_SECONDS", 86400)
