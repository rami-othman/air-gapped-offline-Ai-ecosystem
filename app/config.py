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

# Vector database
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# Models
GENERAL_GENERATION_MODEL = os.getenv("GENERAL_GENERATION_MODEL", "gemma3:12b")
TECHNICAL_GENERATION_MODEL = os.getenv("TECHNICAL_GENERATION_MODEL", "qwen2.5-coder:14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Retrieval / ingestion defaults
TOP_K = _get_int("TOP_K", 5)
CHUNK_SIZE = _get_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
