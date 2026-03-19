"""Centralized configuration for the Week 2 RAG baseline."""

# Service endpoints
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Vector database
CHROMA_COLLECTION = "documents"

# Models
GENERAL_GENERATION_MODEL = "gemma3:12b"
TECHNICAL_GENERATION_MODEL = "qwen2.5-coder:14b"

# Temporary default until embedding evaluation is completed in later Week 2 tasks
#! EMBEDDING_MODEL = "bge-m3"
EMBEDDING_MODEL = "nomic-embed-text" 

# Retrieval / ingestion defaults
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DOCS_DIR = "data/docs"
