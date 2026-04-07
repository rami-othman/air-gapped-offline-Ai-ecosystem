import time
import uuid

import requests

from ...config import DOCS_DIR
from .logging_service import log_chat_interaction


class RAGServiceError(Exception):
    def __init__(self, message: str, *, error_code: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


def _normalize_question(question: str) -> str:
    normalized = (question or "").strip()
    if not normalized:
        raise RAGServiceError(
            "Question must not be empty.",
            error_code="validation_error",
            status_code=422,
        )
    return normalized


def _to_session_id(session_id: str | None) -> str:
    candidate = (session_id or "").strip()
    return candidate or str(uuid.uuid4())


def _raise_backend_error(exc: Exception, action: str) -> None:
    message = str(exc).lower()

    if isinstance(exc, requests.exceptions.RequestException):
        raise RAGServiceError(
            f"Ollama request failed during {action}.",
            error_code="ollama_error",
            status_code=503,
        ) from exc

    if "chroma" in message:
        raise RAGServiceError(
            f"ChromaDB operation failed during {action}.",
            error_code="chroma_error",
            status_code=503,
        ) from exc

    raise RAGServiceError(
        f"RAG operation failed during {action}.",
        error_code="rag_error",
        status_code=500,
    ) from exc


def run_query(question: str, top_k: int | None = None, session_id: str | None = None) -> dict:
    normalized_question = _normalize_question(question)
    active_session_id = _to_session_id(session_id)

    try:
        from ...full_rag import run_rag_query

        result = run_rag_query(normalized_question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        _raise_backend_error(exc, action="query")

    sources = result.get("retrieved_sources", [])
    answer = result.get("answer", "")
    retrieval_time_sec = float(result.get("retrieval_time_sec", 0.0))
    generation_time_sec = float(result.get("generation_time_sec", 0.0))
    total_time_sec = retrieval_time_sec + generation_time_sec

    if result.get("status") == "success":
        log_chat_interaction(
            question=normalized_question,
            answer=answer,
            sources=sources,
        )

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "total_time_sec": total_time_sec,
        "session_id": active_session_id,
    }


def run_search(question: str, top_k: int | None = None) -> dict:
    normalized_question = _normalize_question(question)
    start = time.perf_counter()

    try:
        from ...full_rag import retrieve

        docs, metadatas = retrieve(normalized_question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        _raise_backend_error(exc, action="search")

    retrieval_time_sec = time.perf_counter() - start
    chunks = []
    for index, doc in enumerate(docs):
        metadata = metadatas[index] if index < len(metadatas) else {}
        chunks.append(
            {
                "content": doc,
                "metadata": metadata or {},
            }
        )

    return {
        "chunks": chunks,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": 0.0,
        "total_time_sec": retrieval_time_sec,
    }


def run_ingestion(docs_dir: str | None = None) -> dict:
    active_docs_dir = docs_dir or DOCS_DIR
    start = time.perf_counter()

    try:
        from ...ingest_documents import ingest_directory

        documents_ingested, chunks_ingested = ingest_directory(active_docs_dir)
    except Exception as exc:  # noqa: BLE001
        _raise_backend_error(exc, action="ingestion")

    return {
        "docs_dir": active_docs_dir,
        "documents_ingested": documents_ingested,
        "chunks_ingested": chunks_ingested,
        "total_time_sec": time.perf_counter() - start,
    }
