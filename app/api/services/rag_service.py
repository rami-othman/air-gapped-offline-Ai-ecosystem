import time
import uuid
import logging

import requests

from ...config import DOCS_DIR
from .logging_service import log_chat_interaction

logger = logging.getLogger(__name__)


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
    endpoint = "/api/v1/rag/query"
    start = time.perf_counter()

    try:
        from ...full_rag import run_rag_query

        result = run_rag_query(normalized_question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed. endpoint=%s session_id=%s", endpoint, active_session_id)
        _raise_backend_error(exc, action="query")

    sources = result.get("retrieved_sources", [])
    answer = result.get("answer", "")
    retrieval_time_sec = float(result.get("retrieval_time_sec", 0.0))
    generation_time_sec = float(result.get("generation_time_sec", 0.0))
    total_time_sec = float(result.get("total_time_sec", retrieval_time_sec + generation_time_sec))
    status = "success"
    interaction_id = None

    if result.get("status") == "success":
        interaction = log_chat_interaction(
            question=normalized_question,
            answer=answer,
            sources=sources,
        )
        if interaction:
            interaction_id = interaction.get("id")

    rag_status = result.get("status", "unknown")
    logger.info(
        "RAG query completed. endpoint=%s session_id=%s rag_status=%s retrieval_time_sec=%.4f generation_time_sec=%.4f total_time_sec=%.4f",
        endpoint,
        active_session_id,
        rag_status,
        retrieval_time_sec,
        generation_time_sec,
        time.perf_counter() - start,
    )

    return {
        "status": status,
        "answer": answer,
        "sources": sources,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "total_time_sec": total_time_sec,
        "session_id": active_session_id,
        "interaction_id": interaction_id,
    }


def run_search(query: str, top_k: int | None = None) -> dict:
    normalized_query = _normalize_question(query)
    endpoint = "/api/v1/rag/search"
    start = time.perf_counter()

    try:
        from ...full_rag import retrieve

        docs, metadatas = retrieve(normalized_query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG search failed. endpoint=%s", endpoint)
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

    logger.info(
        "RAG search completed. endpoint=%s chunks=%d retrieval_time_sec=%.4f",
        endpoint,
        len(chunks),
        retrieval_time_sec,
    )

    return {
        "status": "success",
        "chunks": chunks,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": 0.0,
        "total_time_sec": retrieval_time_sec,
    }


def run_ingestion(docs_dir: str | None = None) -> dict:
    active_docs_dir = docs_dir or DOCS_DIR
    endpoint = "/api/v1/rag/ingest"
    start = time.perf_counter()

    try:
        from ...ingest_documents import ingest_directory

        documents_ingested, chunks_ingested = ingest_directory(active_docs_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG ingestion failed. endpoint=%s docs_dir=%s", endpoint, active_docs_dir)
        _raise_backend_error(exc, action="ingestion")

    elapsed = time.perf_counter() - start
    logger.info(
        "RAG ingestion completed. endpoint=%s docs_dir=%s documents=%d chunks=%d total_time_sec=%.4f",
        endpoint,
        active_docs_dir,
        documents_ingested,
        chunks_ingested,
        elapsed,
    )

    return {
        "status": "success",
        "docs_dir": active_docs_dir,
        "documents_ingested": documents_ingested,
        "chunks_ingested": chunks_ingested,
        "total_time_sec": elapsed,
    }
