import hashlib
import re
import time
import uuid
import logging
from typing import Any

import requests

from ...config import (
    CHROMA_COLLECTION,
    DOCS_DIR,
    EMBEDDING_MODEL,
    GENERAL_GENERATION_MODEL,
    MODEL_NUM_CTX,
    MODEL_NUM_PREDICT,
    RAG_INDEX_VERSION,
    RAG_PROMPT_VERSION,
    RAG_RESPONSE_CACHE_ENABLED,
    TOP_K,
)
from ...cache_store import response_cache
from .concurrency_service import (
    RagLimiterRejectedError,
    RagLimiterTimeoutError,
    rag_request_limiter,
)
from .logging_service import log_chat_interaction

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details


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


def _normalize_response_cache_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _build_response_cache_key(question: str, top_k: int) -> str:
    key_parts = [
        _normalize_response_cache_question(question),
        str(top_k),
        GENERAL_GENERATION_MODEL,
        RAG_PROMPT_VERSION,
        RAG_INDEX_VERSION,
        str(MODEL_NUM_CTX),
        str(MODEL_NUM_PREDICT),
        CHROMA_COLLECTION,
        EMBEDDING_MODEL,
    ]
    raw_key = "\n".join(key_parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _is_cacheable_rag_result(result: dict) -> bool:
    answer = str(result.get("answer", "")).strip()
    return (
        result.get("status") == "success"
        and bool(answer)
        and answer != "No relevant context found in the vector database."
    )


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
        "RAG query failed while processing the request.",
        error_code="rag_execution_error",
        status_code=500,
        details={"action": action},
    ) from exc


def run_query(question: str, top_k: int | None = None, session_id: str | None = None) -> dict:
    normalized_question = _normalize_question(question)
    active_session_id = _to_session_id(session_id)
    actual_top_k = top_k if top_k is not None else TOP_K
    endpoint = "/api/v1/rag/query"
    start = time.perf_counter()
    response_cache_key = _build_response_cache_key(normalized_question, actual_top_k)

    try:
        with rag_request_limiter.acquire() as limiter_metrics:
            result = None
            if RAG_RESPONSE_CACHE_ENABLED:
                cached_response = response_cache.get(response_cache_key)
                if cached_response is not None:
                    logger.info(
                        "Response cache hit. endpoint=%s session_id=%s top_k=%d",
                        endpoint,
                        active_session_id,
                        actual_top_k,
                    )
                    result = {
                        "status": "success",
                        "answer": cached_response.get("answer", ""),
                        "retrieved_sources": cached_response.get("retrieved_sources", []),
                        "retrieval_time_sec": 0.0,
                        "generation_time_sec": 0.0,
                        "total_time_sec": time.perf_counter() - start,
                        "cache_hit": True,
                        "cache_type": "response",
                    }
                else:
                    logger.info(
                        "Response cache miss. endpoint=%s session_id=%s top_k=%d",
                        endpoint,
                        active_session_id,
                        actual_top_k,
                    )
            else:
                logger.debug("Response cache disabled.")

            if result is None:
                from ...full_rag import run_rag_query

                result = run_rag_query(normalized_question, top_k=actual_top_k)
                if RAG_RESPONSE_CACHE_ENABLED and _is_cacheable_rag_result(result):
                    response_cache.set(
                        response_cache_key,
                        {
                            "answer": result.get("answer", ""),
                            "retrieved_sources": result.get("retrieved_sources", []),
                            "retrieval_time_sec": float(result.get("retrieval_time_sec", 0.0)),
                        },
                    )
    except RagLimiterRejectedError as exc:
        logger.warning(
            "RAG query rejected by limiter. endpoint=%s session_id=%s reason=queue_full active=%d waiting=%d max_active=%d max_waiting=%d",
            endpoint,
            active_session_id,
            exc.active_requests,
            exc.waiting_requests,
            exc.max_active_requests,
            exc.max_waiting_requests,
        )
        raise RAGServiceError(
            "The AI server is currently busy. Please try again shortly.",
            error_code="server_busy",
            status_code=503,
            details={
                "max_concurrent_requests": exc.max_active_requests,
                "max_waiting_requests": exc.max_waiting_requests,
                "max_queue_wait_seconds": exc.max_queue_wait_seconds,
            },
        ) from exc
    except RagLimiterTimeoutError as exc:
        logger.warning(
            "RAG query rejected by limiter. endpoint=%s session_id=%s reason=queue_timeout wait_time_sec=%.4f timeout_sec=%d active=%d waiting=%d",
            endpoint,
            active_session_id,
            exc.queue_wait_time_sec,
            exc.max_queue_wait_seconds,
            exc.active_requests,
            exc.waiting_requests,
        )
        raise RAGServiceError(
            "Your request waited too long because the AI server is busy. Please try again shortly.",
            error_code="queue_timeout",
            status_code=503,
            details={
                "queue_wait_time_sec": round(exc.queue_wait_time_sec, 4),
                "max_queue_wait_seconds": exc.max_queue_wait_seconds,
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed. endpoint=%s session_id=%s", endpoint, active_session_id)
        _raise_backend_error(exc, action="query")

    sources = result.get("retrieved_sources", [])
    answer = result.get("answer", "")
    retrieval_time_sec = float(result.get("retrieval_time_sec", 0.0))
    generation_time_sec = float(result.get("generation_time_sec", 0.0))
    total_time_sec = float(result.get("total_time_sec", retrieval_time_sec + generation_time_sec))
    cache_hit = bool(result.get("cache_hit", False))
    cache_type = result.get("cache_type") if cache_hit else None
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
        "RAG query completed. endpoint=%s session_id=%s rag_status=%s retrieval_time_sec=%.4f generation_time_sec=%.4f total_time_sec=%.4f queue_wait_time_sec=%.4f active_llm_requests=%d waiting_rag_requests=%d model_name=%s top_k=%d prompt_version=%s index_version=%s cache_hit=%s",
        endpoint,
        active_session_id,
        rag_status,
        retrieval_time_sec,
        generation_time_sec,
        time.perf_counter() - start,
        limiter_metrics.queue_wait_time_sec,
        limiter_metrics.active_llm_requests,
        limiter_metrics.waiting_rag_requests,
        GENERAL_GENERATION_MODEL,
        actual_top_k,
        RAG_PROMPT_VERSION,
        RAG_INDEX_VERSION,
        cache_hit,
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
        "queue_wait_time_sec": limiter_metrics.queue_wait_time_sec,
        "active_llm_requests": limiter_metrics.active_llm_requests,
        "waiting_rag_requests": limiter_metrics.waiting_rag_requests,
        "cache_hit": cache_hit,
        "cache_type": cache_type,
        "model_name": GENERAL_GENERATION_MODEL,
        "top_k": actual_top_k,
        "prompt_version": RAG_PROMPT_VERSION,
        "index_version": RAG_INDEX_VERSION,
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
