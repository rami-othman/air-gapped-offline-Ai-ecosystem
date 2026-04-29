from fastapi import APIRouter, Body, Depends, Header

from ..config import INGEST_API_KEY_HEADER
from ..schemas.admin import JobCreateResponse
from ..schemas.rag import (
    RagIngestRequest,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagSearchRequest,
    RagSearchResponse,
)
from ..services.auth_service import ensure_ingest_access
from ..services import job_service, rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


def _require_ingest_access(
    api_key: str | None = Header(default=None, alias=INGEST_API_KEY_HEADER),
) -> None:
    ensure_ingest_access(provided_api_key=api_key)


def _job_create_response(job) -> JobCreateResponse:
    return JobCreateResponse(
        status="queued",
        job_id=job.job_id,
        job_status=job.status,
        message="Job queued successfully.",
    )


@router.post("/query", response_model=RagQueryResponse)
def query_rag(payload: RagQueryRequest) -> RagQueryResponse:
    result = rag_service.run_query(
        question=payload.question,
        top_k=payload.top_k,
        session_id=payload.session_id,
    )
    return RagQueryResponse(**result)


@router.post("/search", response_model=RagSearchResponse)
def search_rag(payload: RagSearchRequest) -> RagSearchResponse:
    result = rag_service.run_search(
        query=payload.query,
        top_k=payload.top_k,
    )
    return RagSearchResponse(**result)


@router.post("/ingest", response_model=RagIngestResponse)
def ingest_rag(
    payload: RagIngestRequest = Body(default_factory=RagIngestRequest),
    _: None = Depends(_require_ingest_access),
) -> RagIngestResponse:
    result = rag_service.run_ingestion(docs_dir=payload.docs_dir)
    return RagIngestResponse(**result)


@router.post("/ingest/async", response_model=JobCreateResponse)
def ingest_rag_async(
    payload: RagIngestRequest = Body(default_factory=RagIngestRequest),
    _: None = Depends(_require_ingest_access),
) -> JobCreateResponse:
    job = job_service.create_job(
        "ingest_documents",
        rag_service.run_ingestion,
        metadata={"docs_dir": payload.docs_dir},
        docs_dir=payload.docs_dir,
    )
    return _job_create_response(job)
