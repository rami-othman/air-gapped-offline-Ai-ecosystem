from fastapi import APIRouter, Body, Depends, Header

from ..config import INGEST_API_KEY_HEADER
from ..schemas.rag import (
    RagIngestRequest,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagSearchRequest,
    RagSearchResponse,
)
from ..services.auth_service import ensure_ingest_access
from ..services import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


def _require_ingest_access(
    api_key: str | None = Header(default=None, alias=INGEST_API_KEY_HEADER),
) -> None:
    ensure_ingest_access(provided_api_key=api_key)


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
