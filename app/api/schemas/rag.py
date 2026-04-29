from typing import Any

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    status: str
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: ServiceHealth
    chroma: ServiceHealth


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=50)
    session_id: str | None = Field(default=None)


class RagQueryResponse(BaseModel):
    status: str
    answer: str
    sources: list[str]
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    session_id: str
    interaction_id: str | None = None
    queue_wait_time_sec: float | None = None
    active_llm_requests: int | None = None
    waiting_rag_requests: int | None = None
    cache_hit: bool | None = None
    cache_type: str | None = None
    model_name: str | None = None
    top_k: int | None = None
    prompt_version: str | None = None
    index_version: str | None = None


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User retrieval query")
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class RagSearchResponse(BaseModel):
    status: str
    chunks: list[RetrievedChunk]
    retrieval_time_sec: float
    generation_time_sec: float = 0.0
    total_time_sec: float


class RagIngestRequest(BaseModel):
    docs_dir: str | None = None


class RagIngestResponse(BaseModel):
    status: str
    docs_dir: str
    documents_ingested: int
    chunks_ingested: int
    total_time_sec: float


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    message: str
    details: dict[str, Any] | None = None
