from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=50)
    session_id: str | None = Field(default=None)


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    session_id: str


class RagSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class RagSearchResponse(BaseModel):
    chunks: list[RetrievedChunk]
    retrieval_time_sec: float
    generation_time_sec: float = 0.0
    total_time_sec: float


class RagIngestRequest(BaseModel):
    docs_dir: str | None = None


class RagIngestResponse(BaseModel):
    docs_dir: str
    documents_ingested: int
    chunks_ingested: int
    total_time_sec: float


class ErrorResponse(BaseModel):
    error: str
    message: str
