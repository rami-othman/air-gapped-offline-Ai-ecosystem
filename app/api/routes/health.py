from fastapi import APIRouter

from ..schemas.rag import HealthResponse
from ..services.health_service import get_health_payload

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    payload = get_health_payload()
    return HealthResponse(**payload)
