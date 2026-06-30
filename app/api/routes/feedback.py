from fastapi import APIRouter

from ..schemas.feedback import FeedbackUpdateRequest, FeedbackUpdateResponse
from ..services import feedback_service

router = APIRouter(prefix="/api/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackUpdateResponse)
def update_feedback(payload: FeedbackUpdateRequest) -> FeedbackUpdateResponse:
    result = feedback_service.update_feedback(
        interaction_id=payload.interaction_id,
        helpful=payload.helpful,
    )
    return FeedbackUpdateResponse(**result)
