from pydantic import BaseModel, Field, StrictBool


class FeedbackUpdateRequest(BaseModel):
    interaction_id: str = Field(..., min_length=1, description="Chat interaction ID")
    helpful: StrictBool


class FeedbackUpdateResponse(BaseModel):
    status: str
    interaction_id: str
    helpful: bool
