from pydantic import BaseModel, Field


class ReasoningResponse(BaseModel):
    provider: str
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str