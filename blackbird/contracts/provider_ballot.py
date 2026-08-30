from typing import Literal

from pydantic import BaseModel, Field


class ProviderBallot(BaseModel):
    voter: str = Field(min_length=1)
    candidate_id: Literal["A", "B", "C"]
    selection_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)