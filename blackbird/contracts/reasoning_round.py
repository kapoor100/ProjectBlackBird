from pydantic import BaseModel, Field

from blackbird.contracts.reasoning_response import ReasoningResponse


class ReasoningRound(BaseModel):
    round_number: int = Field(ge=1)
    responses: list[ReasoningResponse]
    selected_response: ReasoningResponse
    threshold_met: bool