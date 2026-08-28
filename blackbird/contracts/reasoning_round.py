from pydantic import BaseModel, Field

from blackbird.contracts.provider_failure import ProviderFailure
from blackbird.contracts.reasoning_response import ReasoningResponse


class ReasoningRound(BaseModel):
    round_number: int = Field(ge=1)
    responses: list[ReasoningResponse]
    failures: list[ProviderFailure] = Field(default_factory=list)
    selected_response: ReasoningResponse
    quorum_met: bool
    threshold_met: bool