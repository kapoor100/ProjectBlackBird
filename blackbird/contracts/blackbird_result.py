from pydantic import BaseModel

from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.contracts.reasoning_round import ReasoningRound


class BlackbirdResult(BaseModel):
    rounds: list[ReasoningRound]
    selected_response: ReasoningResponse
    threshold_met: bool