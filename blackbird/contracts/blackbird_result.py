from typing import Literal

from pydantic import BaseModel, Field

from blackbird.contracts.provider_ballot import ProviderBallot
from blackbird.contracts.provider_failure import ProviderFailure
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.contracts.reasoning_round import ReasoningRound


class BlackbirdResult(BaseModel):
    rounds: list[ReasoningRound]
    selected_response: ReasoningResponse
    quorum_met: bool
    threshold_met: bool
    candidates: dict[str, ReasoningResponse] = Field(
        default_factory=dict
    )
    ballots: list[ProviderBallot] = Field(default_factory=list)
    voting_failures: list[ProviderFailure] = Field(
        default_factory=list
    )
    vote_counts: dict[str, int] = Field(default_factory=dict)
    winning_candidate_id: Literal["A", "B", "C"] | None = None
    voting_quorum_met: bool = False