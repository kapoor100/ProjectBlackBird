from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from typing import Literal
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider
from blackbird.contracts.provider_ballot import ProviderBallot


class AnthropicReasoningResult(BaseModel):
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str


class AnthropicBallotResult(BaseModel):
    candidate_id: Literal["A", "B", "C"]
    selection_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: AsyncAnthropic | None = None,
    ):
        self.model = model
        self.client = client or AsyncAnthropic()

    async def reason(self, prompt: str) -> ReasoningResponse:
        message = await self.client.messages.parse(
            model=self.model,
            max_tokens=2048,
            system=(
                "Analyze the user's request carefully. "
                "Return a concise answer and a confidence score "
                "between 0.0 and 1.0. "
                "Keep the response under 700 words."
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            output_format=AnthropicReasoningResult,
        )

        if message.parsed_output is None:
            raise RuntimeError(
                "Anthropic returned no structured reasoning result."
            )

        return ReasoningResponse(
            provider="anthropic",
            self_confidence=message.parsed_output.self_confidence,
            response=message.parsed_output.response
        )

    async def vote(self, prompt: str) -> ProviderBallot:
        message = await self.client.messages.parse(
            model=self.model,
            max_tokens=512,
            system=(
                "Act as an impartial evaluator. Review the anonymous candidates "
                "and select the best-supported answer. Return only the requested "
                "structured ballot with a selection confidence between 0.0 and 1.0 "
                "and a concise rationale."
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            output_format=AnthropicBallotResult,
        )

        if message.parsed_output is None:
            raise RuntimeError(
                "Anthropic returned no structured ballot."
            )

        return ProviderBallot(
            voter="anthropic",
            candidate_id=message.parsed_output.candidate_id,
            selection_confidence=(
                message.parsed_output.selection_confidence
            ),
            rationale=message.parsed_output.rationale,
        )
