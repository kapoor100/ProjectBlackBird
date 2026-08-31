from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider
from blackbird.contracts.provider_ballot import ProviderBallot
from blackbird.providers.prompts import (
    REASONING_SYSTEM_PROMPT,
    VOTING_SYSTEM_PROMPT,
)


class OpenAIReasoningResult(BaseModel):
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        client: AsyncOpenAI | None = None,
    ):
        self.model = model
        self.client = client or AsyncOpenAI()

    async def reason(self, prompt: str) -> ReasoningResponse:
        completion = await self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": REASONING_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=OpenAIReasoningResult,
        )

        message = completion.choices[0].message

        if message.refusal:
            raise RuntimeError(f"OpenAI refused the request: {message.refusal}")

        if message.parsed is None:
            raise RuntimeError("OpenAI returned no structured reasoning result.")

        return ReasoningResponse(
            provider="openai",
            self_confidence=message.parsed.self_confidence,
            response=message.parsed.response,
        )

    async def vote(self, prompt: str) -> ProviderBallot:
        completion = await self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": VOTING_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=OpenAIBallotResult,
        )

        message = completion.choices[0].message

        if message.refusal:
            raise RuntimeError(
                f"OpenAI refused the ballot request: {message.refusal}"
            )

        if message.parsed is None:
            raise RuntimeError("OpenAI returned no structured ballot.")

        return ProviderBallot(
            voter="openai",
            candidate_id=message.parsed.candidate_id,
            selection_confidence=(
                message.parsed.selection_confidence
            ),
            rationale=message.parsed.rationale,
        )


class OpenAIBallotResult(BaseModel):
    candidate_id: Literal["A", "B", "C"]
    selection_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str