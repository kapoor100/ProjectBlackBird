from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider
from blackbird.contracts.provider_ballot import ProviderBallot

class OpenAIReasoningResult(BaseModel):
    candidate_id: Literal["A", "B", "C"]
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
                    "content": (
                        "Analyze the user's request carefully. "
                        "Return a concise answer and a confidence score "
                        "between 0.0 and 1.0."
                    ),
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
                    "content": (
                        "Act as an impartial anonymous evaluator. "
                        "Evaluate candidate answers only for correctness, "
                        "evidence, relevance, and completeness. "
                        "Do not guess who authored them. Select exactly one "
                        "candidate: A, B, or C. Return the candidate ID, "
                        "a calibrated selection confidence between 0.0 and "
                        "1.0, and a concise rationale."
                    ),
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