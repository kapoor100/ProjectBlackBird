from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider
from blackbird.contracts.provider_ballot import ProviderBallot
from blackbird.providers.prompts import (
    REASONING_SYSTEM_PROMPT,
    VOTING_SYSTEM_PROMPT,
)


class GeminiReasoningResult(BaseModel):
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str


class GeminiBallotResult(BaseModel):
    candidate_id: Literal["A", "B", "C"]
    selection_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        client: genai.Client | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()

    async def reason(self, prompt: str) -> ReasoningResponse:
        result = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    REASONING_SYSTEM_PROMPT
                ),
                response_mime_type="application/json",
                response_schema=GeminiReasoningResult,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        if result.parsed is None:
            raise RuntimeError(
                "Gemini returned no structured reasoning result."
            )

        parsed = GeminiReasoningResult.model_validate(result.parsed)

        return ReasoningResponse(
            provider="gemini",
            self_confidence=parsed.self_confidence,
            response=parsed.response,
        )

    async def vote(self, prompt: str) -> ProviderBallot:

        result = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=VOTING_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=GeminiBallotResult,
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                ),
            ),
        )

        if result.parsed is None:
            raise RuntimeError(
                "Gemini returned no structured ballot."
            )

        parsed = GeminiBallotResult.model_validate(result.parsed)

        return ProviderBallot(
            voter="gemini",
            candidate_id=parsed.candidate_id,
            selection_confidence=parsed.selection_confidence,
            rationale=parsed.rationale,
        )
