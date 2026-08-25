from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider


class OpenAIReasoningResult(BaseModel):
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
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
