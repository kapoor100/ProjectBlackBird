from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.providers.base import BaseProvider


class AnthropicReasoningResult(BaseModel):
    self_confidence: float = Field(ge=0.0, le=1.0)
    response: str


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
            max_tokens=1024,
            system=(
                "Analyze the user's request carefully. "
                "Return a concise answer and a confidence score "
                "between 0.0 and 1.0."
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
