import asyncio

from blackbird.contracts.blackbird_result import BlackbirdResult
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.contracts.reasoning_round import ReasoningRound
from blackbird.providers.base import BaseProvider


class BlackbirdCoordinator:
    def __init__(
        self,
        providers: list[BaseProvider],
        confidence_threshold: float = 0.8,
    ):
        self.providers = providers
        self.confidence_threshold = confidence_threshold

    async def _run_round(
        self,
        prompt: str,
        round_number: int,
    ) -> ReasoningRound:
        responses: list[ReasoningResponse] = list(
            await asyncio.gather(
                *(
                    provider.reason(prompt)
                    for provider in self.providers
                )
            )
        )

        selected_response = max(
            responses,
            key=lambda response: response.self_confidence,
        )

        return ReasoningRound(
            round_number=round_number,
            responses=responses,
            selected_response=selected_response,
            threshold_met=(
                selected_response.self_confidence
                >= self.confidence_threshold
            ),
        )

    def _build_challenge_prompt(
        self,
        original_prompt: str,
        previous_round: ReasoningRound,
    ) -> str:
        responses = "\n\n".join(
            (
                f"Provider: {response.provider}\n"
                f"Self-confidence: {response.self_confidence}\n"
                f"Response:\n{response.response}"
            )
            for response in previous_round.responses
        )

        return (
            f"Original request:\n{original_prompt}\n\n"
            f"Independent responses:\n\n{responses}\n\n"
            "Critically evaluate the responses above. "
            "Identify agreements, disagreements, unsupported assumptions, "
            "and possible shared errors. Then provide your strongest revised "
            "answer. Calibrate self-confidence based on evidence and reasoning, "
            "not merely on agreement with another provider."
        )

    async def reason(self, prompt: str) -> BlackbirdResult:
        first_round = await self._run_round(
            prompt=prompt,
            round_number=1,
        )

        challenge_prompt = self._build_challenge_prompt(
            original_prompt=prompt,
            previous_round=first_round,
        )

        second_round = await self._run_round(
            prompt=challenge_prompt,
            round_number=2,
        )

        selected_response = max(
            second_round.responses,
            key=lambda response: response.self_confidence,
        )

        return BlackbirdResult(
            rounds=[first_round, second_round],
            selected_response=selected_response,
            threshold_met=(
                selected_response.self_confidence
                >= self.confidence_threshold
            ),
        )