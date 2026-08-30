import asyncio

from blackbird.contracts.blackbird_result import BlackbirdResult
from blackbird.contracts.provider_failure import ProviderFailure
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.contracts.reasoning_round import ReasoningRound
from blackbird.providers.base import BaseProvider


class BlackbirdCoordinator:
    def __init__(
        self,
        providers: list[BaseProvider],
        confidence_threshold: float = 0.8,
        minimum_responses: int = 3,
    ):
        if not providers:
            raise ValueError("At least one provider is required.")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        if minimum_responses < 1:
            raise ValueError("minimum_responses must be at least 1.")

        if minimum_responses > len(providers):
            raise ValueError(
                "minimum_responses cannot exceed "
                "the configured provider count."
            )
        self.providers = providers
        self.confidence_threshold = confidence_threshold
        self.minimum_responses = minimum_responses

    @staticmethod
    def _provider_name(provider: BaseProvider) -> str:
        return (
            provider.__class__.__name__
            .removesuffix("Provider")
            .lower()
        )

    async def _run_round(
        self,
        prompt: str,
        round_number: int,
    ) -> ReasoningRound:
        results = await asyncio.gather(
            *(provider.reason(prompt) for provider in self.providers),
            return_exceptions=True,
        )

        responses: list[ReasoningResponse] = []
        failures: list[ProviderFailure] = []

        for provider, result in zip(self.providers, results):
            if isinstance(result, asyncio.CancelledError):
                raise result

            if isinstance(result, Exception):
                failures.append(
                    ProviderFailure(
                        provider=self._provider_name(provider),
                        error_type=type(result).__name__,
                        message=str(result),
                    )
                )
            elif not isinstance(result, ReasoningResponse):
                failures.append(
                    ProviderFailure(
                        provider=self._provider_name(provider),
                        error_type="InvalidProviderResponse",
                        message=(
                            "Expected ReasoningResponse, "
                            f"received {type(result).__name__}."
                        ),
                    )
                )
            else:
                responses.append(result)
        if not responses:
            raise RuntimeError(
                f"All providers failed during round {round_number}."
            )

        selected_response = max(
            responses,
            key=lambda response: response.self_confidence,
        )

        quorum_met = len(responses) >= self.minimum_responses

        return ReasoningRound(
            round_number=round_number,
            responses=responses,
            failures=failures,
            selected_response=selected_response,
            quorum_met=quorum_met,
            threshold_met=(
                quorum_met
                and selected_response.self_confidence
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
            "answer. Calibrate self-confidence based on evidence and reasoning,"
            "not merely on agreement with another provider."
        )

    async def reason(self, prompt: str) -> BlackbirdResult:
        if not prompt or not prompt.strip(): raise ValueError("Prompt must not be empty.")
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

        selected_response = second_round.selected_response
        quorum_met = first_round.quorum_met and second_round.quorum_met

        return BlackbirdResult(
            rounds=[first_round, second_round],
            selected_response=selected_response,
            quorum_met=quorum_met,
            threshold_met=(
                quorum_met
                and selected_response.self_confidence
                >= self.confidence_threshold
            ),
        )