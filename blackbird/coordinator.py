import asyncio
import random
from blackbird.contracts.blackbird_result import BlackbirdResult
from blackbird.contracts.provider_failure import ProviderFailure
from blackbird.contracts.reasoning_response import ReasoningResponse
from blackbird.contracts.reasoning_round import ReasoningRound
from blackbird.contracts.provider_ballot import ProviderBallot
from blackbird.providers.base import BaseProvider

CANDIDATE_IDS = ("A", "B", "C")


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

        if minimum_responses != 3:
            raise ValueError(
                "minimum_responses must be exactly 3."
            )

        if len(providers) != 3:
            raise ValueError("Exactly three providers are required.")

        self.providers = providers
        self.confidence_threshold = confidence_threshold
        self.minimum_responses = minimum_responses

    @staticmethod
    def _provider_name(provider: BaseProvider) -> str:
        configured_name = getattr(provider, "name", None)

        if (
            isinstance(configured_name, str)
            and configured_name.strip()
        ):
            return configured_name.strip()

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
                f"Response {index}\n"
                f"Answer:\n{response.response}"
            )
            for index, response in enumerate(
                previous_round.responses,
                start=1,
            )
        )

        return (
            f"Original request:\n{original_prompt}\n\n"
            f"Anonymous independent responses:\n\n{responses}\n\n"
            "Critically evaluate the responses above. "
            "Identify agreements, disagreements, unsupported assumptions, "
            "and possible shared errors. Then provide your strongest revised "
            "answer. Calibrate self-confidence based on evidence and reasoning,"
            "not merely on agreement with another provider."
        )

    @staticmethod
    def _build_ballot_prompt(
        original_prompt: str,
        responses: list[ReasoningResponse],
    ) -> tuple[dict[str, ReasoningResponse], str]:
        if len(responses) != len(CANDIDATE_IDS):
            raise ValueError(
                "Anonymous voting requires exactly three candidates."
            )

        shuffled_responses = responses.copy()
        random.SystemRandom().shuffle(shuffled_responses)

        candidates = dict(
            zip(
                CANDIDATE_IDS,
                shuffled_responses,
                strict=True,
            )
        )

        candidate_text = "\n\n".join(
            (
                f"Candidate {candidate_id}:\n"
                f"{response.response}"
            )
            for candidate_id, response in candidates.items()
        )

        prompt = (
            f"Original request:\n{original_prompt}\n\n"
            f"Anonymous revised candidates:\n\n"
            f"{candidate_text}\n\n"
            "Select the strongest candidate based on correctness, "
            "evidence, relevance, and completeness. Do not attempt "
            "to identify the author."
        )

        return candidates, prompt

    async def _run_vote(
        self,
        prompt: str,
    ) -> tuple[
        list[ProviderBallot],
        list[ProviderFailure],
        bool,
    ]:
        results = await asyncio.gather(
            *(provider.vote(prompt) for provider in self.providers),
            return_exceptions=True,
        )

        ballots: list[ProviderBallot] = []
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
            elif not isinstance(result, ProviderBallot):
                failures.append(
                    ProviderFailure(
                        provider=self._provider_name(provider),
                        error_type="InvalidProviderBallot",
                        message=(
                            "Expected ProviderBallot, "
                            f"received {type(result).__name__}."
                        ),
                    )
                )
            else:
                ballots.append(
                    result.model_copy(
                        update={
                            "voter": self._provider_name(provider),
                        }
                    )
                )

        quorum_met = len(ballots) >= self.minimum_responses

        return ballots, failures, quorum_met

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

        reasoning_quorum_met = (
            first_round.quorum_met
            and second_round.quorum_met
        )

        if not reasoning_quorum_met:
            return BlackbirdResult(
                rounds=[first_round, second_round],
                selected_response=second_round.selected_response,
                quorum_met=False,
                threshold_met=False,
            )

        candidates, ballot_prompt = self._build_ballot_prompt(
            original_prompt=prompt,
            responses=second_round.responses,
        )

        ballots, voting_failures, voting_quorum_met = (
            await self._run_vote(ballot_prompt)
        )

        winning_candidate_id, vote_counts = (
            self._select_winning_candidate(
                ballots=ballots,
                candidates=candidates,
            )
        )

        if not voting_quorum_met:
            winning_candidate_id = None

        if winning_candidate_id is None:
            selected_response = second_round.selected_response
        else:
            selected_response = candidates[winning_candidate_id]

        quorum_met = (
            reasoning_quorum_met
            and voting_quorum_met
        )

        return BlackbirdResult(
            rounds=[first_round, second_round],
            selected_response=selected_response,
            candidates=candidates,
            ballots=ballots,
            voting_failures=voting_failures,
            vote_counts=vote_counts,
            winning_candidate_id=winning_candidate_id,
            voting_quorum_met=voting_quorum_met,
            quorum_met=quorum_met,
            threshold_met=(
                quorum_met
                and winning_candidate_id is not None
                and selected_response.self_confidence
                >= self.confidence_threshold
            ),
        )

    @staticmethod
    def _select_winning_candidate(
        candidates: dict[str, ReasoningResponse],
        ballots: list[ProviderBallot],
    ) -> tuple[str | None, dict[str, int]]:
        vote_counts = {
            candidate_id: 0
            for candidate_id in candidates
        }
        for ballot in ballots:
            if ballot.candidate_id not in candidates:
                raise ValueError(
                    f"Unknown candidate: {ballot.candidate_id}"
                )

            vote_counts[ballot.candidate_id] += 1

        if not ballots:
            return None, vote_counts

        highest_vote_count = max(vote_counts.values())
        vote_leaders = [
            candidate_id
            for candidate_id, count in vote_counts.items()
            if count == highest_vote_count
        ]

        if len(vote_leaders) == 1:
            return vote_leaders[0], vote_counts

        return None, vote_counts
