import asyncio
import unittest
from typing import Literal
from unittest.mock import patch

from blackbird.contracts import ProviderBallot, ReasoningResponse
from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        confidence: float = 0.9,
        vote_for: Literal["A", "B", "C"] = "A",
        vote_confidence: float = 0.9,
    ) -> None:
        self.name = name
        self.confidence = confidence
        self.vote_for = vote_for
        self.vote_confidence = vote_confidence
        self.prompts: list[str] = []
        self.vote_prompts: list[str] = []

    async def reason(self, prompt: str) -> ReasoningResponse:
        self.prompts.append(prompt)
        await asyncio.sleep(0)

        return ReasoningResponse(
            provider=self.name,
            self_confidence=self.confidence,
            response=f"{self.name}: {prompt}",
        )

    async def vote(self, prompt: str) -> ProviderBallot:
        self.vote_prompts.append(prompt)
        await asyncio.sleep(0)

        return ProviderBallot(
            voter=self.name,
            candidate_id=self.vote_for,
            selection_confidence=self.vote_confidence,
            rationale=f"{self.name} selected {self.vote_for}.",
        )


class FailingVoter(FakeProvider):
    async def vote(self, prompt: str) -> ProviderBallot:
        await asyncio.sleep(0)
        raise RuntimeError("Simulated ballot failure.")


class MalformedBallotProvider(FakeProvider):
    async def vote(self, prompt: str) -> ProviderBallot:
        await asyncio.sleep(0)
        return {"unexpected": "value"}  # type: ignore[return-value]


class MalformedProvider(BaseProvider):
    async def reason(self, prompt: str) -> ReasoningResponse:
        await asyncio.sleep(0)
        return {"unexpected": "value"}  # type: ignore[return-value]

    async def vote(self, prompt: str) -> ProviderBallot:
        raise AssertionError(
            "Voting must not run without reasoning quorum."
        )


class FailingProvider(BaseProvider):
    async def reason(self, prompt: str) -> ReasoningResponse:
        await asyncio.sleep(0)
        raise RuntimeError("Simulated provider failure.")

    async def vote(self, prompt: str) -> ProviderBallot:
        raise AssertionError(
            "Voting must not run without reasoning quorum."
        )


class BlackbirdCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_two_rounds_and_selects_best_response(self) -> None:
        first = FakeProvider("first", 0.80, vote_for="C")
        second = FakeProvider("second", 0.85, vote_for="C")
        third = FakeProvider("third", 0.95, vote_for="A")

        coordinator = BlackbirdCoordinator(
            [first, second, third],
            confidence_threshold=0.8,
            minimum_responses=3,
        )

        with patch(
            "blackbird.coordinator.random.SystemRandom.shuffle",
            return_value=None,
        ):
            result = await coordinator.reason("test prompt")

        self.assertEqual(
            [item.round_number for item in result.rounds],
            [1, 2],
        )

    async def test_provider_failure_prevents_quorum(self) -> None:
        coordinator = BlackbirdCoordinator(
            [
                FakeProvider("first"),
                FailingProvider(),
                FakeProvider("third"),
            ],
            minimum_responses=3,
        )

        result = await coordinator.reason("test prompt")

        self.assertFalse(result.quorum_met)
        self.assertFalse(result.threshold_met)

        for reasoning_round in result.rounds:
            self.assertEqual(len(reasoning_round.responses), 2)
            self.assertEqual(len(reasoning_round.failures), 1)
            self.assertEqual(
                reasoning_round.failures[0].provider,
                "failing",
            )

    async def test_rejects_blank_prompt(self) -> None:
        coordinator = BlackbirdCoordinator(
            [FakeProvider("fake")],
            minimum_responses=1,
        )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            await coordinator.reason("   ")

    async def test_malformed_provider_response_is_isolated(self) -> None:
        coordinator = BlackbirdCoordinator(
            [
                FakeProvider("first"),
                MalformedProvider(),
                FakeProvider("third"),
            ],
            minimum_responses=3,
        )

        result = await coordinator.reason("test prompt")

        self.assertFalse(result.quorum_met)
        self.assertFalse(result.threshold_met)

        for reasoning_round in result.rounds:
            self.assertEqual(len(reasoning_round.responses), 2)
            self.assertEqual(len(reasoning_round.failures), 1)
            self.assertEqual(
                reasoning_round.failures[0].error_type,
                "InvalidProviderResponse",
            )                     

    def test_requires_a_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one provider"):
            BlackbirdCoordinator([])

    def test_rejects_invalid_confidence_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            BlackbirdCoordinator(
                [FakeProvider("fake")],
                confidence_threshold=1.1,
                minimum_responses=1,
            )

    def test_rejects_impossible_quorum(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "cannot exceed",
        ):
            BlackbirdCoordinator(
                [FakeProvider("fake")],
                minimum_responses=3,
            )

    def test_ballot_prompt_anonymizes_provider_identity(self) -> None:
        responses = [
            ReasoningResponse(
                provider="openai",
                self_confidence=0.8,
                response="Answer alpha.",
            ),
            ReasoningResponse(
                provider="anthropic",
                self_confidence=0.9,
                response="Answer beta.",
            ),
            ReasoningResponse(
                provider="gemini",
                self_confidence=0.7,
                response="Answer gamma.",
            ),
        ]

        candidates, prompt = (
            BlackbirdCoordinator._build_ballot_prompt(
                "test prompt",
                responses,
            )
        )

        self.assertEqual(set(candidates), {"A", "B", "C"})
        self.assertEqual(
            {item.provider for item in candidates.values()},
            {"openai", "anthropic", "gemini"},
        )

        self.assertNotIn("openai", prompt)
        self.assertNotIn("anthropic", prompt)
        self.assertNotIn("gemini", prompt)

        for response in responses:
            self.assertIn(response.response, prompt)

    async def test_collects_provider_ballots(self) -> None:
        first = FakeProvider("first", vote_for="A")
        second = FakeProvider("second", vote_for="B")
        third = FakeProvider("third", vote_for="A")

        coordinator = BlackbirdCoordinator(
            [first, second, third],
            minimum_responses=3,
        )

        ballots, failures, quorum_met = (
            await coordinator._run_vote("ballot prompt")
        )

        self.assertEqual(
            [ballot.voter for ballot in ballots],
            ["first", "second", "third"],
        )
        self.assertEqual(
            [ballot.candidate_id for ballot in ballots],
            ["A", "B", "A"],
        )
        self.assertEqual(failures, [])
        self.assertTrue(quorum_met)

        self.assertEqual(first.vote_prompts, ["ballot prompt"])
        self.assertEqual(second.vote_prompts, ["ballot prompt"])
        self.assertEqual(third.vote_prompts, ["ballot prompt"])

    async def test_anonymizes_challenge_round(self) -> None:
        first = FakeProvider("first")
        second = FakeProvider("second")
        third = FakeProvider("third")

        coordinator = BlackbirdCoordinator(
            [first, second, third],
            minimum_responses=3,
        )

        await coordinator.reason("test prompt")

        challenge_prompt = first.prompts[1]

        self.assertNotIn("Provider:", challenge_prompt)
        self.assertNotIn("Self-confidence:", challenge_prompt)
        self.assertIn(
            "Anonymous independent responses:",
            challenge_prompt,
        )
        self.assertIn("Response 1", challenge_prompt)
        self.assertIn("Response 2", challenge_prompt)
        self.assertIn("Response 3", challenge_prompt)

        self.assertEqual(challenge_prompt, second.prompts[1])
        self.assertEqual(challenge_prompt, third.prompts[1])

    def test_majority_vote_selects_candidate(self) -> None:
        candidates = {
            "A": ReasoningResponse(
                provider="first",
                self_confidence=0.8,
                response="Alpha",
            ),
            "B": ReasoningResponse(
                provider="second",
                self_confidence=0.9,
                response="Beta",
            ),
            "C": ReasoningResponse(
                provider="third",
                self_confidence=0.7,
                response="Gamma",
            ),
        }
        ballots = [
            ProviderBallot(
                voter="first",
                candidate_id="A",
                selection_confidence=0.7,
                rationale="A",
            ),
            ProviderBallot(
                voter="second",
                candidate_id="B",
                selection_confidence=0.9,
                rationale="B",
            ),
            ProviderBallot(
                voter="third",
                candidate_id="A",
                selection_confidence=0.6,
                rationale="A",
            ),
        ]

        winner, vote_counts = (
            BlackbirdCoordinator._select_winning_candidate(
                candidates,
                ballots,
            )
        )

        self.assertEqual(winner, "A")
        self.assertEqual(
            vote_counts,
            {"A": 2, "B": 1, "C": 0},
        )

    def test_confidence_breaks_three_way_vote_tie(self) -> None:
        candidates = {
            "A": ReasoningResponse(
                provider="first",
                self_confidence=0.8,
                response="Alpha",
            ),
            "B": ReasoningResponse(
                provider="second",
                self_confidence=0.9,
                response="Beta",
            ),
            "C": ReasoningResponse(
                provider="third",
                self_confidence=0.7,
                response="Gamma",
            ),
        }
        ballots = [
            ProviderBallot(
                voter="first",
                candidate_id="A",
                selection_confidence=0.6,
                rationale="A",
            ),
            ProviderBallot(
                voter="second",
                candidate_id="B",
                selection_confidence=0.95,
                rationale="B",
            ),
            ProviderBallot(
                voter="third",
                candidate_id="C",
                selection_confidence=0.7,
                rationale="C",
            ),
        ]

        winner, vote_counts = (
            BlackbirdCoordinator._select_winning_candidate(
                candidates,
                ballots,
            )
        )

        self.assertEqual(winner, "B")
        self.assertEqual(
            vote_counts,
            {"A": 1, "B": 1, "C": 1},
        )

    async def test_ballot_failure_prevents_quorum(self) -> None:
        coordinator = BlackbirdCoordinator(
            [
                FakeProvider("first"),
                FailingVoter("second"),
                FakeProvider("third"),
            ],
            minimum_responses=3,
        )

        result = await coordinator.reason("test prompt")

        self.assertIsNone(result.winning_candidate_id)
        self.assertFalse(result.voting_quorum_met)
        self.assertFalse(result.quorum_met)
        self.assertFalse(result.threshold_met)
        self.assertEqual(len(result.ballots), 2)
        self.assertEqual(len(result.voting_failures), 1)
        self.assertEqual(
            result.voting_failures[0].provider,
            "second",
        )
        self.assertEqual(
            result.voting_failures[0].error_type,
            "RuntimeError",
        )

    async def test_malformed_ballot_is_isolated(self) -> None:
        coordinator = BlackbirdCoordinator(
            [
                FakeProvider("first"),
                MalformedBallotProvider("second"),
                FakeProvider("third"),
            ],
            minimum_responses=3,
        )

        result = await coordinator.reason("test prompt")

        self.assertFalse(result.voting_quorum_met)
        self.assertFalse(result.quorum_met)
        self.assertFalse(result.threshold_met)
        self.assertEqual(len(result.ballots), 2)
        self.assertEqual(len(result.voting_failures), 1)
        self.assertEqual(
            result.voting_failures[0].provider,
            "second",
        )
        self.assertEqual(
            result.voting_failures[0].error_type,
            "InvalidProviderBallot",
        )


if __name__ == "__main__":
    unittest.main()