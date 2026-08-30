import asyncio
import unittest

from blackbird.contracts import ReasoningResponse
from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.base import BaseProvider
from typing import Literal
from blackbird.contracts import ProviderBallot


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
        first = FakeProvider("first", 0.80)
        second = FakeProvider("second", 0.85)
        third = FakeProvider("third", 0.95)

        coordinator = BlackbirdCoordinator(
            [first, second, third],
            confidence_threshold=0.8,
            minimum_responses=3,
        )

        result = await coordinator.reason("test prompt")

        self.assertEqual(
            [item.round_number for item in result.rounds],
            [1, 2],
        )
        self.assertEqual(result.selected_response.provider, "third")
        self.assertTrue(result.quorum_met)
        self.assertTrue(result.threshold_met)

        self.assertEqual(len(first.prompts), 2)
        self.assertEqual(len(second.prompts), 2)
        self.assertEqual(len(third.prompts), 2)

        self.assertIn(
            "Original request:\ntest prompt",
            first.prompts[1],
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


if __name__ == "__main__":
    unittest.main()