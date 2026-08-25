import asyncio
import unittest

from blackbird.contracts import ReasoningResponse
from blackbird.coordinator import BlackbirdCoordinator, ReasoningCoordinator
from blackbird.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.prompts: list[str] = []

    async def reason(self, prompt: str) -> ReasoningResponse:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        return ReasoningResponse(
            provider=self.name,
            confidence_score=0.9,
            response=f"{self.name}: {prompt}",
        )


class ReasoningCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_all_providers_in_order(self) -> None:
        first = FakeProvider("first")
        second = FakeProvider("second")
        coordinator = ReasoningCoordinator([first, second])

        responses = await coordinator.reason("test prompt")

        self.assertEqual([item.provider for item in responses], ["first", "second"])
        self.assertEqual(first.prompts, ["test prompt"])
        self.assertEqual(second.prompts, ["test prompt"])

    async def test_rejects_blank_prompt(self) -> None:
        coordinator = ReasoningCoordinator([FakeProvider("fake")])

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            await coordinator.reason("   ")

    def test_requires_a_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one provider"):
            ReasoningCoordinator([])


class BlackbirdCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_two_rounds_and_selects_best_response(self) -> None:
        first = FakeProvider("first")
        second = FakeProvider("second")
        coordinator = BlackbirdCoordinator([first, second])

        result = await coordinator.reason("test prompt")

        self.assertEqual([item.round_number for item in result.rounds], [1, 2])
        self.assertEqual(result.selected_response.provider, "first")
        self.assertTrue(result.threshold_met)
        self.assertEqual(len(first.prompts), 2)
        self.assertEqual(len(second.prompts), 2)
        self.assertIn("Original request:\ntest prompt", first.prompts[1])

    def test_rejects_invalid_confidence_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            BlackbirdCoordinator([FakeProvider("fake")], 1.1)


if __name__ == "__main__":
    unittest.main()
