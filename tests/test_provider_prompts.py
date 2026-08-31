import unittest

from blackbird.providers.prompts import VOTING_SYSTEM_PROMPT


class ProviderPromptTests(unittest.TestCase):
    def test_voting_prompt_preserves_anonymity_and_criteria(self) -> None:
        self.assertIn(
            "Do not guess who authored them",
            VOTING_SYSTEM_PROMPT,
        )

        for criterion in (
            "correctness",
            "evidence",
            "relevance",
            "completeness",
        ):
            self.assertIn(criterion, VOTING_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
