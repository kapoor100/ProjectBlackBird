import asyncio
from time import perf_counter

from dotenv import load_dotenv

from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.anthropic_provider import AnthropicProvider
from blackbird.providers.gemini_provider import GeminiProvider
from blackbird.providers.openai_provider import OpenAIProvider


load_dotenv(".env", override=True)


async def main() -> None:
    coordinator = BlackbirdCoordinator(
        providers=[
            OpenAIProvider(),
            AnthropicProvider(),
            GeminiProvider(),
        ],
        confidence_threshold=0.8,
        minimum_responses=3,
    )

    prompt = (
        "Should production systems use multiple independent LLMs "
        "for high-impact decisions? Explain the tradeoffs."
    )

    start = perf_counter()
    result = await coordinator.reason(prompt)
    elapsed = perf_counter() - start

    for reasoning_round in result.rounds:
        print(
            f"\n========== ROUND "
            f"{reasoning_round.round_number} =========="
        )
        print(f"Quorum met: {reasoning_round.quorum_met}")

        for response in reasoning_round.responses:
            print(f"\n--- {response.provider.upper()} ---")
            print(f"Confidence: {response.self_confidence}")
            print(response.response)

        for failure in reasoning_round.failures:
            print(
                f"\nFAILED: {failure.provider} — "
                f"{failure.error_type}: {failure.message}"
            )

    print("\n========== ANONYMOUS CANDIDATES ==========")

    for candidate_id, response in result.candidates.items():
        print(
            f"Candidate {candidate_id}: "
            f"{response.provider}"
        )

    print("\n========== BALLOTS ==========")

    for ballot in result.ballots:
        print(
            f"{ballot.voter} voted for "
            f"{ballot.candidate_id} "
            f"({ballot.selection_confidence})"
        )
        print(ballot.rationale)

    for failure in result.voting_failures:
        print(
            f"FAILED BALLOT: {failure.provider} — "
            f"{failure.error_type}: {failure.message}"
        )

    print("\n========== BLACKBIRD FINAL DECISION ==========")
    print(f"Vote counts: {result.vote_counts}")
    print(f"Winning candidate: {result.winning_candidate_id}")
    print(f"Voting quorum met: {result.voting_quorum_met}")
    print(f"Final quorum met: {result.quorum_met}")
    print(f"Threshold met: {result.threshold_met}")
    print(
        "Selected provider: "
        f"{result.selected_response.provider}"
    )
    print(
        "Selected confidence: "
        f"{result.selected_response.self_confidence}"
    )
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("\nSelected response:")
    print(result.selected_response.response)


if __name__ == "__main__":
    asyncio.run(main())