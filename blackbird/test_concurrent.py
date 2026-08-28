import asyncio
from time import perf_counter

from dotenv import load_dotenv

from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.anthropic_provider import AnthropicProvider
from blackbird.providers.gemini_provider import GeminiProvider
from blackbird.providers.openai_provider import OpenAIProvider


load_dotenv(".env", override=True)


async def main():
    coordinator = BlackbirdCoordinator(
        providers=[
            OpenAIProvider(),
            AnthropicProvider(),
            GeminiProvider(),
        ],
        confidence_threshold=0.8,
    )

    start = perf_counter()

    result = await coordinator.reason(
        "Should production systems use multiple independent LLMs "
        "for high-impact decisions? Explain the tradeoffs."
    )

    elapsed = perf_counter() - start

    for reasoning_round in result.rounds:
        print(f"\n========== ROUND {reasoning_round.round_number} ==========")

        for response in reasoning_round.responses:
            print(f"\n--- {response.provider.upper()} ---")
            print(f"Confidence: {response.self_confidence}")
            print(response.response)

        print(f"\nQuorum met: {reasoning_round.quorum_met}")

        for failure in reasoning_round.failures:
            print(f"FAILED: {failure.provider} — "
            f"{failure.error_type}: {failure.message}"
        )

    selected = result.selected_response

    print("\n========== BLACKBIRD FINAL DECISION ==========")
    print(f"Selected provider: {selected.provider}")
    print(f"Selected confidence: {selected.self_confidence}")
    print(f"Final quorum met: {result.quorum_met}")
    print(f"Threshold met: {result.threshold_met}")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("\nSelected response:")
    print(selected.response)


if __name__ == "__main__":
    asyncio.run(main())