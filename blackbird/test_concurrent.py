import asyncio
from time import perf_counter

from dotenv import load_dotenv

from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.anthropic_provider import AnthropicProvider
from blackbird.providers.openai_provider import OpenAIProvider


load_dotenv(".env", override=True)


async def main():
    coordinator = BlackbirdCoordinator(
        providers=[
            OpenAIProvider(),
            AnthropicProvider(),
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
            print(f"Confidence: {response.confidence_score}")
            print(response.response)
dfasdfad selected = result.selected_response

    print("\n========== BLACKBIRD FINAL DECISION ==========")
    print(f"Selected provider: {selected.provider}")
    print(f"Selected confidence: {selected.confidence_score}")
    print(f"Threshold met: {result.threshold_met}")
    print(f"Total runtime: {elapsed:.2f} seconds.")
    print("\nSelected response:")
    print(selected.response)


asyncio.run(main())