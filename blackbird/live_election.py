import argparse
import asyncio
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from blackbird.coordinator import BlackbirdCoordinator
from blackbird.providers.anthropic_provider import AnthropicProvider
from blackbird.providers.gemini_provider import GeminiProvider
from blackbird.providers.openai_provider import OpenAIProvider
from blackbird.trials import TrialRecorder, build_trial_record


DEFAULT_PROMPT = (
    "Should production systems use multiple independent LLMs "
    "for high-impact decisions? Explain the tradeoffs."
)


def positive_integer(value: str) -> int:
    parsed_value = int(value)

    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")

    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and record live Blackbird election trials."
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=positive_integer, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/blackbird_trials.jsonl"),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    providers = [
        OpenAIProvider(),
        AnthropicProvider(),
        GeminiProvider(),
    ]
    coordinator = BlackbirdCoordinator(
        providers=providers,
        confidence_threshold=0.8,
        minimum_responses=3,
    )
    provider_models = {
        "openai": providers[0].model,
        "anthropic": providers[1].model,
        "gemini": providers[2].model,
    }
    recorder = TrialRecorder(args.output)

    for run_number in range(1, args.runs + 1):
        start = perf_counter()
        result = await coordinator.reason(args.prompt)
        elapsed = perf_counter() - start

        record = build_trial_record(
            original_prompt=args.prompt,
            provider_models=provider_models,
            elapsed_seconds=elapsed,
            result=result,
        )
        recorder.append(record)

        print(f"\n========== TRIAL {run_number}/{args.runs} ==========")
        print(f"Trial ID: {record.trial_id}")
        print(f"Self-votes: {record.self_vote_count}")

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

    print(f"\nRecorded {args.runs} trial(s) in {args.output}")


if __name__ == "__main__":
    load_dotenv(".env", override=True)
    asyncio.run(main(parse_args()))
