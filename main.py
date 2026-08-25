import argparse
import asyncio
import json
from collections.abc import Sequence

from dotenv import load_dotenv

from blackbird import BlackbirdCoordinator
from blackbird.providers import (
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    OpenAIProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask one or more AI providers to reason about a prompt."
    )
    parser.add_argument("prompt", help="The prompt to send")
    parser.add_argument(
        "--provider",
        action="append",
        choices=("openai", "anthropic", "gemini"),
        help="Provider to use; repeat to select multiple (default: all)",
    )
    return parser.parse_args()


def build_providers(names: Sequence[str]) -> list[BaseProvider]:
    factories = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }
    return [factories[name]() for name in names]


async def run(prompt: str, provider_names: Sequence[str]) -> None:
    coordinator = BlackbirdCoordinator(build_providers(provider_names))
    result = await coordinator.reason(prompt)
    print(json.dumps(result.model_dump(), indent=2))


def main() -> None:
    load_dotenv()
    args = parse_args()
    asyncio.run(
        run(args.prompt, args.provider or ("openai", "anthropic", "gemini"))
    )


if __name__ == "__main__":
    main()
