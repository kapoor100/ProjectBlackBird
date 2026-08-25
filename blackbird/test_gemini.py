import asyncio

from dotenv import load_dotenv

from blackbird.providers.gemini_provider import GeminiProvider


load_dotenv(".env", override=True)


async def main():
    provider = GeminiProvider()

    result = await provider.reason(
        "Explain why independent LLM agreement can improve reliability."
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
