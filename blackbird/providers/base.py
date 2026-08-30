from abc import ABC, abstractmethod

from blackbird.contracts.provider_ballot import ProviderBallot
from blackbird.contracts.reasoning_response import ReasoningResponse


class BaseProvider(ABC):
    @abstractmethod
    async def reason(self, prompt: str) -> ReasoningResponse:
        """Return a normalized response for *prompt*."""
        raise NotImplementedError

    @abstractmethod
    async def vote(self, prompt: str) -> ProviderBallot:
        """Return a normalized anonymous ballot for *prompt*."""
        raise NotImplementedError