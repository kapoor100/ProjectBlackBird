from abc import ABC, abstractmethod

from blackbird.contracts.reasoning_response import ReasoningResponse


class BaseProvider(ABC):
    @abstractmethod
    async def reason(self, prompt: str) -> ReasoningResponse:
        """Return a normalized response for *prompt*."""
        raise NotImplementedError
