"""Abstract LLM client interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMClient(ABC):
    """Minimal interface for language model providers."""

    @abstractmethod
    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronously call the LLM with provided messages."""
        raise NotImplementedError
