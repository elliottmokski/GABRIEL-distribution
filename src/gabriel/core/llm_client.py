"""Abstract LLM client interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os

from ..utils.openai_utils import get_all_responses, get_response


class LLMClient(ABC):
    """Minimal interface for language model providers."""

    @abstractmethod
    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronously call the LLM with provided messages."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Concrete LLM client leveraging :func:`get_response`."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        # ``get_response`` manages a shared client; API key via env or arg
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        responses, _ = await get_response(prompt, **kwargs)
        return responses[0] if responses else ""

    async def get_all_responses(self, **kwargs: Any):
        """Convenience wrapper around :func:`get_all_responses`."""
        return await get_all_responses(**kwargs)
