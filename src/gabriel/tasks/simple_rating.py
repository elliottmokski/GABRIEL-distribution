"""Simple rating task."""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SimpleRating:
    """Placeholder class for simple rating functionality."""

    def predict(self, texts: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Dummy predict method."""
        raise NotImplementedError
