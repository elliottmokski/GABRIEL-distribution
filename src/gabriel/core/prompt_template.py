"""Prompt template utilities."""
from dataclasses import dataclass
from jinja2 import Template
from typing import Dict


@dataclass
class PromptTemplate:
    """Simple Jinja2-based prompt template."""

    text: str

    def render(self, **params: Dict[str, str]) -> str:
        """Render the template with the given parameters."""
        return Template(self.text).render(**params)
