"""Prompt template utilities."""
from dataclasses import dataclass
from importlib import resources
from jinja2 import Template
from typing import Dict

@dataclass
class PromptTemplate:
    """Simple Jinja2-based prompt template."""

    text: str

    def render(self, **params: Dict[str, str]) -> str:
        """Render the template with the given parameters."""
        return Template(self.text).render(**params)

    @classmethod
    def from_package(
        cls,
        filename: str,
        package: str = "gabriel.prompts",
    ) -> "PromptTemplate":
        """Load a template from the given package file."""
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        return cls(text)
