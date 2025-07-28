"""Prompt template utilities."""
from dataclasses import dataclass
from importlib import resources
from jinja2 import Environment, Template
import random
from typing import Dict


def _shuffled_dict(value) -> Dict[str, str]:
    """Return a new dict with items in random order."""
    if isinstance(value, dict):
        items = list(value.items())
        random.shuffle(items)
        return dict(items)
    else:
        items = list(value)
        random.shuffle(items)
        return {k: k for k in items}


def _shuffled(value):
    if isinstance(value, dict):
        items = list(value.keys())
    else:
        items = list(value)
    random.shuffle(items)
    return items

@dataclass
class PromptTemplate:
    """Simple Jinja2-based prompt template."""

    text: str

    def render(self, **params: Dict[str, str]) -> str:
        """Render the template with the given parameters."""
        attrs = params.get("attributes")
        descs = params.get("descriptions")
        if isinstance(attrs, list):
            if isinstance(descs, list) and len(descs) == len(attrs):
                params["attributes"] = {a: d for a, d in zip(attrs, descs)}
            else:
                params["attributes"] = {a: a for a in attrs}
        env = Environment()
        env.filters["shuffled_dict"] = _shuffled_dict
        env.filters["shuffled"] = _shuffled
        template = env.from_string(self.text)
        return template.render(**params)

    @classmethod
    def from_package(
        cls,
        filename: str,
        package: str = "gabriel.prompts",
    ) -> "PromptTemplate":
        """Load a template from the given package file."""
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        return cls(text)
