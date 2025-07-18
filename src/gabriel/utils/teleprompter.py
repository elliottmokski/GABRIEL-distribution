from __future__ import annotations

import os
from typing import Dict, List, Union

from jinja2 import Environment, FileSystemLoader


class Teleprompter:
    """Lightweight template renderer for prompt Jinja files."""

    def __init__(self, prompt_path: str | None = None) -> None:
        if prompt_path is None:
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
        self.env = Environment(loader=FileSystemLoader(os.path.abspath(prompt_path)))

    def generic_elo_prompt(
        self,
        *,
        text_circle: str,
        text_square: str,
        attributes: Union[Dict[str, str], List[str]],
        instructions: str = "",
        additional_guidelines: str = "",
    ) -> str:
        """Render the generic elo comparison prompt."""
        if isinstance(attributes, list):
            attributes = {a: a for a in attributes}
        template = self.env.get_template("generic_elo_prompt.jinja2")
        return template.render(
            text_circle=text_circle,
            text_square=text_square,
            attributes=attributes,
            instructions=instructions,
            additional_guidelines=additional_guidelines,
        )
