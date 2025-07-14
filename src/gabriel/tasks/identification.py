"""Identification task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class IdentificationConfig:
    classes: Dict[str, str]
    model: str = "gpt-3.5-turbo"
    n_parallels: int = 50
    save_path: str = "identification.csv"
    use_dummy: bool = False
    timeout: float = 60.0


class Identification:
    """Classify entities into user-specified categories using an LLM."""

    def __init__(
        self, cfg: IdentificationConfig, template: PromptTemplate | None = None
    ) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package(
            "generic_classification_prompt.jinja2"
        )

    async def classify(self, entities: List[str]) -> pd.DataFrame:
        prompts = []
        ids = []
        class_defs = "\n".join(
            f"- '{cls}': {desc}" for cls, desc in self.cfg.classes.items()
        )
        for idx, ent in enumerate(entities):
            prompts.append(
                self.template.render(
                    entity_list=ent,
                    possible_classes=list(self.cfg.classes.keys()),
                    class_definitions=class_defs,
                    entity_category="entities",
                    output_format="{\n  \"<entity\": \"<class>\"\n}"
                )
            )
            ids.append(str(idx))

        df = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=self.cfg.save_path,
            use_dummy=self.cfg.use_dummy,
            timeout=self.cfg.timeout,
            json_mode=True,
        )
        return df

