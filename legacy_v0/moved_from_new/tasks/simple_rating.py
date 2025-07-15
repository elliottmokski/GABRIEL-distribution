"""Simple rating task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class RatingConfig:
    attributes: Dict[str, str]
    model: str = "gpt-3.5-turbo"
    n_parallels: int = 50
    save_path: str = "ratings.csv"
    use_dummy: bool = False
    timeout: float = 60.0


class SimpleRating:
    """LLM-based attribute rating for single passages."""

    def __init__(self, cfg: RatingConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    async def predict(self, texts: List[str]) -> pd.DataFrame:
        prompts = []
        ids = []
        for idx, passage in enumerate(texts):
            prompts.append(
                self.template.render(
                    attributes=list(self.cfg.attributes.keys()),
                    descriptions=list(self.cfg.attributes.values()),
                    passage=passage,
                    object_category="text",
                    attribute_category="attributes",
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
