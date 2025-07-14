from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class RatingsConfig:
    """Configuration for :class:`Ratings`."""

    attributes: Dict[str, str]
    model: str = "gpt-3.5-turbo"
    n_parallels: int = 50
    save_path: str = "ratings.csv"
    use_dummy: bool = False
    timeout: float = 60.0


class Ratings:
    """Rate passages on a set of attributes."""

    def __init__(self, cfg: RatingsConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    @staticmethod
    def _parse_json(txt: str) -> Optional[List[Dict[str, str]]]:
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                return data.get("data")
        except Exception:
            pass
        return None

    async def run(self, texts: List[str]) -> pd.DataFrame:
        prompts: List[str] = []
        ids: List[str] = []
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

        ratings_data = []
        for resp in df["Response"]:
            main = resp[0] if isinstance(resp, list) and resp else ""
            parsed = self._parse_json(main) or []
            ratings_data.append(parsed)

        df = df.assign(ratings=ratings_data)
        return df
