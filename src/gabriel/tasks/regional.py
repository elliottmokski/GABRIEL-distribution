from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class RegionalConfig:
    """Configuration for :class:`Regional`."""

    model: str = "o4-mini"
    n_parallels: int = 400
    save_dir: str = os.path.expanduser("~/Documents/runs")
    run_name: str | None = None
    use_dummy: bool = False
    additional_instructions: str = ""
    additional_guidelines: str = ""
    reasoning_effort: str = "medium"
    search_context_size: str = "medium"
    print_example_prompt: bool = True


class Regional:
    """Run regional topic analysis prompts in parallel."""

    def __init__(
        self,
        df: pd.DataFrame,
        region_col: str,
        topics: List[str],
        cfg: RegionalConfig | None = None,
    ) -> None:
        self.df = df.copy()
        self.region_col = region_col
        self.topics = topics
        self.cfg = cfg or RegionalConfig()
        self.cfg.run_name = self.cfg.run_name or f"regional_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(self.cfg.save_dir, self.cfg.run_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.template = PromptTemplate.from_package("regional_analysis_prompt.jinja2")

    def _build(self) -> tuple[List[str], List[str]]:
        prompts: List[str] = []
        ids: List[str] = []
        unique_regions = self.df[self.region_col].astype(str).dropna().unique()
        for region in unique_regions:
            for topic in self.topics:
                prompts.append(
                    self.template.render(
                        region=region,
                        topic=topic,
                        additional_instructions=self.cfg.additional_instructions,
                        additional_guidelines=self.cfg.additional_guidelines,
                    )
                )
                ids.append(f"{region}|{topic}")
        return prompts, ids

    async def run(self, *, reset_files: bool = False, **kwargs) -> pd.DataFrame:
        prompts, ids = self._build()
        csv_path = os.path.join(self.save_path, "regional_responses.csv")
        resp_df = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            use_web_search=True,
            search_context_size=self.cfg.search_context_size,
            reasoning_effort=self.cfg.reasoning_effort,
            save_path=csv_path,
            reset_files=reset_files,
            use_dummy=self.cfg.use_dummy,
            print_example_prompt=self.cfg.print_example_prompt,
            max_tokens=50000,
            timeout=450,
            **kwargs,
        )

        records = []
        for ident, resp in zip(resp_df.Identifier, resp_df.Response):
            region, topic = ident.split("|", 1)
            text = resp[0] if isinstance(resp, list) else resp
            records.append({"region": region, "topic": topic, "report": text})

        df_long = pd.DataFrame(records)
        df_wide = df_long.pivot(index="region", columns="topic", values="report").reset_index()
        df_wide.to_csv(os.path.join(self.save_path, "regional_reports.csv"), index=False)
        return df_wide
