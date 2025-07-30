from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from utility_functions import Teleprompter, get_all_responses


class Regional:
    """Run regional topic analysis prompts in parallel."""

    def __init__(
        self,
        df: pd.DataFrame,
        region_col: str,
        topics: List[str],
        *,
        save_dir: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        model: str = "o4-mini",
        n_parallels: int = 400,
        use_dummy: bool = False,
        additional_instructions: str = "",
        reasoning_effort: str = "medium",
        search_context_size: str = "medium",
        print_example_prompt: bool = True,
    ) -> None:
        self.df = df.copy()
        self.region_col = region_col
        self.topics = topics
        self.model = model
        self.n_parallels = n_parallels
        self.use_dummy = use_dummy
        self.additional_instructions = additional_instructions
        self.reasoning_effort = reasoning_effort
        self.search_context_size = search_context_size
        self.print_example_prompt = print_example_prompt

        run_name = run_name or f"regional_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(save_dir, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "regional")
        self.tele = Teleprompter(prompt_dir)

    def _build(self) -> tuple[List[str], List[str]]:
        prompts, ids = [], []
        unique_regions = self.df[self.region_col].astype(str).dropna().unique()
        for region in unique_regions:
            for topic in self.topics:
                prompts.append(
                    self.tele.regional_analysis_prompt(
                        region=region,
                        topic=topic,
                        additional_instructions=self.additional_instructions,
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
            n_parallels=self.n_parallels,
            model=self.model,
            use_web_search=True,
            search_context_size=self.search_context_size,
            reasoning_effort=self.reasoning_effort,
            save_path=csv_path,
            reset_files=reset_files,
            use_dummy=self.use_dummy,
            print_example_prompt=self.print_example_prompt,
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
