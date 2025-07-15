from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, List, Type

import pandas as pd

from ..core.prompt_template import PromptTemplate
from .openai_utils import get_all_responses


@dataclass
class PromptParaphraserConfig:
    """Configuration for :class:`PromptParaphraser`."""

    n_variants: int = 3
    model: str = "o4-mini"
    n_parallels: int = 25
    save_dir: str = "paraphraser"
    use_dummy: bool = False
    timeout: float = 60.0


class PromptParaphraser:
    """Generate paraphrased versions of a task prompt and rerun the task."""

    def __init__(self, cfg: PromptParaphraserConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("prompt_paraphraser_prompt.jinja2")
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    async def _paraphrase(self, prompt_text: str) -> List[str]:
        prompts = [self.template.render(baseline_prompt=prompt_text) for _ in range(self.cfg.n_variants)]
        ids = [f"variant_{i}" for i in range(1, self.cfg.n_variants + 1)]
        csv_path = os.path.join(self.cfg.save_dir, "paraphrases.csv")
        df = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=csv_path,
            reset_files=True,
            use_dummy=self.cfg.use_dummy,
            timeout=self.cfg.timeout,
        )
        return [resp[0] if isinstance(resp, list) else resp for resp in df.Response]

    async def run(
        self,
        task_cls: Type[Any],
        task_cfg: Any,
        *run_args: Any,
        template: PromptTemplate | None = None,
        **run_kwargs: Any,
    ) -> pd.DataFrame:
        base_template = template or getattr(task_cls(task_cfg), "template")
        variants = await self._paraphrase(base_template.text)

        results = []

        base_task = task_cls(task_cfg, template=base_template)
        df_base = await base_task.run(*run_args, **run_kwargs)
        df_base = df_base.copy()
        df_base["prompt_variant"] = "baseline"
        results.append(df_base)

        for idx, text in enumerate(variants, start=1):
            variant_template = PromptTemplate(text)
            cfg_variant = copy.deepcopy(task_cfg)
            if hasattr(cfg_variant, "save_path"):
                base, ext = os.path.splitext(cfg_variant.save_path)
                cfg_variant.save_path = f"{base}_p{idx}{ext}"
            if hasattr(cfg_variant, "save_dir"):
                cfg_variant.save_dir = os.path.join(cfg_variant.save_dir, f"variant_{idx}")
                os.makedirs(cfg_variant.save_dir, exist_ok=True)
            task = task_cls(cfg_variant, template=variant_template)
            df = await task.run(*run_args, **run_kwargs)
            df = df.copy()
            df["prompt_variant"] = f"variant_{idx}"
            results.append(df)

        return pd.concat(results, ignore_index=True)
