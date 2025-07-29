# src/gabriel/tasks/ratings.py
# ════════════════════════════════════════════════════════════════════
# Robust passage-rating task with optional debug logging.
# ════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional
import os

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class RatingsConfig:
    attributes: Dict[str, str]
    save_dir: str = "ratings"
    file_name: str = "ratings.csv"
    model: str = "o4-mini"
    n_parallels: int = 100
    n_runs: int = 1
    use_dummy: bool = False
    timeout: float = 60.0
    rating_scale: str | None = None
    additional_instructions: str | None = None


# ────────────────────────────
# Main Ratings task
# ────────────────────────────
class Ratings:
    """Rate passages on specified attributes (0–100)."""


    # -----------------------------------------------------------------
    def __init__(self, cfg: RatingsConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Parse raw LLM output into {attribute: float}
    # -----------------------------------------------------------------
    async def _parse(self, raw: Any) -> Dict[str, Optional[float]]:
        obj = await safest_json(raw)
        out: Dict[str, Optional[float]] = {}
        if isinstance(obj, dict):
            for attr in self.cfg.attributes:
                try:
                    out[attr] = float(obj.get(attr)) if obj.get(attr) is not None else None
                except Exception:
                    out[attr] = None
            return out

        return {attr: None for attr in self.cfg.attributes}

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        text_column: str,
        *,
        debug: bool = False,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return ``df`` with one column per attribute rating."""

        df_proc = df.reset_index(drop=True).copy()
        texts = df_proc[text_column].astype(str).tolist()

        prompts: List[str] = []
        ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_text: Dict[str, str] = {}

        # Build prompts, deduplicating identical passages
        for row, passage in enumerate(texts):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            id_to_text[sha8] = passage
            prompts.append(
                self.template.render(
                    text=passage,
                    attributes=self.cfg.attributes,
                    scale=self.cfg.rating_scale,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            ids.append(sha8)

        csv_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        base_root, ext = os.path.splitext(csv_path)

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        async def _run_once(idx: int):
            path = csv_path if self.cfg.n_runs == 1 else f"{base_root}_run{idx}{ext}"
            return await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=path,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                json_mode=True,
                reset_files=reset_files,
                **kwargs,
            )

        if self.cfg.n_runs == 1:
            df_resps = [await _run_once(1)]
        else:
            df_resps = await asyncio.gather(*[_run_once(i) for i in range(1, self.cfg.n_runs + 1)])

        if debug:
            print("\n── raw LLM responses ──")
            for run_idx, df_resp in enumerate(df_resps, start=1):
                for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                    r = raw[0] if isinstance(raw, list) and raw else raw
                    print(f"[run {run_idx}] {ident} →\n{r}\n")
            print("────────────────────────\n")

        # parse each run and build disaggregated records
        full_records: List[Dict[str, Any]] = []
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {}
            for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                main = raw[0] if isinstance(raw, list) and raw else raw
                id_to_ratings[ident] = await self._parse(main)
            for ident in ids:
                parsed = id_to_ratings.get(ident, {attr: None for attr in self.cfg.attributes})
                rec = {"text": id_to_text[ident], "run": run_idx}
                rec.update({attr: parsed.get(attr) for attr in self.cfg.attributes})
                full_records.append(rec)

        full_df = pd.DataFrame(full_records).set_index(["text", "run"])
        disagg_path = f"{base_root}_full_disaggregated.csv"
        full_df.to_csv(disagg_path, index_label=["text", "run"])

        # aggregate across runs
        agg_df = full_df.groupby("text")[list(self.cfg.attributes)].mean()

        out_path = os.path.splitext(csv_path)[0] + "_final.csv"
        result = df_proc.merge(agg_df, left_on=text_column, right_index=True, how="left")
        result.to_csv(out_path, index=False)
        return result
