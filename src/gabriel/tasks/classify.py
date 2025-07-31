from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class ClassifyConfig:
    """Configuration for :class:`Classify`."""

    labels: Dict[str, str]  # {"label_name": "description", ...}
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "o4-mini"
    n_parallels: int = 400
    n_runs: int = 1
    additional_instructions: str = ""
    additional_guidelines: str = ""
    use_dummy: bool = False
    timeout: float = 60.0


# ────────────────────────────
# Main Basic classifier task
# ────────────────────────────
class Classify:
    """Robust passage classifier using an LLM.

    * Accepts a list of *texts* (not a DataFrame) just like :class:`Rate`.
    * Persists/reads cached responses via the **save_path** attribute (same pattern as
      :class:`Rate`).
    """

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    # -----------------------------------------------------------------
    def __init__(self, cfg: ClassifyConfig, template: Optional[PromptTemplate] = None) -> None:  # noqa: D401,E501
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package(
            "basic_classifier_prompt.jinja2"
        )

    # -----------------------------------------------------------------
    # Build prompts (deduplicating identical passages)
    # -----------------------------------------------------------------
    def _build(self, texts: List[str]):
        prompts: List[str] = []
        ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_text: Dict[str, str] = {}

        for row, passage in enumerate(texts):
            clean = " ".join(str(passage).split())
            sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue  # duplicate passage – no need to prompt again
            id_to_text[sha8] = passage

            prompts.append(
                self.template.render(
                    text=passage,
                    labels=self.cfg.labels,
                    additional_instructions=self.cfg.additional_instructions,
                    additional_guidelines=self.cfg.additional_guidelines,
                )
            )
            ids.append(sha8)

        dup_ct = len(texts) - len(prompts)
        if dup_ct:
            print(f"[Classify] Skipped {dup_ct} duplicate prompt(s).")
        return prompts, ids, id_to_rows, id_to_text

    # -----------------------------------------------------------------
    # Helpers for parsing raw model output
    # -----------------------------------------------------------------
    @staticmethod
    def _regex(raw: str, labels: Dict[str, str]) -> Dict[str, Optional[bool]]:
        out: Dict[str, Optional[bool]] = {}
        for lab in labels:
            pat = re.compile(rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)', re.I | re.S)
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    async def _parse(self, resp: Any) -> Dict[str, Optional[bool]]:
        # unwrap common response containers (list-of-one, bytes, fenced blocks)
        if isinstance(resp, list) and len(resp) == 1:
            resp = resp[0]
        if isinstance(resp, (bytes, bytearray)):
            resp = resp.decode()
        if isinstance(resp, str):
            m = self._FENCE_RE.search(resp)
            if m:
                resp = m.group(1).strip()

        data = await safest_json(resp)
        if isinstance(data, dict):
            norm = {
                k.strip().lower(): (
                    True
                    if str(v).strip().lower() in {"true", "yes", "1"}
                    else False
                    if str(v).strip().lower() in {"false", "no", "0"}
                    else None
                )
                for k, v in data.items()
            }
            return {lab: norm.get(lab.lower(), None) for lab in self.cfg.labels}

        # fallback to regex extraction
        return self._regex(str(resp), self.cfg.labels)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        text_column: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Classify texts in ``df[text_column]`` and return ``df`` with label columns."""

        df_proc = df.reset_index(drop=True).copy()
        texts = df_proc[text_column].astype(str).tolist()

        prompts, ids, id_to_rows, id_to_text = self._build(texts)

        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        if self.cfg.n_runs == 1:
            df_resp_all = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                save_path=csv_path,
                reset_files=reset_files,
                json_mode=True,
                model=self.cfg.model,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")
            df_resps = [df_resp_all]
        else:
            prompts_all: List[str] = []
            ids_all: List[str] = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                prompts_all.extend(prompts)
                ids_all.extend([f"{ident}_run{run_idx}" for ident in ids])

            df_resp_all = await get_all_responses(
                prompts=prompts_all,
                identifiers=ids_all,
                n_parallels=self.cfg.n_parallels,
                save_path=csv_path,
                reset_files=reset_files,
                json_mode=True,
                model=self.cfg.model,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
                df_resps.append(sub)

        # parse each run and construct disaggregated records
        full_records: List[Dict[str, Any]] = []
        total_orphans = 0
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_labels: Dict[str, Dict[str, Optional[bool]]] = {}
            orphans = 0
            for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                if ident not in id_to_rows:
                    orphans += 1
                    continue
                parsed = await self._parse(raw)
                id_to_labels[ident] = parsed
            total_orphans += orphans
            for ident in ids:
                parsed = id_to_labels.get(ident, {lab: None for lab in self.cfg.labels})
                rec = {"text": id_to_text[ident], "run": run_idx}
                rec.update({lab: parsed.get(lab) for lab in self.cfg.labels})
                full_records.append(rec)

        if total_orphans:
            print(
                f"[Classify] WARNING: {total_orphans} response(s) had no matching passage this run."
            )

        full_df = pd.DataFrame(full_records).set_index(["text", "run"])
        disagg_path = os.path.join(self.cfg.save_dir, f"{base_name}_full_disaggregated.csv")
        full_df.to_csv(disagg_path, index_label=["text", "run"])

        # aggregate across runs using mode
        def _mode(s: pd.Series) -> Optional[bool]:
            ser = s.dropna()
            if ser.empty:
                return None
            return ser.mode().iloc[0]

        agg_df = pd.DataFrame({lab: full_df[lab].groupby("text").apply(_mode) for lab in self.cfg.labels})

        filled = agg_df.dropna(how="all").shape[0]
        print(f"[Classify] Filled {filled}/{len(agg_df)} unique texts.")

        total = len(agg_df)
        print("\n=== Label coverage (non-null) ===")
        for lab in self.cfg.labels:
            n = agg_df[lab].notna().sum()
            print(f"{lab:<55s}: {n / total:6.2%} ({n}/{total})")
        print("=================================\n")

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result = df_proc.merge(agg_df, left_on=text_column, right_index=True, how="left")
        result.to_csv(out_path, index=False)

        # keep raw response files for reference

        return result

