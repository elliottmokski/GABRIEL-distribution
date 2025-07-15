from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class BasicClassifierConfig:
    labels: Dict[str, str]
    save_dir: str = "classifier"
    model: str = "o4-mini"
    n_parallels: int = 400
    additional_instructions: str = ""
    use_dummy: bool = False
    timeout: float = 60.0


class BasicClassifier:
    """Robust passage classifier using an LLM."""

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    @staticmethod
    def _safe_json(txt: Any) -> dict:
        try:
            if isinstance(txt, dict):
                return txt
            if isinstance(txt, list):
                if txt and isinstance(txt[0], dict):
                    return txt[0]
                if txt and isinstance(txt[0], str):
                    txt = txt[0]
            cleaned = str(txt).strip()
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                cleaned.startswith("'") and cleaned.endswith("'")
            ):
                cleaned = cleaned[1:-1]
            try:
                return json.loads(cleaned)
            except Exception:
                try:
                    return ast.literal_eval(cleaned)
                except Exception:
                    return {}
        except Exception:
            return {}

    def __init__(self, cfg: BasicClassifierConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package(
            "basic_classifier_prompt.jinja2"
        )
        os.makedirs(cfg.save_dir, exist_ok=True)

    def _build(self, df: pd.DataFrame, text_col: str):
        prompts: List[str] = []
        ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        for row, txt in df[text_col].astype(str).items():
            clean = " ".join(txt.split())
            sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            prompts.append(
                self.template.render(
                    text=txt,
                    labels=self.cfg.labels,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            ids.append(sha8)
        dup_ct = len(df) - len(prompts)
        if dup_ct:
            print(f"[BasicClassifier] Skipped {dup_ct} duplicate prompt(s).")
        return prompts, ids, id_to_rows

    @staticmethod
    def _regex(raw: str, labels: Dict[str, str]) -> Dict[str, Optional[bool]]:
        out: Dict[str, Optional[bool]] = {}
        for lab in labels:
            pat = re.compile(rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)', re.I | re.S)
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    def _parse(self, resp: Any) -> Dict[str, Optional[bool]]:
        if isinstance(resp, list) and len(resp) == 1:
            resp = resp[0]
        if isinstance(resp, (bytes, bytearray)):
            resp = resp.decode()
        if isinstance(resp, str):
            m = self._FENCE_RE.search(resp)
            if m:
                resp = m.group(1).strip()
        data = self._safe_json(resp)
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
        return self._regex(str(resp), self.cfg.labels)

    async def run(
        self, df: pd.DataFrame, text_column: str, *, reset_files: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        prompts, ids, id_to_rows = self._build(df, text_column)
        csv_path = os.path.join(self.cfg.save_dir, "basic_classifier_responses.csv")

        df_resp = await get_all_responses(
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
        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")

        parsed_master = {idx: {lab: None for lab in self.cfg.labels} for idx in df.index}
        orphans = 0
        for ident, raw in zip(df_resp.Identifier, df_resp.Response):
            if ident not in id_to_rows:
                orphans += 1
                continue
            parsed = self._parse(raw)
            for row in id_to_rows[ident]:
                parsed_master[row] = parsed

        filled = sum(any(v is not None for v in d.values()) for d in parsed_master.values())
        if orphans:
            print(
                f"[BasicClassifier] WARNING: {orphans} response(s) had no matching passage this run."
            )
        print(f"[BasicClassifier] Filled {filled}/{len(parsed_master)} rows.")

        parsed_df = pd.DataFrame.from_dict(parsed_master, orient="index")

        total = len(parsed_df)
        print("\n=== Label coverage (non-null) ===")
        for lab in self.cfg.labels:
            n = parsed_df[lab].notna().sum()
            print(f"{lab:<55s}: {n / total:6.2%} ({n}/{total})")
        print("=================================\n")

        return df.join(parsed_df, how="left")

