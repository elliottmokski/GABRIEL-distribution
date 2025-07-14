import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd
from utility_functions import Teleprompter, get_all_responses


class BasicClassifier:
    """
    Robust passage-classifier that turns LLM JSON (or JSON-ish text)
    into boolean columns for every label.
    """

    # ─────────────────────────────────────────── helpers
    _FENCE_RE = (
        Teleprompter._JSON_FENCE_RE
        if hasattr(Teleprompter, "_JSON_FENCE_RE")
        else re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
    )
    _safe_json = getattr(Teleprompter, "_safe_json", staticmethod(json.loads))

    # ─────────────────────────────────────────── ctor
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        labels: Dict[str, str],
        save_dir: str,
        additional_instructions: str = "",
        *,
        model: str = "o4-mini",
        n_parallels: int = 400,
        use_dummy: bool = False,
    ):
        self.df = df.copy()
        self.text_col, self.labels = text_col, labels
        self.save_dir, self.model = save_dir, model
        self.n_parallels, self.use_dummy = n_parallels, use_dummy
        self.tp = Teleprompter("prompts")
        self.additional_instructions = additional_instructions
        os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────── prompt builder
    def _build(self):
        prompts, ids = [], []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)

        for row, txt in self.df[self.text_col].astype(str).items():
            clean = " ".join(txt.split())
            sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]  # ← stable id
            id_to_rows[sha8].append(row)

            if len(id_to_rows[sha8]) > 1:  # duplicate passage
                continue

            prompts.append(
                self.tp.basic_classifier_prompt(
                    text=txt,
                    labels=self.labels,
                    additional_instructions=self.additional_instructions,
                )
            )
            ids.append(sha8)

        dup_ct = len(self.df) - len(prompts)
        if dup_ct:
            print(f"[BasicClassifier] Skipped {dup_ct} duplicate prompt(s).")

        return prompts, ids, id_to_rows

    # ─────────────────────────────────────────── unwrap helpers
    @classmethod
    def _unwrap(cls, x: Any) -> Any:
        """
        Collapse common wrappers so we end up with a parsable payload.
        Handles:
          • single-item lists        [ " {...} " ]
          • bytes                    b'{"k":true}'
          • ```json fenced blocks
          • stray trailing comma at the end of JSON object
        """
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        if isinstance(x, (bytes, bytearray)):
            x = x.decode()

        if isinstance(x, str):
            m = cls._FENCE_RE.search(x)
            if m:
                x = m.group(1).strip()

            # remove leading/trailing whitespace
            s = x.strip()

            # strip outer [] if it's a list of exactly one element
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                # but only if braces/parens/brackets are balanced
                if inner.count("{") == inner.count("}"):
                    s = inner
            # remove trailing comma inside {...}
            if s.startswith("{") and s.rstrip().endswith(",}"):
                s = re.sub(r",\s*}$", "}", s.rstrip())

            x = s
        return x

    # ─────────────────────────────────────────── regex fallback
    @staticmethod
    def _regex(raw: str, labels: Dict[str, str]) -> Dict[str, Optional[bool]]:
        out = {}
        for lab in labels:
            pat = re.compile(rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)', re.I | re.S)
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    # ─────────────────────────────────────────── parse single payload
    def _parse(self, resp: Any) -> Dict[str, Optional[bool]]:
        resp = self._unwrap(resp)

        # --- try strict JSON first ---------------------------------
        if isinstance(resp, (str, bytes, bytearray)):
            try:
                resp = self._safe_json(resp)
            except Exception:
                # sometimes the string is a Python dict repr; try ast.literal_eval
                try:
                    resp = ast.literal_eval(resp)
                except Exception:
                    resp = None

        if isinstance(resp, dict):
            norm = {
                k.strip().lower(): (
                    True
                    if str(v).strip().lower() in {"true", "yes", "1"}
                    else False
                    if str(v).strip().lower() in {"false", "no", "0"}
                    else None
                )
                for k, v in resp.items()
            }
            return {lab: norm.get(lab.lower(), None) for lab in self.labels}

        # --- fallback regex ----------------------------------------
        return self._regex(str(resp), self.labels)

    # ─────────────────────────────────────────── main
    async def run(self, *, reset_files: bool = False, **kwargs):
        prompts, ids, id_to_rows = self._build()
        csv_path = os.path.join(self.save_dir, "basic_classifier_responses.csv")

        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.n_parallels,
            save_path=csv_path,
            reset_files=reset_files,
            json_mode=True,
            model=self.model,
            use_dummy=self.use_dummy,
            print_example_prompt=True,
            **kwargs,
        )

        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")

        # broadcast parsed results to every original row index
        parsed_master = {idx: {lab: None for lab in self.labels} for idx in self.df.index}
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
                f"[BasicClassifier] WARNING: {orphans} response(s) "
                f"had no matching passage this run."
            )
        print(f"[BasicClassifier] Filled {filled}/{len(parsed_master)} rows.")

        parsed_df = pd.DataFrame.from_dict(parsed_master, orient="index")

        # coverage metrics
        total = len(parsed_df)
        print("\n=== Label coverage (non-null) ===")
        for lab in self.labels:
            n = parsed_df[lab].notna().sum()
            print(f"{lab:<55s}: {n / total:6.2%} ({n}/{total})")
        print("=================================\n")

        return self.df.join(parsed_df, how="left")
