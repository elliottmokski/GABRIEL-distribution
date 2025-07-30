import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm
from utility_functions import Teleprompter, get_all_responses


class Faceless:
    """Pipeline for deidentifying personal or sensitive entities across grouped texts."""

    def __init__(self, teleprompter: Teleprompter) -> None:
        self.teleprompter = teleprompter

    def parse_json(self, response_text: Any) -> Optional[dict]:
        if not isinstance(response_text, str):
            return None
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None

    def chunk_by_words(self, text: str, max_words: int) -> List[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        grouping_column: Optional[str] = None,
        max_words_per_call: int = 75000,
        deid_guidelines: str = "",
        n_parallels: int = 400,
        model: str = "o3-mini",
        save_root: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        reset_files: bool = False,
        debug_print: bool = False,
        use_dummy: bool = False,
    ) -> pd.DataFrame:
        """Process all groups of text, iteratively building per-group mapping dictionaries of sensitive entities and applying substitutions. Returns updated dataframe."""
        if run_name is None:
            run_name = f"faceless_run_{int(random.random() * 1e6)}"
        save_path = os.path.join(save_root, run_name)
        os.makedirs(save_path, exist_ok=True)
        df_proc = df.reset_index(drop=True)
        # assign group ids
        if grouping_column is None:
            df_proc["group_id"] = df_proc.index.astype(str)
        else:
            df_proc["group_id"] = df_proc[grouping_column].astype(str)
        group_ids = df_proc["group_id"].unique().tolist()
        # Build list of segments to process per group
        group_segments: Dict[str, List[str]] = {}
        for gid in group_ids:
            segments: List[str] = []
            texts = (
                df_proc.loc[df_proc["group_id"] == gid, column_name].fillna("").astype(str).tolist()
            )
            for text in texts:
                segments.extend(self.chunk_by_words(text, max_words_per_call))
            group_segments[gid] = segments
        max_rounds = max(len(segs) for segs in group_segments.values()) if group_segments else 0
        # initial empty mapping per group
        group_to_map: Dict[str, dict] = {gid: {} for gid in group_ids}
        expected_schema: dict = {
            "__dynamic__": {
                "type": {
                    "real forms": [str],
                    "casted form": str,
                }
            }
        }
        for round_idx in range(max_rounds):
            prompts: List[str] = []
            identifiers: List[str] = []
            active_gids: List[str] = []
            for gid in group_ids:
                segs = group_segments[gid]
                if round_idx < len(segs):
                    prompt = self.teleprompter.faceless_prompt(
                        text=segs[round_idx],
                        current_map=group_to_map[gid],
                        guidelines=deid_guidelines,
                    )
                    prompts.append(prompt)
                    identifiers.append(f"{gid}_seg_{round_idx}")
                    active_gids.append(gid)
            if not prompts:
                continue
            if debug_print and prompts:
                print(f"\n[DEBUG] First prompt for round {round_idx}:\n", prompts[0][:500], "\n")
            batch_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                n_parallels=n_parallels,
                save_path=os.path.join(save_path, f"round_{round_idx}.csv"),
                reset_files=reset_files,
                json_mode=True,
                expected_schema=None,
                model=model,
                use_dummy=use_dummy,
            )
            # update group mappings
            for ident, resp in zip(batch_df["Identifier"], batch_df["Response"]):
                gid = ident.split("_seg_")[0]
                main = resp[0] if isinstance(resp, list) and resp else ""
                parsed = self.parse_json(main) or {}
                group_to_map[gid] = parsed
        # apply mappings to original texts
        mappings_column: List[dict] = []
        deidentified_texts: List[str] = []
        for _, row in df_proc.iterrows():
            gid = row["group_id"]
            mapping = group_to_map.get(gid, {})
            mappings_column.append(mapping)
            text = str(row[column_name])
            deid_text = text
            # Build list of (real form, casted) pairs
            pairs: List[tuple[str, str]] = []
            for entry in mapping.values():
                if isinstance(entry, dict):
                    casted = entry.get("casted form", "")
                    for real_form in entry.get("real forms", []):
                        pairs.append((real_form, casted))
            # sort by length desc
            pairs.sort(key=lambda x: len(x[0]), reverse=True)
            for real_form, casted in pairs:
                pattern = re.compile(rf"\b{re.escape(real_form)}\b", flags=re.IGNORECASE)
                deid_text = pattern.sub(casted, deid_text)
            deidentified_texts.append(deid_text)
        df_proc["mapping"] = mappings_column
        df_proc["deidentified_text"] = deidentified_texts
        df_proc.to_csv(os.path.join(save_path, "deidentified_output.csv"), index=False)
        return df_proc
