from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


@dataclass
class DeidentifyConfig:
    """Configuration for :class:`Deidentifier`."""

    model: str = "o4-mini"
    n_parallels: int = 50
    save_dir: str = "deidentify"
    file_name: str = "deidentified.csv"
    use_dummy: bool = False
    timeout: float = 60.0
    max_words_per_call: int = 7500
    guidelines: str = ""
    additional_guidelines: str = ""


class Deidentifier:
    """Iterative de-identification of sensitive entities in text."""

    def __init__(self, cfg: DeidentifyConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("faceless_prompt.jinja2")
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    @staticmethod
    def _chunk_by_words(text: str, max_words: int) -> List[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


    async def run(
        self,
        df: pd.DataFrame,
        text_column: str,
        grouping_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Deidentify all texts in ``df[text_column]``.

        A ``grouping_column`` can be provided to ensure that the mapping is built
        across multiple rows belonging to the same group.
        """
        df_proc = df.reset_index(drop=True).copy()
        if grouping_column is None:
            df_proc["group_id"] = df_proc.index.astype(str)
        else:
            df_proc["group_id"] = df_proc[grouping_column].astype(str)

        group_ids = df_proc["group_id"].unique().tolist()
        group_segments: Dict[str, List[str]] = {}

        csv_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        base_root, ext = os.path.splitext(csv_path)
        for gid in group_ids:
            segs: List[str] = []
            texts = (
                df_proc.loc[df_proc["group_id"] == gid, text_column]
                .fillna("")
                .astype(str)
                .tolist()
            )
            for text in texts:
                segs.extend(self._chunk_by_words(text, self.cfg.max_words_per_call))
            group_segments[gid] = segs

        group_to_map: Dict[str, dict] = {gid: {} for gid in group_ids}
        max_rounds = max(len(s) for s in group_segments.values()) if group_segments else 0

        for rnd in range(max_rounds):
            prompts: List[str] = []
            identifiers: List[str] = []
            active_gids: List[str] = []
            for gid in group_ids:
                segs = group_segments[gid]
                if rnd < len(segs):
                    prompts.append(
                        self.template.render(
                            text=segs[rnd],
                            current_map=json.dumps(group_to_map[gid], ensure_ascii=False),
                            guidelines=self.cfg.guidelines,
                            additional_guidelines=self.cfg.additional_guidelines,
                        )
                    )
                    identifiers.append(f"{gid}_seg_{rnd}")
                    active_gids.append(gid)
            if not prompts:
                continue
            batch_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=f"{base_root}_round{rnd}{ext}",
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                json_mode=True,
            )
            for ident, resp in zip(batch_df["Identifier"], batch_df["Response"]):
                gid = ident.split("_seg_")[0]
                main = resp[0] if isinstance(resp, list) and resp else ""
                parsed = await safest_json(main)
                if parsed:
                    group_to_map[gid] = parsed

        mappings_col: List[dict] = []
        deidentified_texts: List[str] = []
        for _, row in df_proc.iterrows():
            gid = row["group_id"]
            mapping = group_to_map.get(gid, {})
            mappings_col.append(mapping)
            text = str(row[text_column])
            deid_text = text
            pairs: List[tuple[str, str]] = []
            for entry in mapping.values():
                if isinstance(entry, dict):
                    casted = entry.get("casted form", "")
                    for real in entry.get("real forms", []):
                        pairs.append((real, casted))
            pairs.sort(key=lambda x: len(x[0]), reverse=True)
            for real, casted in pairs:
                pattern = re.compile(rf"\b{re.escape(real)}\b", flags=re.IGNORECASE)
                deid_text = pattern.sub(casted, deid_text)
            deidentified_texts.append(deid_text)

        df_proc["mapping"] = mappings_col
        df_proc["deidentified_text"] = deidentified_texts
        df_proc.to_csv(csv_path, index=False)
        return df_proc
