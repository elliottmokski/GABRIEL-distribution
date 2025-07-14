from __future__ import annotations

import ast
import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd
from utility_functions import Teleprompter, get_all_responses

# Ensure EloRater and all users expect Teleprompter(prompt_path='prompts')


@dataclass
class EloConfig:
    # Accepts either a dict of {attribute: definition} or a list of attribute names
    attributes: Union[dict, List[str]]
    n_rounds: int = 15
    k_factor: float = 32.0
    n_parallels: int = 400
    model: str = "o4-mini"
    save_dir: str = os.path.expanduser("~/Documents/runs")
    run_name: str = f"elo_{datetime.now():%Y%m%d_%H%M%S}"
    use_dummy: bool = False
    timeout: float = 45.0
    power_matching: bool = False
    print_example_prompt: bool = True
    instructions: str = ""


class EloRater:
    """Pairwise ELO-style ranking of texts across multiple attributes."""

    def __init__(self, teleprompter: Teleprompter, cfg: EloConfig) -> None:
        self.tele = teleprompter
        self.cfg = cfg
        self.save_path = os.path.join(cfg.save_dir, cfg.run_name)
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _safe_json(txt: Any) -> dict:
        import ast

        try:
            if isinstance(txt, dict):
                return txt
            if isinstance(txt, list) and txt and isinstance(txt[0], dict):
                return txt[0]
            cleaned = str(txt).strip()
            # Remove outer quotes if present
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

    async def run(self, df: pd.DataFrame, text_col: str, id_col: str) -> pd.DataFrame:
        import pandas as pd

        final_path = os.path.join(self.save_path, "elo_final.csv")
        if os.path.exists(final_path):
            # print(f"[ELO] Loading ELO results from {final_path}")
            return pd.read_csv(final_path)
        # Ensure identifiers are treated consistently as strings
        texts = list(zip(df[id_col].astype(str), df[text_col]))
        # Always use attribute names as keys (dict keys if dict, else list)
        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)
        ratings = {id_: {a: 1000.0 for a in attr_keys} for id_, _ in texts}

        def expected(r_a, r_b):
            return 1 / (1 + 10 ** ((r_b - r_a) / 400))

        for rnd in range(self.cfg.n_rounds):
            if self.cfg.power_matching:
                texts.sort(key=lambda it: sum(ratings[it[0]].values()))
            else:
                random.shuffle(texts)
            pairs = [(texts[i], texts[i + 1]) for i in range(0, len(texts) - 1, 2)]
            attr_batches = [attr_keys[i : i + 8] for i in range(0, len(attr_keys), 8)]
            prompts, ids = [], []
            for batch_idx, batch in enumerate(attr_batches):
                for (id_a, t_a), (id_b, t_b) in pairs:
                    prompts.append(
                        self.tele.generic_elo_prompt(
                            text_circle=t_a,
                            text_square=t_b,
                            attributes={a: self.cfg.attributes[a] for a in batch},
                            instructions=self.cfg.instructions,
                        )
                    )
                    ids.append(f"{rnd}|{batch_idx}|{id_a}|{id_b}")
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=True,
                save_path=os.path.join(self.save_path, f"elo_round{rnd}.csv"),
                reset_files=False,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                print_example_prompt=self.cfg.print_example_prompt,
            )
            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                _, batch_idx, a_id, b_id = ident.split("|")
                batch_idx = int(batch_idx)
                safe = self._safe_json(resp)
                if not safe:
                    continue
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                safe = {}
                if isinstance(resp, list) and len(resp) > 0:
                    safe = self._safe_json(resp[0])
                elif isinstance(resp, str):
                    try:
                        resp_list = ast.literal_eval(resp)
                        if isinstance(resp_list, list) and len(resp_list) > 0:
                            safe = self._safe_json(resp_list[0])
                    except Exception:
                        pass
                for attr, winner in safe.items():
                    attr_key = str(attr).strip().lower()
                    if attr_key not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key]
                    if winner == "circle":
                        score_a, score_b = 1, 0
                    elif winner == "square":
                        score_a, score_b = 0, 1
                    else:
                        continue
                    exp_a = expected(ratings[a_id][real_attr], ratings[b_id][real_attr])
                    exp_b = 1 - exp_a
                    ratings[a_id][real_attr] += self.cfg.k_factor * (score_a - exp_a)
                    ratings[b_id][real_attr] += self.cfg.k_factor * (score_b - exp_b)

        rows = []
        for ident, text in texts:
            row = {"identifier": ident, "text": text}
            for attr in self.cfg.attributes:
                row[attr] = ratings[ident][attr]
            rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(self.save_path, "elo_final.csv"), index=False)
        return df_out
