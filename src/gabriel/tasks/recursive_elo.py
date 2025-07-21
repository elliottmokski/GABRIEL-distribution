# recursive_elo_rater.py (same package/folder as elo_rater.py)

from __future__ import annotations

import os
import math
import copy
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import pandas as pd

from .elo_rater import EloRater, EloConfig  # adjust import path as needed


# ---------------------------------------------------------------------
# Config for RecursiveEloRater
# ---------------------------------------------------------------------
@dataclass
class RecursiveEloConfig:
    """
    Configuration for RecursiveEloRater.

    Parameters
    ----------
    base_cfg : EloConfig
        The baseline EloConfig used for each recursion stage; we will clone/modify
        (e.g., run_name, n_rounds) per stage. Do *not* reuse the same run_name across
        multiple RecursiveEloRater instances unless you want to overwrite.
    cut_attr : str | None
        Attribute to use for ranking & filtering each recursion step. If None, defaults
        to the first attribute in base_cfg.attributes.
    cut_side : str
        'top' (keep highest) or 'bottom' (keep lowest). Default: 'top'.
    fraction : float
        Fraction of items to keep each stage (applied after current ratings). Default: 1/3.
    min_remaining : int
        Minimum number of items to allow in the final stage. If a cut would go below
        this, we set remaining to exactly this (min) and then do one final stage (with
        rounds multiplied) and stop. Default: 30.
    final_round_multiplier : int
        How much to multiply base_cfg.n_rounds for the *final* recursion stage to sharpen
        the last group’s ratings. Default: 3.
    rewrite_func : Optional[Callable[[str, str, int], str]]
        Optional text rewriter called on each surviving passage between stages.
        Signature: (text, identifier, stage_idx) -> new_text
        If None, no rewriting.
    rewrite_text_col : str
        Column name to write modified text into for next stage. (We rewrite in-place by default.)
    """
    base_cfg: EloConfig
    cut_attr: Optional[str] = None
    cut_side: str = "top"
    fraction: float = 1.0 / 3.0
    min_remaining: int = 30
    final_round_multiplier: int = 3
    rewrite_func: Optional[Callable[[str, str, int], str]] = None
    rewrite_text_col: str = "text"  # by default we will modify the text_col in place

    # internal / housekeeping
    # whether to concatenate stage outputs in final result
    keep_stage_columns: bool = True
    # optional: keep raw stage scores separately
    add_stage_suffix: bool = True


# ---------------------------------------------------------------------
# RecursiveEloRater
# ---------------------------------------------------------------------
class RecursiveEloRater:
    """
    Orchestrates recursive Elo/BT rating by repeatedly calling EloRater on progressively
    filtered subsets of items, carrying forward cumulative scores, and (optionally) rewriting
    texts between stages.

    Typical workflow
    ----------------
    rer = RecursiveEloRater(teleprompter, rec_cfg)
    final_df = await rer.run(df, text_col="text", id_col="identifier")

    Returns
    -------
    final_df : pd.DataFrame
        Contains:
          - identifier
          - final text (post rewrites if any)
          - cumulative_<attr> (cumulative scores across all stages for each attr)
          - last-stage (raw) <attr>, <attr>_z, <attr>_se (same as EloRater output on final stage)
          - exit_stage : stage index at which this item was removed (or final_stage if survived)
          - stage_<k>_<attr> columns (if keep_stage_columns=True)
    """

    def __init__(self, teleprompter, cfg: RecursiveEloConfig) -> None:
        self.tele = teleprompter
        self.cfg = cfg

        # Validate cut_side
        if self.cfg.cut_side not in ("top", "bottom"):
            raise ValueError("cut_side must be 'top' or 'bottom'")

        # Determine attribute order from base_cfg
        if isinstance(self.cfg.base_cfg.attributes, dict):
            self._attr_list = list(self.cfg.base_cfg.attributes.keys())
        else:
            self._attr_list = list(self.cfg.base_cfg.attributes)

        if not self._attr_list:
            raise ValueError("No attributes found in base_cfg.attributes")

        if self.cfg.cut_attr is None:
            self.cut_attr = self._attr_list[0]
        else:
            if self.cfg.cut_attr not in self._attr_list:
                raise ValueError(f"cut_attr '{self.cfg.cut_attr}' not in attributes")
            self.cut_attr = self.cfg.cut_attr

        # Keep cumulative scores across stages
        # {attr: {identifier: cumulative_score}}
        self._cumulative_scores: Dict[str, Dict[str, float]] = {}

        # Stage results list
        self._stage_dfs: List[pd.DataFrame] = []

    # ------------------------------ Helpers ------------------------------

    def _clone_cfg_for_stage(self, stage_idx: int, n_rounds: int) -> EloConfig:
        """Clone base EloConfig, adjusting run_name + n_rounds so each stage has its own folder."""
        base = self.cfg.base_cfg
        # new run_name
        stage_run_name = f"{base.run_name}_stage{stage_idx}"
        new_cfg = copy.deepcopy(base)
        new_cfg.run_name = stage_run_name
        new_cfg.n_rounds = n_rounds
        # ensure save_dir exists
        os.makedirs(os.path.join(new_cfg.save_dir, new_cfg.run_name), exist_ok=True)
        return new_cfg

    def _init_cumulative_scores(self, ids: Sequence[str]) -> None:
        if self._cumulative_scores:
            return
        for attr in self._attr_list:
            self._cumulative_scores[attr] = {i: 0.0 for i in ids}

    def _update_cumulative(self, stage_df: pd.DataFrame) -> None:
        """Add this stage's (mean-centered) scores to cumulative totals."""
        for attr in self._attr_list:
            if attr not in stage_df.columns:
                continue
            for i, val in zip(stage_df["identifier"], stage_df[attr]):
                self._cumulative_scores[attr][i] += float(val)

    def _get_rank_series(self, ids: Sequence[str]) -> pd.Series:
        """
        Return a ranking series (index=identifier, value=cumulative score for cut_attr)
        used to decide top/bottom slice.
        """
        data = {i: self._cumulative_scores[self.cut_attr][i] for i in ids}
        s = pd.Series(data, name="cumulative")
        return s.sort_values(ascending=(self.cfg.cut_side == "bottom"))  # sort ascending if keeping bottom

    def _select_next_ids(self, current_ids: Sequence[str]) -> List[str]:
        """Select fraction of current_ids according to cumulative scores & cut parameters."""
        n = len(current_ids)
        if n <= self.cfg.min_remaining:
            return list(current_ids)

        keep_n = max(int(math.ceil(n * self.cfg.fraction)), self.cfg.min_remaining)
        ranked = self._get_rank_series(current_ids)
        keep_ids = ranked.head(keep_n).index.tolist()
        return keep_ids

    def _maybe_rewrite_texts(
        self,
        df: pd.DataFrame,
        ids_to_keep: Sequence[str],
        stage_idx: int,
        text_col: str,
    ) -> pd.DataFrame:
        """Apply rewrite_func to surviving texts if provided."""
        if self.cfg.rewrite_func is None:
            return df
        mask = df["identifier"].isin(ids_to_keep)
        rewritten = []
        for idx, row in df[mask].iterrows():
            new_text = self.cfg.rewrite_func(row[text_col], row["identifier"], stage_idx)
            rewritten.append(new_text)
        df.loc[mask, self.cfg.rewrite_text_col] = rewritten
        return df

    # ------------------------------ Public ------------------------------

    async def run(
        self, df: pd.DataFrame, text_col: str, id_col: str, *, reset_files: bool = False
    ) -> pd.DataFrame:
        """
        Execute recursive Elo rating.

        Parameters
        ----------
        df : DataFrame
            Must include columns `text_col` and `id_col`.
        text_col : str
        id_col : str

        Returns
        -------
        final_df : DataFrame
        """

        # Make a working copy; standardize identifier -> str
        work_df = df.copy()
        work_df[id_col] = work_df[id_col].astype(str)
        work_df = work_df.rename(columns={id_col: "identifier"})
        if text_col != "text":
            work_df = work_df.rename(columns={text_col: "text"})

        # Initialize cumulative score storage
        all_ids = work_df["identifier"].tolist()
        self._init_cumulative_scores(all_ids)

        # Track exit stage for each id
        exit_stage = {i: None for i in all_ids}

        current_ids = list(all_ids)
        stage_idx = 0
        final_stage_idx = None

        while True:
            stage_idx += 1

            # Determine if this is final stage: cutting after this stage would drop below min_remaining
            # We'll *pre-check* by seeing how many would remain if we cut; if <= min_remaining, we plan this as final stage
            # (If current <= min_remaining, also final)
            is_final_stage = False
            if len(current_ids) <= self.cfg.min_remaining:
                is_final_stage = True
            else:
                # Simulate next cut
                n = len(current_ids)
                next_keep_n = max(int(math.ceil(n * self.cfg.fraction)), self.cfg.min_remaining)
                if next_keep_n <= self.cfg.min_remaining:
                    # The *next* stage would be <= min_remaining, so run final (with multiplier) now,
                    # then stop.
                    is_final_stage = True

            # Adjust n_rounds
            if is_final_stage:
                n_rounds = self.cfg.base_cfg.n_rounds * self.cfg.final_round_multiplier
            else:
                n_rounds = self.cfg.base_cfg.n_rounds

            stage_cfg = self._clone_cfg_for_stage(stage_idx, n_rounds=n_rounds)

            # Subset DF for this stage
            stage_df_in = work_df[work_df["identifier"].isin(current_ids)].copy()

            # Run EloRater
            elo = EloRater(self.tele, stage_cfg)
            stage_df_out = await elo.run(
                stage_df_in,
                text_col="text",
                id_col="identifier",
                reset_files=reset_files,
            )

            # Record stage columns if desired
            if self.cfg.keep_stage_columns:
                stage_cols = [c for c in stage_df_out.columns if c not in ("text")]
                # rename columns with stage suffix to avoid collision
                if self.cfg.add_stage_suffix:
                    renamed = {
                        c: f"stage{stage_idx}_{c}"
                        for c in stage_cols
                        if c != "identifier"
                    }
                    stage_df_stage = stage_df_out.rename(columns=renamed)
                else:
                    stage_df_stage = stage_df_out.copy()
                self._stage_dfs.append(stage_df_stage)

            # Update cumulative scores
            self._update_cumulative(stage_df_out)

            # Decide if done
            if is_final_stage:
                final_stage_idx = stage_idx
                # mark exit_stage for survivors
                for i in current_ids:
                    exit_stage[i] = stage_idx
                break

            # Otherwise, decide next ids
            next_ids = self._select_next_ids(current_ids)

            # mark exit for those removed
            removed = set(current_ids) - set(next_ids)
            for i in removed:
                exit_stage[i] = stage_idx

            # Optional rewriting
            work_df = self._maybe_rewrite_texts(
                df=work_df,
                ids_to_keep=next_ids,
                stage_idx=stage_idx,
                text_col="text",
            )

            # Continue
            current_ids = next_ids

        # -------------------------------------------------------------
        # Build final output
        # -------------------------------------------------------------
        # Last stage raw results (stage_df_out) already has z, se, etc.
        # We'll merge:
        #  1) cumulative scores
        #  2) last-stage raw columns for items that survived final stage
        #  3) exit_stage
        #  4) optional per-stage columns

        # cumulative
        cum_rows = []
        for i in all_ids:
            row = {"identifier": i}
            for attr in self._attr_list:
                row[f"cumulative_{attr}"] = self._cumulative_scores[attr][i]
            cum_rows.append(row)
        cum_df = pd.DataFrame(cum_rows)

        # exit stage
        exit_df = pd.DataFrame({"identifier": list(exit_stage.keys()),
                                "exit_stage": list(exit_stage.values())})

        # final survivors raw (stage_df_out)
        final_raw = stage_df_out.copy()
        final_raw_cols = [c for c in final_raw.columns if c != "text"]  # we keep text separately
        # We'll suffix them with 'final_' to avoid ambiguity
        final_raw_ren = {
            c: (c if c == "identifier" else f"final_{c}")
            for c in final_raw_cols
        }
        final_raw = final_raw.rename(columns=final_raw_ren)

        # main text we want is current (post rewrite) text in work_df for the *latest* version of each item
        latest_text_df = work_df[["identifier", "text"]].copy()

        # merge everything
        out = cum_df.merge(exit_df, on="identifier", how="left") \
                    .merge(latest_text_df, on="identifier", how="left") \
                    .merge(final_raw, on="identifier", how="left")

        # add per-stage columns if desired
        if self.cfg.keep_stage_columns and self._stage_dfs:
            for sdf in self._stage_dfs:
                out = out.merge(sdf, on="identifier", how="left")

        # For items removed before final stage, final_* columns will be NaN. That's fine;
        # cumulative holds their final score.

        # Reorder columns (identifier, text, exit_stage, cumulative_*, final_*, rest)
        prefixed_cum = [c for c in out.columns if c.startswith("cumulative_")]
        prefixed_final = [c for c in out.columns if c.startswith("final_")]
        cols = ["identifier", "text", "exit_stage"] + prefixed_cum + prefixed_final
        # add any leftover columns after
        remaining = [c for c in out.columns if c not in cols]
        out = out[cols + remaining]

        # Save final
        base_folder = os.path.join(self.cfg.base_cfg.save_dir, self.cfg.base_cfg.run_name)
        os.makedirs(base_folder, exist_ok=True)
        final_path = os.path.join(base_folder, "recursive_final.csv")
        out.to_csv(final_path, index=False)

        return out
