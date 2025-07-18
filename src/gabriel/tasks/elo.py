"""Advanced Elo rating implementation."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.teleprompter import Teleprompter
from ..utils.openai_utils import get_all_responses
from ..utils import safe_json


@dataclass
class EloConfig:
    """Configuration for :class:`EloRater`."""

    attributes: Union[dict, List[str]]
    n_rounds: int = 15
    matches_per_round: int = 3
    power_matching: bool = True
    power_match_mode: str = "info_gain"
    power_match_explore_frac: float = 0.2
    power_match_candidate_neighbors: int = 20
    power_match_high_se_frac: float = 0.25
    rating_method: str = "bt"
    k_factor: float = 32.0
    bt_pseudo_count: float = 0.1
    bt_max_iter: int = 1000
    bt_tol: float = 1e-6
    compute_se: bool = True
    se_ridge: float = 1e-9
    accept_multiway: bool = False
    add_zscore: bool = True
    final_filename: str = "ratings_final.csv"
    save_per_round: bool = True
    n_parallels: int = 400
    model: str = "o4-mini"
    use_dummy: bool = False
    timeout: float = 45.0
    print_example_prompt: bool = True
    instructions: str = ""
    additional_guidelines: str = ""
    save_dir: str = os.path.expanduser("~/Documents/runs")
    run_name: str = f"elo_{datetime.now():%Y%m%d_%H%M%S}"
    seed: Optional[int] = None


class EloRater:
    """Pairwise (or multiway in future) Elo/BT rating of texts."""

    def __init__(self, teleprompter: Teleprompter, cfg: EloConfig) -> None:
        self.tele = teleprompter
        self.cfg = cfg
        self.save_path = os.path.join(cfg.save_dir, cfg.run_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.rng = random.Random(cfg.seed)

        self.history_multi: Dict[str, List[List[str]]] = {}
        self._last_se_agg: Optional[Dict[str, float]] = None

    def add_multiway_ranking(self, attr: str, ranking: List[str]) -> None:
        if attr not in self.history_multi:
            self.history_multi[attr] = []
        self.history_multi[attr].append(ranking)


    def _fit_bt(
        self,
        item_ids: List[str],
        outcomes: List[Tuple[str, str]],
        pseudo: float,
        max_iter: int,
        tol: float,
        return_info: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}

        wins = np.zeros((n, n), dtype=float)
        for w, l in outcomes:
            if w in idx and l in idx:
                wins[idx[w], idx[l]] += 1.0

        n_ij = wins + wins.T
        w_i = wins.sum(axis=1)

        n_ij += pseudo
        w_i += pseudo

        p = np.ones(n, dtype=float)

        for _ in range(max_iter):
            denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new

        s = np.log(p)
        s -= s.mean()

        if not return_info:
            return {item: float(val) for item, val in zip(item_ids, s)}

        exp_s = np.exp(s)
        p_ij = exp_s[:, None] / (exp_s[:, None] + exp_s[None, :])

        return {item: float(val) for item, val in zip(item_ids, s)}, n_ij, p_ij

    def _bt_standard_errors(
        self,
        s: np.ndarray,
        n_ij: np.ndarray,
        p_ij: np.ndarray,
        ridge: float,
    ) -> np.ndarray:
        n = len(s)
        q_ij = n_ij * p_ij * (1 - p_ij)

        I = np.zeros((n, n), dtype=float)
        diag = q_ij.sum(axis=1)
        I[np.diag_indices(n)] = diag
        I -= q_ij

        I_sub = I[:-1, :-1].copy()
        I_sub[np.diag_indices(n - 1)] += ridge

        try:
            cov_sub = np.linalg.inv(I_sub)
        except np.linalg.LinAlgError:
            cov_sub = np.linalg.pinv(I_sub)

        ones = np.ones((n - 1, 1))
        var_last = float(ones.T @ cov_sub @ ones)

        se = np.zeros(n, dtype=float)
        se[:-1] = np.sqrt(np.diag(cov_sub))
        se[-1] = np.sqrt(var_last)
        return se

    def _fit_pl(
        self,
        item_ids: List[str],
        rankings: List[List[str]],
        pseudo: float,
        max_iter: int,
        tol: float,
    ) -> Dict[str, float]:
        if not rankings:
            return {i: 0.0 for i in item_ids}

        if all(len(r) == 2 for r in rankings):
            outcomes = [(r[0], r[1]) for r in rankings]
            return self._fit_bt(
                item_ids, outcomes, pseudo, max_iter, tol, return_info=False
            )

        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        w_i = np.zeros(n, dtype=float)
        rankings_idx = []
        for r in rankings:
            r_idx = [idx[x] for x in r if x in idx]
            if len(r_idx) < 2:
                continue
            rankings_idx.append(r_idx)
            for i_ in r_idx:
                w_i[i_] += 1.0
        if len(rankings_idx) == 0:
            return {i: 0.0 for i in item_ids}

        w_i += pseudo
        p = np.ones(n, dtype=float)

        for _ in range(max_iter):
            denom = np.zeros(n, dtype=float)
            for r_idx in rankings_idx:
                remaining = np.array(r_idx, dtype=int)
                sum_p = p[remaining].sum()
                for i_ in r_idx:
                    denom[i_] += 1.0 / sum_p
                    sum_p -= p[i_]
            denom[denom == 0] = 1e-12
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new

        s = np.log(p)
        s -= s.mean()
        return {item: float(val) for item, val in zip(item_ids, s)}

    def _pairs_random(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        pairs_set: set[Tuple[str, str]] = set()
        for a in item_ids:
            others = [x for x in item_ids if x != a]
            if not others:
                continue
            k = min(mpr, len(others))
            opponents = self.rng.sample(others, k)
            for b in opponents:
                pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_adjacent(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        pairs_set: set[Tuple[str, str]] = set()
        sorted_ids = sorted(item_ids, key=lambda i: current_ratings[i])
        n = len(sorted_ids)
        for i, a in enumerate(sorted_ids):
            for off in range(1, mpr + 1):
                b = sorted_ids[(i + off) % n]
                if a == b:
                    continue
                pairs_set.add(tuple(sorted((a, b))))
        n_random_targets = int(self.cfg.power_match_explore_frac * n * mpr)
        for _ in range(n_random_targets):
            if n < 2:
                break
            a, b = self.rng.sample(item_ids, 2)
            pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_info_gain(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        se_agg: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        n = len(item_ids)
        if n < 2:
            return []
        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        num_high_se = max(1, int(self.cfg.power_match_high_se_frac * n))
        high_se_ids = sorted(item_ids, key=lambda i: se_agg[i], reverse=True)[:num_high_se]
        candidate_neighbors = max(1, self.cfg.power_match_candidate_neighbors)
        candidate_pairs_set: set[Tuple[str, str]] = set()
        for i_id in item_ids:
            pos = idx_of[i_id]
            lower = max(0, pos - candidate_neighbors)
            upper = min(n, pos + candidate_neighbors + 1)
            for j in ids_sorted[lower:upper]:
                if i_id == j:
                    continue
                candidate_pairs_set.add(tuple(sorted((i_id, j))))
        for hs in high_se_ids:
            others = [x for x in item_ids if x != hs]
            k = min(candidate_neighbors, len(others))
            samp = self.rng.sample(others, k)
            for j in samp:
                candidate_pairs_set.add(tuple(sorted((hs, j))))
        n_random_targets = int(self.cfg.power_match_explore_frac * n * mpr)
        for _ in range(n_random_targets):
            a, b = self.rng.sample(item_ids, 2)
            candidate_pairs_set.add(tuple(sorted((a, b))))

        def logistic_clip(x: float) -> float:
            """Stable logistic function for large |x|."""
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))

        scored_pairs: List[Tuple[float, str, str]] = []
        for a, b in candidate_pairs_set:
            diff = current_ratings[a] - current_ratings[b]
            p = logistic_clip(diff)
            outcome_var = p * (1 - p)
            var_a = se_agg.get(a, 1.0) ** 2
            var_b = se_agg.get(b, 1.0) ** 2
            param_unc = var_a + var_b
            score = outcome_var * param_unc
            scored_pairs.append((score, a, b))

        needed = {i: mpr for i in item_ids}
        pairs_selected: List[Tuple[str, str]] = []
        pairs_selected_set: set[Tuple[str, str]] = set()
        for score, a, b in sorted(scored_pairs, key=lambda x: x[0], reverse=True):
            if needed[a] > 0 and needed[b] > 0:
                tup = (a, b) if a < b else (b, a)
                if tup in pairs_selected_set:
                    continue
                pairs_selected.append(tup)
                pairs_selected_set.add(tup)
                needed[a] -= 1
                needed[b] -= 1

        ids_needing = [i for i, cnt in needed.items() if cnt > 0]
        attempts = 0
        while ids_needing and attempts < 10000:
            attempts += 1
            a = self.rng.choice(ids_needing)
            others = [x for x in item_ids if x != a]
            if not others:
                needed[a] = 0
                ids_needing = [i for i, cnt in needed.items() if cnt > 0]
                continue
            b = self.rng.choice(others)
            tup = (a, b) if a < b else (b, a)
            if tup not in pairs_selected_set:
                pairs_selected_set.add(tup)
                pairs_selected.append(tup)
                needed[a] -= 1
                needed[b] -= 1
            ids_needing = [i for i, cnt in needed.items() if cnt > 0]

        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_selected_set]

    def _generate_pairs(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Optional[Dict[str, float]],
        se_agg: Optional[Dict[str, float]],
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        mpr = max(1, self.cfg.matches_per_round)
        if not self.cfg.power_matching or not current_ratings:
            return self._pairs_random(item_ids, texts_by_id, mpr)
        if (
            self.cfg.power_match_mode == "info_gain"
            and se_agg is not None
            and len(se_agg) == len(item_ids)
        ):
            return self._pairs_info_gain(
                item_ids, texts_by_id, current_ratings, se_agg, mpr
            )
        return self._pairs_adjacent(item_ids, texts_by_id, current_ratings, mpr)

    async def run(self, df: pd.DataFrame, text_col: str, id_col: str) -> pd.DataFrame:
        final_path = os.path.join(self.save_path, self.cfg.final_filename)
        if os.path.exists(final_path):
            return pd.read_csv(final_path)

        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        texts = list(zip(df[id_col], df[text_col]))
        texts_by_id = {i: t for i, t in texts}
        item_ids = [i for i, _ in texts]

        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)

        ratings: Dict[str, Dict[str, float]] = {
            i: {a: 0.0 for a in attr_keys} for i in item_ids
        }
        history_pairs: Dict[str, List[Tuple[str, str]]] = {a: [] for a in attr_keys}

        def expected(r_a: float, r_b: float) -> float:
            return 1 / (1 + 10 ** ((r_b - r_a) / 400))

        se_store: Dict[str, Dict[str, float]] = {a: {i: np.nan for i in item_ids} for a in attr_keys}

        for rnd in range(self.cfg.n_rounds):
            current_agg = {i: float(np.mean(list(ratings[i].values()))) for i in item_ids}
            se_agg = self._last_se_agg

            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if rnd > 0 else None,
                se_agg=se_agg if rnd > 0 else None,
            )
            if not pairs:
                break

            attr_batches = [attr_keys[i:i + 8] for i in range(0, len(attr_keys), 8)]
            prompts, ids = [], []
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, ((id_a, t_a), (id_b, t_b)) in enumerate(pairs):
                    prompts.append(
                        self.tele.generic_elo_prompt(
                            text_circle=t_a,
                            text_square=t_b,
                            attributes=attr_def_map,
                            instructions=self.cfg.instructions,
                            additional_guidelines=self.cfg.additional_guidelines,
                        )
                    )
                    ids.append(f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}")
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=True,
                save_path=os.path.join(self.save_path, f"round{rnd}.csv"),
                reset_files=False,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                print_example_prompt=self.cfg.print_example_prompt,
            )
            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                try:
                    rnd_i, batch_idx, pair_idx, a_id, b_id = ident.split("|")
                    batch_idx = int(batch_idx)
                except Exception:
                    continue

                safe = safe_json(resp)
                if isinstance(resp, list) and not safe:
                    safe = safe_json(resp[0])
                if not isinstance(safe, dict) or not safe:
                    continue

                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}

                for attr_raw, winner_raw in safe.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]

                    w = str(winner_raw).strip().lower()
                    if w.startswith("cir"):
                        winner, loser = a_id, b_id
                    elif w.startswith("squ"):
                        winner, loser = b_id, a_id
                    else:
                        continue

                    history_pairs[real_attr].append((winner, loser))

                    if self.cfg.rating_method.lower() == "elo":
                        exp_a = expected(ratings[a_id][real_attr], ratings[b_id][real_attr])
                        exp_b = 1 - exp_a
                        score_a, score_b = (1, 0) if winner == a_id else (0, 1)
                        ratings[a_id][real_attr] += self.cfg.k_factor * (score_a - exp_a)
                        ratings[b_id][real_attr] += self.cfg.k_factor * (score_b - exp_b)

            method = self.cfg.rating_method.lower()

            if method in ("bt", "pl"):
                se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
                se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}

                for attr in attr_keys:
                    outcomes = history_pairs[attr]
                    rankings = []
                    if method == "pl" and self.cfg.accept_multiway:
                        rankings = self.history_multi.get(attr, [])

                    if method == "pl" and rankings:
                        pl_scores = self._fit_pl(
                            item_ids=item_ids,
                            rankings=rankings,
                            pseudo=self.cfg.bt_pseudo_count,
                            max_iter=self.cfg.bt_max_iter,
                            tol=self.cfg.bt_tol,
                        )
                        for i in item_ids:
                            ratings[i][attr] = pl_scores[i]
                        for i in item_ids:
                            se_store[attr][i] = np.nan
                    else:
                        if len(outcomes) == 0:
                            continue
                        bt_scores, n_ij, p_ij = self._fit_bt(
                            item_ids=item_ids,
                            outcomes=outcomes,
                            pseudo=self.cfg.bt_pseudo_count,
                            max_iter=self.cfg.bt_max_iter,
                            tol=self.cfg.bt_tol,
                            return_info=True,
                        )
                        for i in item_ids:
                            ratings[i][attr] = bt_scores[i]

                        if self.cfg.compute_se:
                            s_vec = np.array([bt_scores[i] for i in item_ids])
                            se_vec = self._bt_standard_errors(
                                s=s_vec,
                                n_ij=n_ij,
                                p_ij=p_ij,
                                ridge=self.cfg.se_ridge,
                            )
                            for i, se_val in zip(item_ids, se_vec):
                                se_store[attr][i] = float(se_val)
                                se_agg_next[i] += float(se_val)
                                se_agg_counts[i] += 1

                if self.cfg.compute_se:
                    for i in item_ids:
                        if se_agg_counts[i] > 0:
                            se_agg_next[i] /= se_agg_counts[i]
                        else:
                            se_agg_next[i] = 1.0
                    self._last_se_agg = se_agg_next

                if self.cfg.save_per_round:
                    self._save_round_ratings(
                        ratings=ratings,
                        attr_keys=attr_keys,
                        round_idx=rnd,
                        suffix=f"{method}_round",
                        include_se=self.cfg.compute_se,
                        se_store=se_store,
                    )

            elif method == "elo":
                for attr in attr_keys:
                    vals = [ratings[i][attr] for i in item_ids]
                    mean_val = float(np.mean(vals))
                    for i in item_ids:
                        ratings[i][attr] -= mean_val
                if self.cfg.save_per_round:
                    self._save_round_ratings(
                        ratings=ratings,
                        attr_keys=attr_keys,
                        round_idx=rnd,
                        suffix="elo_round",
                        include_se=False,
                        se_store=None,
                    )

        rows = []
        zscores: Dict[str, Dict[str, float]] = {}
        if self.cfg.add_zscore:
            for attr in attr_keys:
                vals = np.array([ratings[i][attr] for i in item_ids])
                mean, std = vals.mean(), vals.std(ddof=0)
                if std == 0:
                    z = {i: 0.0 for i in item_ids}
                else:
                    z = {i: float((ratings[i][attr] - mean) / std) for i in item_ids}
                zscores[attr] = z

        for i in item_ids:
            row = {"identifier": i, "text": texts_by_id[i]}
            for attr in attr_keys:
                row[attr] = ratings[i][attr]
                if self.cfg.add_zscore:
                    row[f"{attr}_z"] = zscores[attr][i]
                if self.cfg.compute_se:
                    row[f"{attr}_se"] = se_store[attr].get(i, np.nan)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(final_path, index=False)
        return df_out

    def _save_round_ratings(
        self,
        ratings: Dict[str, Dict[str, float]],
        attr_keys: List[str],
        round_idx: int,
        suffix: str,
        include_se: bool,
        se_store: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        rows = []
        for i, attr_map in ratings.items():
            row = {"identifier": i}
            for attr in attr_keys:
                row[attr] = attr_map[attr]
                if include_se and se_store is not None:
                    row[f"{attr}_se"] = se_store[attr].get(i, np.nan)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(self.save_path, f"{suffix}{round_idx}.csv"), index=False
        )
