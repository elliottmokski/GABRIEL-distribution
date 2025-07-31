"""
rank.py
~~~~~~~~

This module implements a simplified yet fully featured ranking engine for
evaluating pairs of passages on a set of attributes.  It draws heavy
inspiration from the existing ``elo.py`` implementation found in the
GABRIEL distribution but removes support for the classic Elo rating
system and focuses solely on the Bradley–Terry (BT) style approach.

Key improvements and changes relative to ``elo.py`` include:

* A streamlined configuration dataclass (`RankConfig`) that exposes the
  parameters most relevant to the BT method.  Irrelevant options
  (e.g. ``rating_method``, ``k_factor``) have been removed, and
  parameter names have been harmonised with the high‑level API
  described in the calling code.  ``file_name`` is now treated as a
  stem; if an extension is provided it will be stripped automatically.

* Support for the new rankings prompt (``rankings_prompt.jinja2``)
  which allows the large language model to return one of four
  outcomes for each attribute: ``"circle"``, ``"square"``, ``"draw``
  or ``"insufficient signal"``.  ``draw`` and ``insufficient signal``
  are both treated as a tie and contribute equally to both items when
  fitting the BT model.

* A cleaned up asynchronous ``run`` method that accepts a pandas
  ``DataFrame`` and the name of the column containing the text to be
  ranked.  Each row is assigned a unique identifier based on its
  index; no external ``id_col`` argument is required.  The method
  produces a DataFrame with one row per input passage, a numeric
  rating for each attribute, optional z‑scores and standard errors,
  and writes the results to disk under ``save_dir``.

The core ranking logic remains largely unchanged from ``elo.py``
because the underlying mathematics of the BT model and the pairing
strategies continue to work well.  However, comments have been added
throughout the code to clarify intent and to highlight areas where
further experimentation (e.g. alternative information gain metrics) can
be incorporated.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import helper utilities from the gabriel package.  These modules are
# expected to be available in the runtime environment.  Should you wish
# to run this module outside of the GABRIEL distribution, you may need
# to adjust these imports accordingly.
from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils.openai_utils import get_all_responses
from gabriel.utils import safest_json


@dataclass
class RankConfig:
    """User‑visible configuration for :class:`Rank`.

    Only a minimal set of parameters are exposed to keep the API
    straightforward.  Additional hyperparameters for the underlying
    Bradley–Terry model and pairing heuristics are fixed at sensible
    values and should not generally need to be changed.  See the
    surrounding documentation for more details.
    
    Parameters
    ----------
    attributes:
        Mapping from attribute names to definitions.  A list of
        attribute names is also accepted; definitions will be set to
        empty strings.
    n_rounds:
        Number of rounds of pairwise comparisons to perform.
    matches_per_round:
        Number of matches per item per round.
    power_matching:
        Whether to use an information‑theoretic pairing heuristic.
    add_zscore:
        If ``True`` append z‑scores (normalised scores) to the output.
    compute_se:
        If ``True`` compute standard errors for each score.
    learning_rate:
        Pseudo‑count used by the BT model to regularise the win/loss
        matrix.  A larger value makes updates more conservative.
    model:
        Name of the language model to call via ``get_all_responses``.
    n_parallels:
        Number of parallel API calls to issue.
    use_dummy:
        Whether to use a dummy model for testing purposes.
    save_dir:
        Directory into which result files should be saved.
    file_name:
        Stem for the output CSV files.  If an extension is present it
        will be removed.
    additional_instructions:
        Extra, user‑supplied instructions passed to the prompt.
    """

    attributes: Union[Dict[str, str], List[str]]
    n_rounds: int = 15
    matches_per_round: int = 3
    power_matching: bool = True
    add_zscore: bool = True
    compute_se: bool = True
    learning_rate: float = 0.1
    model: str = "o4-mini"
    n_parallels: int = 400
    use_dummy: bool = False
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "rankings"
    additional_instructions: str = ""


class Rank:
    """Rank items by comparing passages pairwise on multiple attributes.

    An instance of :class:`Ranker` orchestrates the iterative process
    of sampling pairs, calling a language model to adjudicate which
    passage better exhibits each attribute, and then fitting a
    Bradley–Terry model to those outcomes.  Standard errors and
    z‑scores are optionally computed.  Results are persisted to disk
    after the final round.
    """

    def __init__(self, cfg: RankConfig, template: Optional[PromptTemplate] = None) -> None:
        """Instantiate a ranking engine.

        Parameters
        ----------
        cfg:
            User‑provided configuration.
        template:
            Optional :class:`gabriel.core.prompt_template.PromptTemplate` to
            render the comparison prompts.  If not supplied, the built‑in
            ``rankings_prompt.jinja2`` template is used.
        """
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("rankings_prompt.jinja2")
        # random state; a seed is intentionally omitted from the public
        # configuration to discourage brittle behaviour.  If
        # reproducibility is required, modify this line to pass a
        # specific seed.
        self.rng = random.Random()
        # place holders for multiway rankings and aggregated standard errors
        self.history_multi: Dict[str, List[List[str]]] = {}
        self._last_se_agg: Optional[Dict[str, float]] = None

        # internal constants for the pairing and BT algorithms.  These
        # values are deliberately not exposed through the public API as
        # they seldom need tuning and adjusting them can complicate
        # reproducibility.  Should you need to experiment with these
        # hyperparameters, modify the values below.
        self._EXPLORE_FRAC = 0.2  # fraction of random pairings per round
        self._CANDIDATE_NEIGHBORS = 20  # neighbourhood size for info gain pairing
        self._HIGH_SE_FRAC = 0.25  # fraction of high‑uncertainty items
        self._MAX_ITER = 1000  # maximum iterations for BT optimisation
        self._TOL = 1e-6  # convergence tolerance for BT
        self._SE_RIDGE = 1e-9  # ridge for standard error computation

        # The maximum number of candidate pairs to consider per pairing round.
        # When the number of items becomes very large (e.g. tens of thousands),
        # evaluating all possible pairs is intractable.  We therefore cap the
        # total number of candidate pairs by limiting the neighbourhood size
        # used when constructing candidate pairs.  The default of 200k ensures
        # that information gain pairing remains tractable even with very
        # large data sets: for example, with 10 000 items and a cap of
        # 200 000, each item will only consider approximately 20 neighbours.
        self._MAX_CANDIDATE_PAIRS_PER_ROUND = 200_000

        # timeout in seconds for a batch of language model responses.  Not
        # exposed publicly because changing it rarely benefits typical
        # workloads; if a different timeout is required this can be
        # modified here.
        self._TIMEOUT = 45.0

    # ------------------------------------------------------------------
    # Public API for adding multiway rankings
    # ------------------------------------------------------------------
    def add_multiway_ranking(self, attr: str, ranking: List[str]) -> None:
        """Record a multiway ranking for a given attribute.

        Multiway rankings are stored but not used by the current BT
        implementation.  They are retained for potential future
        extensions where a Plackett–Luce model could be incorporated.
        """
        if attr not in self.history_multi:
            self.history_multi[attr] = []
        self.history_multi[attr].append(ranking)

    # ------------------------------------------------------------------
    # BT / PL fitting utilities
    # ------------------------------------------------------------------
    def _fit_bt(
        self,
        item_ids: List[str],
        outcomes: List[Tuple[str, str]],
        pseudo: float,
        max_iter: int,
        tol: float,
        return_info: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Fit a Bradley–Terry model given pairwise outcomes.

        Parameters
        ----------
        item_ids:
            List of unique item identifiers.
        outcomes:
            List of tuples ``(winner, loser)`` representing outcomes of
            pairwise matches.  Ties can be represented by including
            both ``(a, b)`` and ``(b, a)`` in the list; each entry
            contributes a single increment to the win matrix.
        pseudo:
            Pseudo count added to both win and total match counts.  Acts
            as a smoothing prior.
        max_iter, tol:
            Control convergence of the iterative fixed‑point updates.
        return_info:
            If ``True`` return the intermediate matrices ``n_ij`` and
            ``p_ij`` for downstream standard error computation.

        Returns
        -------
        scores : dict
            Mapping from item identifier to estimated log‑skill.
        (scores, n_ij, p_ij) : tuple
            When ``return_info`` is ``True``, also return the total
            match counts and predicted win probabilities for each pair.
        """
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        # win matrix; wins[i,j] counts how many times i beat j
        wins = np.zeros((n, n), dtype=float)
        for w, l in outcomes:
            if w in idx and l in idx:
                wins[idx[w], idx[l]] += 1.0
        # total matches between each pair
        n_ij = wins + wins.T
        # total wins for each item
        w_i = wins.sum(axis=1)
        # add pseudo counts
        n_ij += pseudo
        w_i += pseudo
        # initialise skill parameters uniformly
        p = np.ones(n, dtype=float)
        for _ in range(max_iter):
            # denominator for each player in the fixed point update
            denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new
        # convert to log space and centre at zero mean
        s = np.log(p)
        s -= s.mean()
        if not return_info:
            return {item: float(val) for item, val in zip(item_ids, s)}
        # predicted win probabilities between each pair
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
        """Estimate standard errors for BT skill parameters.

        This routine computes the observed Fisher information matrix
        using the approach described by Ford (1957) and returns the
        square roots of the variances.  A small ridge term can be
        supplied to stabilise the inversion when the information matrix
        is ill‑conditioned.
        """
        n = len(s)
        # variance contribution for each pair
        q_ij = n_ij * p_ij * (1 - p_ij)
        # Fisher information matrix
        I = np.zeros((n, n), dtype=float)
        diag = q_ij.sum(axis=1)
        I[np.diag_indices(n)] = diag
        I -= q_ij
        # remove last row/col to account for non‑identifiability (sum
        # constraint).  Add ridge to the diagonal to stabilise inversion.
        I_sub = I[:-1, :-1].copy()
        I_sub[np.diag_indices(n - 1)] += ridge
        try:
            cov_sub = np.linalg.inv(I_sub)
        except np.linalg.LinAlgError:
            cov_sub = np.linalg.pinv(I_sub)
        # compute variance of the last parameter using sum constraint
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
        """Fit a Plackett–Luce model for multiway rankings.

        When every ranking is of length two this reduces to the BT
        model and defers to :meth:`_fit_bt`.  If no rankings are
        provided a zero‑centred score is returned for each item.  See
        Hunter (2004) for details on the fitting procedure.
        """
        if not rankings:
            return {i: 0.0 for i in item_ids}
        # if all rankings are of length 2, delegate to BT
        if all(len(r) == 2 for r in rankings):
            outcomes = [(r[0], r[1]) for r in rankings]
            return self._fit_bt(item_ids, outcomes, pseudo, max_iter, tol, return_info=False)
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

    # ------------------------------------------------------------------
    # Pairing strategies
    # ------------------------------------------------------------------
    def _pairs_random(self, item_ids: List[str], texts_by_id: Dict[str, str], mpr: int) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Return a set of random, unique pairs for the given items."""
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
        """Pair each item with its nearest neighbours in rating space."""
        pairs_set: set[Tuple[str, str]] = set()
        sorted_ids = sorted(item_ids, key=lambda i: current_ratings[i])
        n = len(sorted_ids)
        for i, a in enumerate(sorted_ids):
            for off in range(1, mpr + 1):
                b = sorted_ids[(i + off) % n]
                if a == b:
                    continue
                pairs_set.add(tuple(sorted((a, b))))
        # small amount of random exploration to avoid pathological pairings
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
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
        """Select pairs by maximising expected information gain while ensuring
        that every item participates in the prescribed number of matches.

        This implementation differs from the original heuristics by
        considering a bounded set of candidate pairs that scales with the
        number of items.  Each pair is assigned a score based on the
        expected reduction in uncertainty (estimated from the current
        ratings and aggregated standard errors).  Pairs with larger
        scores are chosen first, subject to the constraint that each
        item is matched exactly ``mpr`` times.  If some items remain
        unmatched after exhausting the scored pairs, additional pairs
        are filled in randomly to satisfy the per‑item quota.
        """
        n = len(item_ids)
        if n < 2:
            return []
        max_pairs = max(1, self._MAX_CANDIDATE_PAIRS_PER_ROUND)
        desired_neighbors = max_pairs // max(1, n)
        candidate_neighbors = max(mpr, min(self._CANDIDATE_NEIGHBORS, desired_neighbors))
        def logistic_clip(x: float) -> float:
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))
        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        num_high_se = max(1, int(self._HIGH_SE_FRAC * n))
        high_se_ids = sorted(item_ids, key=lambda i: se_agg.get(i, 1.0), reverse=True)[:num_high_se]
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
        remaining_capacity = max_pairs - len(candidate_pairs_set)
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        if remaining_capacity > 0:
            n_random_targets = min(n_random_targets, remaining_capacity)
            for _ in range(n_random_targets):
                if n < 2:
                    break
                a, b = self.rng.sample(item_ids, 2)
                candidate_pairs_set.add(tuple(sorted((a, b))))
        partners_count = {i: 0 for i in item_ids}
        for a, b in candidate_pairs_set:
            partners_count[a] += 1
            partners_count[b] += 1
        for i_id in item_ids:
            while partners_count[i_id] < mpr:
                potential = [x for x in item_ids if x != i_id]
                if not potential:
                    break
                j = self.rng.choice(potential)
                pair = tuple(sorted((i_id, j)))
                if pair not in candidate_pairs_set:
                    candidate_pairs_set.add(pair)
                    partners_count[i_id] += 1
                    partners_count[j] += 1
                else:
                    partners_count[i_id] += 1
                    partners_count[j] += 1
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
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        needed: Dict[str, int] = {i: mpr for i in item_ids}
        pairs_selected: List[Tuple[str, str]] = []
        pairs_seen: set[Tuple[str, str]] = set()
        for score, a, b in scored_pairs:
            if needed[a] > 0 and needed[b] > 0:
                tup = (a, b) if a < b else (b, a)
                if tup in pairs_seen:
                    continue
                pairs_selected.append(tup)
                pairs_seen.add(tup)
                needed[a] -= 1
                needed[b] -= 1
        while any(cnt > 0 for cnt in needed.values()):
            ids_needing = [i for i, cnt in needed.items() if cnt > 0]
            if not ids_needing:
                break
            # Choose an item that still needs matches
            a = self.rng.choice(ids_needing)
            # Try to pair it with any other item (not just those needing matches) to avoid self‑pairs
            potential = [x for x in item_ids if x != a]
            if not potential:
                # Degenerate case: only one item exists; cannot form a valid pair
                break
            b = self.rng.choice(potential)
            tup = (a, b) if a < b else (b, a)
            pairs_selected.append(tup)
            needed[a] -= 1
            needed[b] -= 1
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_selected]

    def _generate_pairs(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Optional[Dict[str, float]],
        se_agg: Optional[Dict[str, float]],
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Dispatch to the appropriate pairing strategy."""
        mpr = max(1, self.cfg.matches_per_round)
        # Always use information gain pairing to guarantee exact match counts
        if current_ratings is None:
            current_ratings = {i: 0.0 for i in item_ids}
        if se_agg is None or len(se_agg) != len(item_ids):
            se_full = {i: 1.0 for i in item_ids}
        else:
            se_full = se_agg
        return self._pairs_info_gain(item_ids, texts_by_id, current_ratings, se_full, mpr)

    # ------------------------------------------------------------------
    # Main ranking loop
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Execute the ranking procedure.

        Parameters
        ----------
        df:
            Input DataFrame containing the passages to be ranked.
        column_name:
            Name of the column in ``df`` that holds the text for each
            passage.
        reset_files:
            If ``True``, ignore any previously saved results and
            recompute the rankings.  Otherwise, if the final output
            file already exists on disk it will be loaded and returned
            immediately.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`get_all_responses`.  Useful for passing through
            authentication tokens or tracing settings.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per input passage and columns for
            each attribute's score, optional z‑score and standard
            error.  The DataFrame is also written to ``save_dir``.
        """
        # prepare file paths
        base_name = os.path.splitext(self.cfg.file_name)[0]
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")
        # Determine how many rounds have already been processed when
        # `reset_files` is False.  We look for files named
        # ``<base_name>_round<k>.csv`` to infer progress.  If a final
        # checkpoint exists for the last round, reuse it; otherwise we
        # resume from the next incomplete round.  When ``reset_files``
        # is ``True``, all progress is ignored and the computation
        # restarts from round 0.
        start_round = 0
        if not reset_files:
            existing_rounds: List[int] = []
            try:
                for fname in os.listdir(self.cfg.save_dir):
                    if fname.startswith(f"{base_name}_round") and fname.endswith(".csv"):
                        # Extract the integer after "round"
                        try:
                            idx_str = fname[len(base_name) + 6 : -4]  # len("_round") == 6
                            rnd_idx = int(idx_str)
                            existing_rounds.append(rnd_idx)
                        except Exception:
                            continue
            except Exception:
                existing_rounds = []
            if existing_rounds:
                last_completed = max(existing_rounds)
                # If all rounds have been processed, return the final
                # results immediately (if the checkpoint exists)
                if last_completed >= self.cfg.n_rounds - 1 and os.path.exists(final_path):
                    return pd.read_csv(final_path)
                # Otherwise resume from the next round
                start_round = last_completed + 1
        else:
            # When reset_files is True we will recompute from scratch
            start_round = 0
        # copy and prepare the input DataFrame
        df_proc = df.reset_index(drop=True).copy()
        # assign a unique identifier per row using the row index
        df_proc["_id"] = df_proc.index.astype(str)
        # extract texts and build lookup
        texts = list(zip(df_proc["_id"], df_proc[column_name].astype(str)))
        texts_by_id = {i: t for i, t in texts}
        item_ids = [i for i, _ in texts]
        # derive list of attributes
        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)
        # initialise ratings for each item/attribute
        ratings: Dict[str, Dict[str, float]] = {i: {a: 0.0 for a in attr_keys} for i in item_ids}
        # maintain a history of pairwise outcomes for each attribute
        history_pairs: Dict[str, List[Tuple[str, str]]] = {a: [] for a in attr_keys}
        # store per‑attribute standard errors across items
        se_store: Dict[str, Dict[str, float]] = {a: {i: np.nan for i in item_ids} for a in attr_keys}
        # Define attribute batches once to reuse across replay and new rounds
        attr_batches: List[List[str]] = [attr_keys[i : i + 8] for i in range(0, len(attr_keys), 8)]

        # Helper function to write the current results to the final CSV.  This
        # builds the output DataFrame from the current ``df_proc`` and
        # ``ratings``/``se_store``/``zscores`` and writes it to
        # ``final_path``.
        def _write_checkpoint() -> None:
            # Compute z‑scores for each attribute if required
            zscores_local: Dict[str, Dict[str, float]] = {}
            if self.cfg.add_zscore:
                for attr in attr_keys:
                    vals = np.array([ratings[i][attr] for i in item_ids])
                    mean = vals.mean()
                    std = vals.std(ddof=0)
                    if std == 0:
                        zscores_local[attr] = {i: 0.0 for i in item_ids}
                    else:
                        zscores_local[attr] = {
                            i: float((ratings[i][attr] - mean) / std) for i in item_ids
                        }
            # Merge computed results back into the original DataFrame copy.
            for attr in attr_keys:
                # ratings
                val_map = {i: ratings[i][attr] for i in item_ids}
                df_proc[attr] = df_proc["_id"].map(val_map)
                # standard errors
                if self.cfg.compute_se:
                    se_map = {i: se_store[attr].get(i, np.nan) for i in item_ids}
                    df_proc[f"{attr}_se"] = df_proc["_id"].map(se_map)
                # z‑scores
                if self.cfg.add_zscore:
                    z_map = zscores_local.get(attr, {i: np.nan for i in item_ids})
                    df_proc[f"{attr}_z"] = df_proc["_id"].map(z_map)
            # Reorder columns: original user columns first (excluding the internal ``_id``),
            # then for each attribute the score column, followed by the standard error and z‑score.
            original_cols = [c for c in df.columns]  # preserve the order provided by the user
            new_cols: List[str] = []
            for attr in attr_keys:
                new_cols.append(attr)
                if self.cfg.compute_se:
                    new_cols.append(f"{attr}_se")
                if self.cfg.add_zscore:
                    new_cols.append(f"{attr}_z")
            final_cols = original_cols + new_cols
            final_cols = [c for c in final_cols if c in df_proc.columns]
            df_out_local = df_proc[final_cols].copy()
            # Write the final results to disk in CSV format.  Using CSV avoids
            # Excel row limits and unnecessary overhead.
            df_out_local.to_csv(final_path, index=False)

        # If there are completed rounds and we're resuming, replay them to
        # reconstruct the ratings and uncertainties.  After each replayed
        # round we write a checkpoint to ``final_path``.
        if start_round > 0:
            for replay_rnd in range(start_round):
                round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{replay_rnd}.csv")
                if not os.path.exists(round_path):
                    break
                try:
                    # Load existing responses for this round
                    df_round = pd.read_csv(round_path)
                    df_round["Response"] = df_round["Response"].apply(lambda x: None if pd.isna(x) else x)
                except Exception:
                    continue
                # Parse each response to build history_pairs
                for ident, resp_raw in zip(df_round["Identifier"], df_round["Response"]):
                    parts = str(ident).split("|")
                    if len(parts) != 5:
                        continue
                    _, batch_idx_str, pair_idx_str, id_a, id_b = parts
                    batch_idx = int(batch_idx_str)
                    batch = attr_batches[batch_idx]
                    batch_attr_map = {str(k).strip().lower(): k for k in batch}
                    # Coerce response into a dictionary using safest_json
                    async def _coerce_dict_replay(raw: Any) -> Dict[str, Any]:
                        obj = await safest_json(raw)
                        if isinstance(obj, dict):
                            return obj
                        if isinstance(obj, str):
                            obj2 = await safest_json(obj)
                            if isinstance(obj2, dict):
                                return obj2
                        if isinstance(obj, list) and obj:
                            inner = await safest_json(obj[0])
                            if isinstance(inner, dict):
                                return inner
                        return {}
                    safe_obj = await _coerce_dict_replay(resp_raw)
                    if not safe_obj:
                        continue
                    for attr_raw, winner_raw in safe_obj.items():
                        attr_key_l = str(attr_raw).strip().lower()
                        if attr_key_l not in batch_attr_map:
                            continue
                        real_attr = batch_attr_map[attr_key_l]
                        val = winner_raw
                        if isinstance(val, dict) and "winner" in val:
                            val = val.get("winner")
                        if isinstance(val, str):
                            v = val.strip().lower()
                        else:
                            v = ""
                        if v.startswith(("cir", "c", "left", "text a")):
                            history_pairs[real_attr].append((id_a, id_b))
                        elif v.startswith(("squ", "b", "right", "text b")):
                            history_pairs[real_attr].append((id_b, id_a))
                        elif v.startswith("draw") or v.startswith("insufficient"):
                            history_pairs[real_attr].append((id_a, id_b))
                            history_pairs[real_attr].append((id_b, id_a))
                        else:
                            continue
                # After parsing all pairs for this round, update ratings
                se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
                se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
                for attr in attr_keys:
                    outcomes = history_pairs[attr]
                    if len(outcomes) == 0:
                        continue
                    bt_scores, n_ij, p_ij = self._fit_bt(
                        item_ids=item_ids,
                        outcomes=outcomes,
                        pseudo=self.cfg.learning_rate,
                        max_iter=self._MAX_ITER,
                        tol=self._TOL,
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
                            ridge=self._SE_RIDGE,
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
                # Centre ratings to zero mean for each attribute
                for attr in attr_keys:
                    vals = [ratings[i][attr] for i in item_ids]
                    mean_val = float(np.mean(vals))
                    for i in item_ids:
                        ratings[i][attr] -= mean_val
                # Write checkpoint after this replayed round
                _write_checkpoint()

        # Now proceed with new rounds starting from ``start_round``
        for rnd in range(start_round, self.cfg.n_rounds):
            # aggregate current ratings across attributes for pairing
            current_agg = {i: float(np.mean(list(ratings[i].values()))) for i in item_ids}
            se_agg_local = self._last_se_agg
            # generate pairs; on the first new round there may be no se_agg
            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if rnd > 0 or start_round > 0 else None,
                se_agg=se_agg_local if (rnd > 0 or start_round > 0) else None,
            )
            if not pairs:
                break
            prompts: List[str] = []
            ids: List[str] = []
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, ((id_a, t_a), (id_b, t_b)) in enumerate(pairs):
                    prompts.append(
                        self.template.render(
                            passage_circle=t_a,
                            passage_square=t_b,
                            attributes=attr_def_map,
                            additional_instructions=self.cfg.additional_instructions or "",
                        )
                    )
                    ids.append(f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}")
            # obtain responses from the language model for this round
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=True,
                save_path=os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv"),
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                timeout=self._TIMEOUT,
                **kwargs,
            )
            # parse each response
            # reuse the _coerce_dict function defined in the original implementation
            async def _coerce_dict(raw: Any) -> Dict[str, Any]:
                obj = await safest_json(raw)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    obj2 = await safest_json(obj)
                    if isinstance(obj2, dict):
                        return obj2
                if isinstance(obj, list) and obj:
                    inner = await safest_json(obj[0])
                    if isinstance(inner, dict):
                        return inner
                return {}
            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                parts = str(ident).split("|")
                if len(parts) != 5:
                    continue
                _, batch_idx_str, pair_idx_str, id_a, id_b = parts
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                batch_idx = int(batch_idx_str)
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    if v.startswith(("cir", "c", "left", "text a")):
                        history_pairs[real_attr].append((id_a, id_b))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        history_pairs[real_attr].append((id_b, id_a))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        continue
            # update ratings using the BT model for this round
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                if len(outcomes) == 0:
                    continue
                bt_scores, n_ij, p_ij = self._fit_bt(
                    item_ids=item_ids,
                    outcomes=outcomes,
                    pseudo=self.cfg.learning_rate,
                    max_iter=self._MAX_ITER,
                    tol=self._TOL,
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
                        ridge=self._SE_RIDGE,
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
            # Centre ratings to zero mean for each attribute
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
            # Write checkpoint after this new round
            _write_checkpoint()
        # After processing all rounds, return the final DataFrame
        # The checkpoint has already been written in the final iteration
        return pd.read_csv(final_path)
