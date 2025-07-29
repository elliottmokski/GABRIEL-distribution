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
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import helper utilities from the gabriel package.  These modules are
# expected to be available in the runtime environment.  Should you wish
# to run this module outside of the GABRIEL distribution, you may need
# to adjust these imports accordingly.
from gabriel.utils.teleprompter import Teleprompter
from gabriel.utils.openai_utils import get_all_responses
from gabriel.utils import safest_json


@dataclass
class RankConfig:
    """Configuration parameters for :class:`Ranker`.

    Attributes
    ----------
    attributes:
        A mapping from attribute names to definitions.  Definitions may
        be empty strings; when no definition is supplied the language
        model is expected to use its own interpretation of the
        attribute.  A list of strings is also accepted and will be
        internally converted into a dict with the attribute names as
        both keys and values.
    n_rounds:
        Number of rounds of pairwise comparisons to perform.  More
        rounds generally lead to more stable rankings at the cost of
        additional API calls.
    matches_per_round:
        Number of matches (comparisons) to schedule for each item in
        each round.  Higher values provide more information but increase
        cost.
    power_matching:
        Whether to use an information‑theoretic pairing heuristic to
        select which items should be compared.  If ``False``, pairs are
        selected uniformly at random.
    power_match_mode:
        Strategy used to score candidate pairs when ``power_matching``
        is enabled.  Currently only ``"info_gain"`` is supported and
        matches the behaviour of ``elo.py``.
    power_match_explore_frac:
        Fraction of total matches per round to allocate to purely
        random comparisons in order to avoid getting stuck in local
        minima.
    power_match_candidate_neighbors:
        Number of neighbouring items to consider when building the
        candidate set for information gain based pairing.
    power_match_high_se_frac:
        Fraction of items with the highest estimated standard errors
        that should be forced into the candidate set in order to reduce
        their uncertainty.  Only relevant when ``compute_se`` is
        ``True``.
    bt_pseudo_count:
        Pseudo‑count added to the win/loss counts when fitting the BT
        model.  Acts as an implicit prior and prevents pathological
        cases when an item has no comparisons.
    bt_max_iter:
        Maximum number of iterations to run the BT fixed‑point
        optimisation.
    bt_tol:
        Convergence tolerance for the BT optimisation.  Iterations halt
        when the maximum absolute change in skill parameters falls
        below this value.
    compute_se:
        Whether to compute standard error estimates for the skill
        parameters.  If ``True``, standard errors are estimated via
        the observed Fisher information matrix; otherwise they are
        omitted.
    se_ridge:
        Ridge term applied to the Fisher information matrix when
        computing standard errors.  Helps stabilise the inversion.
    accept_multiway:
        Whether to allow multiway rankings (Plackett–Luce style) as
        inputs via the :meth:`add_multiway_ranking` API.  When
        ``False`` only pairwise comparisons are considered.
    add_zscore:
        If ``True`` (the default), z‑scores (normalised scores with
        zero mean and unit variance) will be appended to the output
        DataFrame for each attribute.
    model:
        Name of the language model to use when calling
        :func:`get_all_responses`.
    n_parallels:
        Number of parallel API calls to issue.  Keeping this value
        modest (e.g. 100) helps avoid rate limiting.
    use_dummy:
        Whether to use a dummy language model (for debugging).  If
        enabled, prompts will be echoed back as responses.
    timeout:
        Maximum time in seconds to wait for a batch of responses.
    instructions:
        Additional instructions to append to the prompt.  Included for
        completeness but unused by the current rankings prompt.
    additional_instructions:
        Extra, user‑supplied instructions that will be inserted into
        the prompt template.  May be ``None`` or an empty string.
    save_dir:
        Directory into which results should be written.  The directory
        will be created if it does not already exist.
    file_name:
        Stem for the output CSV files.  If an extension (e.g.
        ``.csv``) is provided it will be stripped off.  The final
        results file will be ``<file_name>_final.csv``.
    seed:
        Seed for the random number generator used in the pairing
        strategies.  Setting a seed makes results reproducible.
    """

    attributes: Union[Dict[str, str], List[str]]
    n_rounds: int = 15
    matches_per_round: int = 3
    power_matching: bool = True
    power_match_mode: str = "info_gain"
    power_match_explore_frac: float = 0.2
    power_match_candidate_neighbors: int = 20
    power_match_high_se_frac: float = 0.25
    bt_pseudo_count: float = 0.1
    bt_max_iter: int = 1000
    bt_tol: float = 1e-6
    compute_se: bool = True
    se_ridge: float = 1e-9
    accept_multiway: bool = False
    add_zscore: bool = True
    model: str = "o4-mini"
    n_parallels: int = 100
    use_dummy: bool = False
    timeout: float = 45.0
    instructions: str = ""
    additional_instructions: str = ""
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "rankings"
    seed: Optional[int] = None


class Ranker:
    """Rank items by comparing passages pairwise on multiple attributes.

    An instance of :class:`Ranker` orchestrates the iterative process
    of sampling pairs, calling a language model to adjudicate which
    passage better exhibits each attribute, and then fitting a
    Bradley–Terry model to those outcomes.  Standard errors and
    z‑scores are optionally computed.  Results are persisted to disk
    after the final round.
    """

    def __init__(self, teleprompter: Teleprompter, cfg: RankConfig) -> None:
        self.tele = teleprompter
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # create save directory eagerly
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        # state for multiway rankings (currently unused unless accept_multiway is True)
        self.history_multi: Dict[str, List[List[str]]] = {}
        # aggregated standard error estimates from previous round
        self._last_se_agg: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API for adding multiway rankings
    # ------------------------------------------------------------------
    def add_multiway_ranking(self, attr: str, ranking: List[str]) -> None:
        """Record a multiway ranking for a given attribute.

        If ``accept_multiway`` is enabled in the configuration, these
        rankings will be incorporated into the Plackett–Luce update of
        the BT model.  Otherwise they are ignored.
        """
        if not self.cfg.accept_multiway:
            return
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
        """Select pairs by maximising expected information gain.

        This strategy approximates the expected reduction in uncertainty
        that would result from observing the outcome of a match between
        two items.  Pairs with high combined uncertainty and high
        outcome variance are prioritised.  The implementation closely
        follows the original ``elo.py`` heuristics.
        """
        n = len(item_ids)
        if n < 2:
            return []
        # sort item identifiers by their current aggregate rating
        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        # identify high‑uncertainty items
        num_high_se = max(1, int(self.cfg.power_match_high_se_frac * n))
        high_se_ids = sorted(item_ids, key=lambda i: se_agg.get(i, 1.0), reverse=True)[:num_high_se]
        candidate_neighbors = max(1, self.cfg.power_match_candidate_neighbors)
        candidate_pairs_set: set[Tuple[str, str]] = set()
        # local neighbourhood pairs
        for i_id in item_ids:
            pos = idx_of[i_id]
            lower = max(0, pos - candidate_neighbors)
            upper = min(n, pos + candidate_neighbors + 1)
            for j in ids_sorted[lower:upper]:
                if i_id == j:
                    continue
                candidate_pairs_set.add(tuple(sorted((i_id, j))))
        # encourage exploration for high‑uncertainty items
        for hs in high_se_ids:
            others = [x for x in item_ids if x != hs]
            k = min(candidate_neighbors, len(others))
            samp = self.rng.sample(others, k)
            for j in samp:
                candidate_pairs_set.add(tuple(sorted((hs, j))))
        # additional purely random candidate pairs
        n_random_targets = int(self.cfg.power_match_explore_frac * n * mpr)
        for _ in range(n_random_targets):
            a, b = self.rng.sample(item_ids, 2)
            candidate_pairs_set.add(tuple(sorted((a, b))))
        # logistic utility function for mapping rating differences to
        # predicted outcome variances.  Clip extreme differences to
        # prevent numerical overflow.
        def logistic_clip(x: float) -> float:
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))
        # score candidate pairs
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
        # greedily select pairs, ensuring each item has mpr assignments
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
        # fill in remaining pairings randomly if necessary
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
        """Dispatch to the appropriate pairing strategy."""
        mpr = max(1, self.cfg.matches_per_round)
        if not self.cfg.power_matching or current_ratings is None:
            return self._pairs_random(item_ids, texts_by_id, mpr)
        # if information gain is requested and we have standard error estimates
        if (
            self.cfg.power_match_mode == "info_gain"
            and se_agg is not None
            and len(se_agg) == len(item_ids)
        ):
            return self._pairs_info_gain(item_ids, texts_by_id, current_ratings, se_agg, mpr)
        # fall back to adjacent pairing
        return self._pairs_adjacent(item_ids, texts_by_id, current_ratings, mpr)

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
        if not reset_files and os.path.exists(final_path):
            # load previously computed results
            return pd.read_csv(final_path)
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

        # convenience function for computing expected win probability
        def expected(r_a: float, r_b: float) -> float:
            return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

        # iterate through rounds
        for rnd in range(self.cfg.n_rounds):
            # aggregate current ratings across attributes for pairing
            current_agg = {i: float(np.mean(list(ratings[i].values()))) for i in item_ids}
            se_agg = self._last_se_agg
            # generate pairs; on the first round there is no se_agg
            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if rnd > 0 else None,
                se_agg=se_agg if rnd > 0 else None,
            )
            if not pairs:
                break
            # split attributes into batches to avoid overly long prompts
            attr_batches = [attr_keys[i : i + 8] for i in range(0, len(attr_keys), 8)]
            prompts: List[str] = []
            ids: List[str] = []
            for batch_idx, batch in enumerate(attr_batches):
                # build a mapping from lower‑cased attribute name to the
                # original attribute.  This is used later to match
                # responses back to the correct canonical attribute name.
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, ((id_a, t_a), (id_b, t_b)) in enumerate(pairs):
                    # render the rankings prompt.  Note that the
                    # "circle" passage corresponds to the first
                    # argument and "square" to the second.  The
                    # template uses the key ``additional_instructions``
                    # which we pull from the configuration.
                    template = self.tele.env.get_template("rankings_prompt.jinja2")
                    prompts.append(
                        template.render(
                            passage_circle=t_a,
                            passage_square=t_b,
                            attributes=attr_def_map,
                            additional_instructions=self.cfg.additional_instructions or "",
                        )
                    )
                    ids.append(f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}")
            # obtain responses from the language model
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=True,
                save_path=os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv"),
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                timeout=self.cfg.timeout,
                **kwargs,
            )
            # parse each response
            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                # identifiers encode round, batch, pair and item ids
                parts = str(ident).split("|")
                if len(parts) != 5:
                    # skip malformed identifiers
                    continue
                _, batch_idx_str, pair_idx_str, id_a, id_b = parts
                # robustly coerce the response into a dict
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
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                # map lowercased attribute names back to canonical keys
                batch_idx = int(batch_idx_str)
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                # iterate through each attribute in the response
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    # determine the outcome
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    # normalise and extract the leading token
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    # decide on winner/loser or tie
                    if v.startswith(("cir", "c", "left", "text a")):
                        winner, loser = id_a, id_b
                        history_pairs[real_attr].append((winner, loser))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        winner, loser = id_b, id_a
                        history_pairs[real_attr].append((winner, loser))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        # tie: record both directions to award equal credit
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        # ignore unrecognised outcomes
                        continue
            # update ratings using the BT (or PL if multiway) model
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                rankings: List[List[str]] = []
                # if PL is accepted and multiway rankings have been added
                if self.cfg.accept_multiway:
                    rankings = self.history_multi.get(attr, [])
                if self.cfg.accept_multiway and rankings:
                    # fit PL scores and ignore standard errors (unsupported)
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
            # aggregate standard errors across attributes
            if self.cfg.compute_se:
                for i in item_ids:
                    if se_agg_counts[i] > 0:
                        se_agg_next[i] /= se_agg_counts[i]
                    else:
                        se_agg_next[i] = 1.0
                self._last_se_agg = se_agg_next
            # centre ratings to zero mean for each attribute after each round
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
        # build final output DataFrame
        rows: List[Dict[str, Any]] = []
        zscores: Dict[str, Dict[str, float]] = {}
        if self.cfg.add_zscore:
            for attr in attr_keys:
                vals = np.array([ratings[i][attr] for i in item_ids])
                mean = vals.mean()
                std = vals.std(ddof=0)
                if std == 0:
                    z = {i: 0.0 for i in item_ids}
                else:
                    z = {i: float((ratings[i][attr] - mean) / std) for i in item_ids}
                zscores[attr] = z
        for i in item_ids:
            row = {"identifier": i, column_name: texts_by_id[i]}
            for attr in attr_keys:
                row[attr] = ratings[i][attr]
                if self.cfg.add_zscore:
                    row[f"{attr}_z"] = zscores[attr][i]
                if self.cfg.compute_se:
                    row[f"{attr}_se"] = se_store[attr].get(i, np.nan)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        # write the final results to disk
        df_out.to_csv(final_path, index=False)
        return df_out
