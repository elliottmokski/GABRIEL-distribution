import os
import random
import pandas as pd
import numpy as np
import asyncio
from typing import List, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
from gabriel import foundational_functions
from gabriel.openai_api_calls import OpenAIClient  # Adjust import if needed

class ELOComparator:
    def __init__(
        self,
        openai_client: OpenAIClient,
        texts: List[str],
        axis_of_interest: str,
        user_description: str,
        prompt_template_path: str = "elo_prompt.jinja",
        k_factor: float = 32.0,
    ):
        """
        :param openai_client: An instance of your OpenAIClient.
        :param texts: A list of texts (strings) to be compared in ELO.
        :param axis_of_interest: The attribute/axis to compare.
        :param user_description: Extra clarifications about how to judge the axis.
        :param prompt_template_path: Path to the Jinja2 template for the prompt.
        :param k_factor: K-factor for ELO rating updates.
        """
        self.openai_client = openai_client
        self.axis_of_interest = axis_of_interest
        self.user_description = user_description
        self.k_factor = k_factor

        # Initialize ELO ratings (e.g., 1200).
        self.elo_df = pd.DataFrame({
            'text': texts,
            'elo': [1200.0] * len(texts)
        })

        # Prepare Jinja environment & template
        package_dir = os.path.dirname(os.path.abspath(foundational_functions.__file__))
        templates_dir = os.path.join(package_dir, 'Prompts')
        print(templates_dir)

        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape()
        )
        self.prompt_template = self.jinja_env.get_template(prompt_template_path)

    @staticmethod
    def _expected_score(elo_a: float, elo_b: float) -> float:
        """
        Standard ELO expected score for text A vs. text B.
        """
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def _update_elo(self, elo_a: float, elo_b: float, score_a: float) -> Tuple[float, float]:
        """
        ELO update:
          - elo_a, elo_b: current ELO ratings of texts A & B
          - score_a: 1.0 if A won, 0.0 if B won
        """
        exp_a = self._expected_score(elo_a, elo_b)
        exp_b = self._expected_score(elo_b, elo_a)

        score_b = 1.0 - score_a
        new_a = elo_a + self.k_factor * (score_a - exp_a)
        new_b = elo_b + self.k_factor * (score_b - exp_b)
        return new_a, new_b

    def _parse_winner(self, response: str) -> float:
        """
        Returns 1.0 if the model output indicates "Circle" won, else 0.0 for "Square".
        By design, the model must answer "Circle" or "Square".
        """
        cleaned = response.strip().lower()
        if "circle" in cleaned:
            return 1.0
        return 0.0

    def _render_prompt(self, circle_text: str, square_text: str) -> str:
        """
        Renders the Jinja template with the current axis, user_description,
        and whichever text is assigned Circle vs. Square.
        """
        return self.prompt_template.render(
            axis=self.axis_of_interest,
            user_description=self.user_description,
            circle_text=circle_text,
            square_text=square_text
        )

    def _select_smart_matchups(self) -> List[Tuple[int, int]]:
        """
        Creates a list of matchups to be used in one round of ELO, pairing up
        texts that have close ELO or, if all ELO are the same, random pairings.

        Returns a list of (idx1, idx2) pairs.
        """
        # If all ELO are exactly the same, random pairings:
        if len(self.elo_df['elo'].unique()) == 1:
            idxs = list(range(len(self.elo_df)))
            random.shuffle(idxs)
            pairs = [(idxs[i], idxs[i+1]) for i in range(0, len(idxs) - 1, 2)]
        else:
            # Sort by ELO ascending
            sorted_df = self.elo_df.sort_values(by='elo').reset_index()
            pairs = []
            for i in range(0, len(sorted_df) - 1, 2):
                idx1 = sorted_df.loc[i, 'index']
                idx2 = sorted_df.loc[i+1, 'index']
                pairs.append((idx1, idx2))
        return pairs

    async def run_elo(self, num_rounds: int = 10, parallel_calls: int = 10, model = "gpt-4o-mini"):
        """
        Conducts multiple "rounds" of matchups. Each round, we:
          - select pairs
          - build prompts (in one go)
          - make a single call to get_all_responses() for all pairs (parallel)
          - parse results, update ELO
        :param num_rounds: how many times we do the pairing-evaluation cycle
        :param parallel_calls: how many parallel calls to allow in get_all_responses()
        """
        for round_index in range(num_rounds):
            pairs = self._select_smart_matchups()
            if not pairs:
                continue

            # We'll store the "old ELO" at the start of this round
            # so that each match uses the same reference for ELO.
            # This means we won't update them in-between matches for this single round.
            # We'll do all updates after we get the responses.
            old_elos = self.elo_df['elo'].values.copy()

            # Step 1: Build the prompts for each pair
            match_data = []
            for match_i, (idx1, idx2) in enumerate(pairs):
                elo_a = old_elos[idx1]
                elo_b = old_elos[idx2]
                text_a = self.elo_df.at[idx1, 'text']
                text_b = self.elo_df.at[idx2, 'text']

                # Randomly assign which text is Circle vs Square
                if random.random() < 0.5:
                    circle_idx, square_idx = idx1, idx2
                else:
                    circle_idx, square_idx = idx2, idx1

                circle_text = self.elo_df.at[circle_idx, 'text']
                square_text = self.elo_df.at[square_idx, 'text']

                prompt = self._render_prompt(circle_text, square_text)

                match_data.append({
                    'identifier': f"round{round_index}_match{match_i}",
                    'idx1': idx1,
                    'idx2': idx2,
                    'elo_a': elo_a,
                    'elo_b': elo_b,
                    'circle_idx': circle_idx,   # which text is assigned circle
                    'square_idx': square_idx,   # which text is assigned square
                    'prompt': prompt,
                })

            # Step 2: Collect all prompts to call in parallel
            prompts = [m['prompt'] for m in match_data]
            identifiers = [m['identifier'] for m in match_data]

            # Step 3: Use get_all_responses for parallel calls
            # The "n_parallels" controls concurrency. Adjust for your usage or token limits.
            responses_df = await self.openai_client.get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                n_parallels=parallel_calls,  # how many parallel requests
                model=model,        # or whichever model you like
                system_instruction=(
                    "Compare the texts on the given axis. Return only Circle or Square."
                ),
                max_tokens=20,
                temperature=0.0,
            )

            # Step 4: Parse the responses and update ELO for each match
            for row in responses_df.itertuples():
                # Row has: row.Identifier, row.Response, row._2 = TimeTaken, etc.
                response_list = row.Response  # usually a list of strings
                if not response_list or not isinstance(response_list, list):
                    # if no response or invalid
                    continue

                response_text = response_list[0]
                # Find the match_data that corresponds to this identifier
                match = next((m for m in match_data if m['identifier'] == row.Identifier), None)
                if match is None:
                    continue

                # Parse winner
                score_circle = self._parse_winner(response_text)

                # Now figure out if circle is idx1 or idx2
                # If circle_idx == idx1 => text A is circle => score_a = score_circle
                # else score_a = 1 - score_circle
                if match['circle_idx'] == match['idx1']:
                    score_a = score_circle
                else:
                    score_a = 1.0 - score_circle

                # We retrieve the old ELO from the match record
                elo_a_old = match['elo_a']
                elo_b_old = match['elo_b']

                new_a, new_b = self._update_elo(elo_a_old, elo_b_old, score_a)

                # Write back the new ELO into the DataFrame
                self.elo_df.at[match['idx1'], 'elo'] = new_a
                self.elo_df.at[match['idx2'], 'elo'] = new_b

            # (Optional) Re-sort after each round
            self.elo_df.sort_values(by='elo', ascending=False, inplace=True)
            self.elo_df.reset_index(drop=True, inplace=True)

    def get_results(self) -> pd.DataFrame:
        """
        Returns the current ELO DataFrame after run_elo() is called.
        """
        return self.elo_df