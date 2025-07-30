from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd

from elo_rater import EloConfig, EloRater
from regional import Regional
from utility_functions import create_county_choropleth


class CountyCounter:
    """Run regional analysis on counties and rate them via ELO."""

    def __init__(
        self,
        df: pd.DataFrame,
        county_col: str,
        topics: List[str],
        *,
        fips_col: Optional[str] = None,
        save_dir: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        model_regional: str = "o4-mini",
        model_elo: str = "o4-mini",
        reasoning_effort: str = "medium",
        search_context_size: str = "medium",
        n_parallels: int = 400,
        n_elo_rounds: int = 15,
        elo_timeout: float = 60.0,
        use_dummy: bool = False,
        additional_instructions: str = "",
        elo_instructions: str = "",
        z_score_choropleth: bool = True,  # Whether to plot z-score choropleths (default True for user intent)
        elo_attributes: Optional[Dict] = None,  # Optional dict of {attr: description}
    ) -> None:
        self.df = df.copy()
        self.county_col = county_col
        self.fips_col = fips_col
        self.topics = topics
        self.model_regional = model_regional
        self.model_elo = model_elo
        self.n_parallels = n_parallels
        self.n_elo_rounds = n_elo_rounds
        self.use_dummy = use_dummy
        self.additional_instructions = additional_instructions
        self.elo_instructions = elo_instructions
        self.reasoning_effort = reasoning_effort
        self.search_context_size = search_context_size
        self.z_score_choropleth = z_score_choropleth
        self.elo_attributes = elo_attributes
        self.elo_timeout = elo_timeout

        run_name = run_name or f"county_counter_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(save_dir, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.regional = Regional(
            df=self.df,
            region_col=self.county_col,
            topics=self.topics,
            save_dir=self.save_path,
            run_name="regional",
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=self.additional_instructions,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=True,
        )

    async def run(self, *, reset_files: bool = False) -> pd.DataFrame:
        reports_df = await self.regional.run(reset_files=reset_files)
        results = reports_df[["region"]].copy()

        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(
                columns={"region": "identifier", topic: "text"}
            )
            # Determine attributes for this topic
            if self.elo_attributes:
                attributes = self.elo_attributes
            else:
                attributes = [topic]
            cfg = EloConfig(
                attributes=attributes,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_{topic}",
                use_dummy=self.use_dummy,
                instructions=self.elo_instructions,
                print_example_prompt=False,
                timeout=self.elo_timeout,
            )
            rater = EloRater(cfg)
            elo_df = await rater.run(df_topic, text_col="text", id_col="identifier")
            elo_df["identifier"] = elo_df["identifier"].astype(str)
            results["region"] = results["region"].astype(str)
            # If elo_attributes, merge all columns from ELO output except text, using attribute key as column name
            if self.elo_attributes:
                for attr in [k for k in elo_df.columns if k not in ("identifier", "text")]:
                    temp_col = f"_elo_temp_{attr}"
                    results = results.merge(
                        elo_df[["identifier", attr]].rename(columns={attr: temp_col}),
                        left_on="region",
                        right_on="identifier",
                        how="left"
                    )
                    results[attr] = results[temp_col]
                    results = results.drop(columns=["identifier", temp_col])
            else:
                # Use a unique temporary column name to avoid accidental duplication
                temp_col = f"_elo_temp_{topic}"
                results = results.merge(
                    elo_df[["identifier", topic]].rename(columns={topic: temp_col}),
                    left_on="region",
                    right_on="identifier",
                    how="left"
                )
                results[topic] = results[temp_col]
                results = results.drop(columns=["identifier", temp_col])

        if self.fips_col and self.fips_col in self.df.columns:
            merged = self.df[[self.county_col, self.fips_col]].drop_duplicates()
            # Ensure FIPS is string and zero-padded
            merged[self.fips_col] = merged[self.fips_col].astype(str).str.zfill(5)
            results = results.merge(
                merged, left_on="region", right_on=self.county_col
            )
            if self.elo_attributes:
                for attr in self.elo_attributes.keys():
                    map_path = os.path.join(self.save_path, f"map_{attr}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=attr,
                        title=f"ELO Rating for {attr}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )
            else:
                for topic in self.topics:
                    map_path = os.path.join(self.save_path, f"map_{topic}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=topic,
                        title=f"ELO Rating for {topic}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )
        results.to_csv(os.path.join(self.save_path, "county_elo.csv"), index=False)
        return results
