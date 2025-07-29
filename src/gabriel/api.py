import asyncio
import os
import pandas as pd

from .tasks import (
    Ratings,
    RatingsConfig,
    BasicClassifier,
    BasicClassifierConfig,
)

def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    additional_instructions: str | None = None,
    model: str = "o4-mini",
    n_parallels: int = 100,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "ratings.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Ratings`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = RatingsConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        **cfg_kwargs,
    )
    return asyncio.run(Ratings(cfg).run(df, column_name, reset_files=reset_files))

def classify(
    df: pd.DataFrame,
    column_name: str,
    *,
    labels: dict[str, str],
    save_dir: str,
    additional_instructions: str | None = None,
    model: str = "o4-mini",
    n_parallels: int = 400,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "basic_classifier_responses.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.BasicClassifier`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = BasicClassifierConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        additional_instructions=additional_instructions or "",
        use_dummy=use_dummy,
        **cfg_kwargs,
    )
    return asyncio.run(BasicClassifier(cfg).run(df, column_name, reset_files=reset_files))
