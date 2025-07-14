"""GABRIEL: LLM-based social science analysis toolkit."""

from .tasks.simple_rating import SimpleRating
from .tasks.elo import EloRater

from .tasks.identification import Identification

__all__ = [
    "SimpleRating",
    "EloRater",
    "Identification",
]
