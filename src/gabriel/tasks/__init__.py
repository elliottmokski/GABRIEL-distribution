"""Task implementations for GABRIEL."""

from .simple_rating import SimpleRating
from .elo import EloRating
from .identification import Identification

__all__ = ["SimpleRating", "EloRating", "Identification"]
