"""Task implementations for GABRIEL."""

from .simple_rating import SimpleRating
from .elo import EloRater
from .identification import Identification

__all__ = ["SimpleRating", "EloRater", "Identification"]