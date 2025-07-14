"""GABRIEL: LLM-based social science analysis toolkit."""

from importlib.metadata import PackageNotFoundError, version as _v

try:
    __version__ = _v("gabriel")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = [
    "SimpleRating",
    "EloRater",
    "Ratings",
    "Deidentifier",
    "Identification",
]

def __getattr__(name: str):
    if name == "SimpleRating":
        from .tasks.simple_rating import SimpleRating

        return SimpleRating
    if name == "EloRater":
        from .tasks.elo import EloRater

        return EloRater
    if name == "Ratings":
        from .tasks.ratings import Ratings

        return Ratings
    if name == "Deidentifier":
        from .tasks.deidentification import Deidentifier

        return Deidentifier
    if name == "Identification":
        from .tasks.identification import Identification

        return Identification
    raise AttributeError(name)
