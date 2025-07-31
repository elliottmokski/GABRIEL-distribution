"""Task implementations for GABRIEL."""

from importlib import import_module

_lazy_imports = {
    "Rate": ".rate",
    "RateConfig": ".rate",
    "Deidentifier": ".deidentify",
    "DeidentifyConfig": ".deidentify",
    "EloRater": ".elo",
    "EloConfig": ".elo",
    "Classify": ".classify",
    "ClassifyConfig": ".classify",
    "Rank": ".rank",
    "RankConfig": ".rank",
    "Regional": ".regional",
    "RegionalConfig": ".regional",
    "RecursiveEloRater": ".recursive_elo",
    "RecursiveEloConfig": ".recursive_elo",
    "CountyCounter": ".county_counter",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name: str):
    if name in _lazy_imports:
        module = import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return __all__
