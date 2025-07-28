"""Task implementations for GABRIEL."""

from importlib import import_module

_lazy_imports = {
    "Ratings": ".ratings",
    "RatingsConfig": ".ratings",
    "Deidentifier": ".deidentification",
    "DeidentifyConfig": ".deidentification",
    "EloRater": ".elo",
    "EloConfig": ".elo",
    "BasicClassifier": ".basic_classifier",
    "BasicClassifierConfig": ".basic_classifier",
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
