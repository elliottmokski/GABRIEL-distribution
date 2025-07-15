"""Task implementations for GABRIEL."""

from importlib import import_module

_lazy_imports = {
    "Ratings": ".ratings",
    "Deidentifier": ".deidentification",
    "EloRater": ".elo",
    "BasicClassifier": ".basic_classifier",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name: str):
    if name in _lazy_imports:
        module = import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return __all__
