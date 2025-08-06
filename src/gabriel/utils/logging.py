"""Simple logging helpers with configurable verbosity.

This module centralises logging configuration for the project.  Users can
control verbosity either programmatically through :func:`set_log_level` or via
the ``GABRIEL_LOG_LEVEL`` environment variable.  Levels mirror typical logging
conventions and add a "silent" option which suppresses all log output.
"""

from __future__ import annotations

import logging
import os
from typing import Union

# ---------------------------------------------------------------------------
# Verbosity handling
# ---------------------------------------------------------------------------

LOG_LEVELS = {
    "silent": logging.CRITICAL + 1,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _parse_level(level: Union[str, int, None]) -> int:
    """Translate a human friendly level to ``logging`` constants."""

    if isinstance(level, str):
        return LOG_LEVELS.get(level.lower(), logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


CURRENT_LEVEL = _parse_level(os.getenv("GABRIEL_LOG_LEVEL", "info"))


def set_log_level(level: Union[str, int]) -> None:
    """Set the global logging level for all GABRIEL loggers."""

    global CURRENT_LEVEL
    CURRENT_LEVEL = _parse_level(level)
    root = logging.getLogger()
    root.setLevel(CURRENT_LEVEL)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setLevel(CURRENT_LEVEL)
    # Update existing loggers to the new level
    for logger in logging.getLogger().manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(CURRENT_LEVEL)
            for h in logger.handlers:
                h.setLevel(CURRENT_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured with the global level."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(CURRENT_LEVEL)
    return logger


# Configure root logger on import according to ``GABRIEL_LOG_LEVEL``.
set_log_level(CURRENT_LEVEL)

