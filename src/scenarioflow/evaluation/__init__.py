"""Published validation evaluation and immutable artifact contracts."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BASELINES",
    "METRIC_PATHS",
    "PUBLISHED_VALIDATION_PROTOCOL",
    "TABLE_THRESHOLD_RULES",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import current_panel

        return getattr(current_panel, name)
    raise AttributeError(name)
