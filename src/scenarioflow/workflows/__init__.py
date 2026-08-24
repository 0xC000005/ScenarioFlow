"""Portable reproduction, foundation, and multi-seed workflows."""

from __future__ import annotations

from typing import Any

__all__ = ["CANDIDATE_IDS", "ROLLOUT_SEEDS", "TRAIN_SEEDS", "build_protocol"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import multiseed

        return getattr(multiseed, name)
    raise AttributeError(name)
