"""Shared interface for final joint-panel baseline samplers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class JointBaseline(Protocol):
    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **kwargs: Any
    ) -> np.ndarray: ...
