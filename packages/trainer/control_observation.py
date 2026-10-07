"""Dependency-neutral runtime contract for control observations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ControlObservation(ABC):
    """Runtime interface implemented by an experiment-owned observation."""

    @abstractmethod
    def observation_size(self, node_ids: tuple[str, ...]) -> int:
        """Return the fixed width emitted by :meth:`observe`."""

    @abstractmethod
    def observe(
        self,
        *,
        node_ids: tuple[str, ...],
        chlorine: np.ndarray,
        current_injection: float,
        maximum_injection: float,
        time_of_year_seconds: int,
        raw_action: float = 0.0,
    ) -> np.ndarray:
        """Advance state once and return the current policy observation."""

    def reset(self) -> None:
        """Reset episode-local state for a stateless observation."""

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state for a stateless observation."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore checkpoint state for a stateless observation."""
        if state:
            raise ValueError("stateless control observation state must be empty")


__all__ = ["ControlObservation"]
