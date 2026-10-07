"""Public reward contracts and runtime construction for control training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..specs import RewardSpec


@dataclass(frozen=True)
class RewardInput:
    """Complete public control state supplied to an experiment-owned reward."""

    nodes_quality: NDArray[np.floating]
    contaminant_values: NDArray[np.floating]
    pressure: NDArray[np.floating]
    demand: NDArray[np.floating]
    flow: NDArray[np.floating]
    current_injection: float
    raw_action: float
    step: int
    current_time: int
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    event_active: bool
    event_strength_by_node: NDArray[np.floating]


@runtime_checkable
class Reward(Protocol):
    def reset(self) -> None: ...
    def compute(self, state: RewardInput) -> tuple[float, dict[str, Any]]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def create_reward(spec: RewardSpec) -> Reward:
    """Construct a fresh experiment-owned reward runtime."""

    reward = spec.factory(spec.as_dict())
    if not isinstance(reward, Reward):
        raise TypeError(f"reward class {spec.factory!r} returned an incompatible object")
    return reward


__all__ = ["Reward", "RewardInput", "create_reward"]
