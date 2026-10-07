"""Experiment-owned control observations shared by training configurations."""

from __future__ import annotations

import numpy as np

from packages.trainer.observations import DefaultControlObservation
from packages.trainer.specs import ControlObservationSpec


class V64Observation(DefaultControlObservation):
    """Default observation plus the normalized post-recovery cooldown."""

    def __init__(
        self,
        *,
        include_time: bool = True,
        recovery_trigger: float = 0.2,
        recovery_target: float = 0.25,
        recovery_clear_steps: int = 6,
        post_recovery_cooldown_steps: int = 12,
    ):
        super().__init__(
            include_time=include_time,
            recovery_trigger=recovery_trigger,
            recovery_target=recovery_target,
            recovery_clear_steps=recovery_clear_steps,
            post_recovery_cooldown_steps=post_recovery_cooldown_steps,
        )


class V70Observation(DefaultControlObservation):
    """V70 recovery state, cooldown, pressure timer, and calendar features."""

    def __init__(
        self,
        *,
        include_time: bool = True,
        recovery_trigger: float = 0.2,
        recovery_target: float = 0.25,
        recovery_clear_steps: int = 6,
        post_recovery_cooldown_steps: int = 12,
        recovery_aggressive_pressure_steps: int = 48,
        recovery_stagnation_window_steps: int = 6,
        recovery_stagnation_min_improvement: float = 0.05,
    ):
        super().__init__(
            include_time=include_time,
            recovery_trigger=recovery_trigger,
            recovery_target=recovery_target,
            recovery_clear_steps=recovery_clear_steps,
            post_recovery_cooldown_steps=post_recovery_cooldown_steps,
            recovery_aggressive_pressure_steps=recovery_aggressive_pressure_steps,
            recovery_stagnation_window_steps=recovery_stagnation_window_steps,
            recovery_stagnation_min_improvement=(
                recovery_stagnation_min_improvement
            ),
        )


class V70RecoveryTimeObservation(V70Observation):
    """V70 recovery/time features without direct chlorine or injection values."""

    def observation_size(self, node_ids: tuple[str, ...]) -> int:
        return super().observation_size(node_ids) - len(node_ids) - 3

    def observe(self, **inputs) -> np.ndarray:
        observation = super().observe(**inputs)
        chlorine_end = len(inputs["node_ids"])
        injection_index = chlorine_end + 2
        return np.asarray(observation[injection_index + 1 :], dtype=np.float32)


class V70DeficitRecoveryTimeObservation(V70Observation):
    """V70 deficit/recovery/time features without raw chlorine or injection."""

    def observation_size(self, node_ids: tuple[str, ...]) -> int:
        return super().observation_size(node_ids) - len(node_ids) - 1

    def observe(self, **inputs) -> np.ndarray:
        observation = super().observe(**inputs)
        chlorine_end = len(inputs["node_ids"])
        injection_index = chlorine_end + 2
        return np.concatenate(
            (
                observation[chlorine_end:injection_index],
                observation[injection_index + 1 :],
            ),
            dtype=np.float32,
        )


V64_OBSERVATION = ControlObservationSpec(V64Observation)
V70_OBSERVATION = ControlObservationSpec(V70Observation)
V70_RECOVERY_TIME_OBSERVATION = ControlObservationSpec(
    V70RecoveryTimeObservation
)
V70_DEFICIT_RECOVERY_TIME_OBSERVATION = ControlObservationSpec(
    V70DeficitRecoveryTimeObservation
)


__all__ = [
    "V64Observation",
    "V64_OBSERVATION",
    "V70Observation",
    "V70DeficitRecoveryTimeObservation",
    "V70RecoveryTimeObservation",
    "V70_DEFICIT_RECOVERY_TIME_OBSERVATION",
    "V70_OBSERVATION",
    "V70_RECOVERY_TIME_OBSERVATION",
]
