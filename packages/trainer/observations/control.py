"""Experiment-owned control observations and trainer-side composition."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..serialization import importable_type_path
from ..control_observation import ControlObservation
from ..simulation import SimulationSample
from ..specs import (
    ControlObservationSpec,
    ControlPolicyObservationSpec,
    FixedPCALatentAssistedControlObservationSpec,
    ForecastAssistedControlObservationSpec,
    LatentAssistedControlObservationSpec,
    ProjectedLatentAssistedControlObservationSpec,
    SharedDynamicsEncoderControlObservationSpec,
    TrainableNextLatentControlObservationSpec,
)
from ..time import calendar_time_seconds, cyclic_time_features_from_time_of_year


class DefaultControlObservation(ControlObservation):
    """The historical chlorine/deficit/recovery control observation."""

    def __init__(
        self,
        *,
        include_time: bool = True,
        recovery_trigger: float = 0.2,
        recovery_target: float = 0.25,
        recovery_clear_steps: int = 6,
        post_recovery_cooldown_steps: int = 0,
        recovery_aggressive_pressure_steps: int = 0,
        recovery_stagnation_window_steps: int = 0,
        recovery_stagnation_min_improvement: float = 0.0,
    ):
        if not isinstance(include_time, bool):
            raise ValueError("include_time must be a boolean")
        if not 0.0 <= recovery_trigger < recovery_target:
            raise ValueError("observation recovery trigger must be below its target")
        if (
            isinstance(recovery_clear_steps, bool)
            or not isinstance(recovery_clear_steps, int)
            or recovery_clear_steps < 1
        ):
            raise ValueError(
                "observation recovery clear steps must be a positive integer"
            )
        for label, value in (
            ("post-recovery cooldown steps", post_recovery_cooldown_steps),
            (
                "recovery aggressive-pressure steps",
                recovery_aggressive_pressure_steps,
            ),
            ("recovery stagnation-window steps", recovery_stagnation_window_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if recovery_aggressive_pressure_steps > 0:
            if recovery_stagnation_window_steps < 1:
                raise ValueError(
                    "aggressive pressure requires a positive stagnation window"
                )
            if not 0.0 < recovery_stagnation_min_improvement <= 1.0:
                raise ValueError(
                    "recovery stagnation minimum improvement must lie in (0, 1]"
                )
        elif recovery_stagnation_min_improvement != 0.0:
            raise ValueError(
                "stagnation improvement requires aggressive-pressure state"
            )

        self.include_time = include_time
        self.recovery_trigger = float(recovery_trigger)
        self.recovery_target = float(recovery_target)
        self.recovery_clear_steps = recovery_clear_steps
        self.post_recovery_cooldown_steps = post_recovery_cooldown_steps
        self.recovery_aggressive_pressure_steps = (
            recovery_aggressive_pressure_steps
        )
        self.recovery_stagnation_window_steps = recovery_stagnation_window_steps
        self.recovery_stagnation_min_improvement = float(
            recovery_stagnation_min_improvement
        )
        self.reset()

    def observation_size(self, node_ids: tuple[str, ...]) -> int:
        size = len(node_ids) + 6
        size += int(self.post_recovery_cooldown_steps > 0)
        size += int(self.recovery_aggressive_pressure_steps > 0)
        size += 6 if self.include_time else 0
        return size

    def reset(self) -> None:
        self._recovery_active = False
        self._recovery_clear_steps = 0
        self._post_recovery_remaining_steps = 0
        self._recovery_aggressive_pressure_remaining_steps = 0
        self._recovery_deficit_history: list[float] = []
        self._has_previous_sample = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "recovery_active": self._recovery_active,
            "recovery_clear_steps": self._recovery_clear_steps,
            "post_recovery_remaining_steps": self._post_recovery_remaining_steps,
            "recovery_aggressive_pressure_remaining_steps": (
                self._recovery_aggressive_pressure_remaining_steps
            ),
            "recovery_deficit_history": list(self._recovery_deficit_history),
            "has_previous_sample": self._has_previous_sample,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        active = bool(state["recovery_active"])
        clear_steps = int(state["recovery_clear_steps"])
        cooldown_remaining = int(state.get("post_recovery_remaining_steps", 0))
        aggressive_remaining = int(
            state.get("recovery_aggressive_pressure_remaining_steps", 0)
        )
        deficit_history = [
            float(value) for value in state.get("recovery_deficit_history", [])
        ]
        if clear_steps < 0 or clear_steps >= self.recovery_clear_steps:
            raise ValueError("observation recovery clear steps are out of range")
        if not active and clear_steps != 0:
            raise ValueError(
                "inactive observation recovery state cannot have a clear streak"
            )
        if not 0 <= cooldown_remaining <= self.post_recovery_cooldown_steps:
            raise ValueError("observation post-recovery cooldown is out of range")
        if active and cooldown_remaining != 0:
            raise ValueError(
                "active recovery cannot also be in post-recovery cooldown"
            )
        if not 0 <= aggressive_remaining <= self.recovery_aggressive_pressure_steps:
            raise ValueError("observation aggressive-pressure state is out of range")
        if not active and aggressive_remaining != 0:
            raise ValueError(
                "inactive observation recovery cannot retain aggressive pressure"
            )
        if (
            len(deficit_history) > self.recovery_stagnation_window_steps + 1
            or not all(
                np.isfinite(value) and 0.0 <= value <= 1.0
                for value in deficit_history
            )
        ):
            raise ValueError("observation recovery-deficit history is invalid")
        if self.recovery_aggressive_pressure_steps == 0 and deficit_history:
            raise ValueError(
                "observation without aggressive pressure cannot retain its history"
            )
        self._recovery_active = active
        self._recovery_clear_steps = clear_steps
        self._post_recovery_remaining_steps = cooldown_remaining
        self._recovery_aggressive_pressure_remaining_steps = aggressive_remaining
        self._recovery_deficit_history = deficit_history
        self._has_previous_sample = bool(state["has_previous_sample"])

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
        del node_ids, raw_action
        chlorine = np.asarray(chlorine, dtype=np.float32)
        if self._has_previous_sample:
            entered, exited = self._advance_recovery_state(chlorine)
            self._advance_aggressive_pressure(
                chlorine, entered=entered, exited=exited
            )
        else:
            self._has_previous_sample = True

        deficit = np.maximum(0.2 - chlorine, 0.0) / 0.2
        recovery_deficit = (
            np.maximum(self.recovery_target - chlorine, 0.0)
            / self.recovery_target
        )
        scalars = [
            float(deficit.mean()),
            float(deficit.max(initial=0.0)),
            current_injection / maximum_injection,
            float(self._recovery_active),
            self._recovery_clear_steps / self.recovery_clear_steps,
            float(recovery_deficit.max(initial=0.0)),
        ]
        if self.post_recovery_cooldown_steps > 0:
            scalars.append(
                self._post_recovery_remaining_steps
                / self.post_recovery_cooldown_steps
            )
        if self.recovery_aggressive_pressure_steps > 0:
            scalars.append(
                self._recovery_aggressive_pressure_remaining_steps
                / self.recovery_aggressive_pressure_steps
            )
        if self.include_time:
            scalars.extend(
                cyclic_time_features_from_time_of_year(time_of_year_seconds)
            )
        return np.concatenate(
            [chlorine, np.asarray(scalars, dtype=np.float32)], dtype=np.float32
        )

    def _advance_recovery_state(
        self, chlorine: np.ndarray
    ) -> tuple[bool, bool]:
        entered = not self._recovery_active and bool(
            np.any(chlorine < self.recovery_trigger)
        )
        exited = False
        if entered:
            self._recovery_active = True
            self._recovery_clear_steps = 0
            self._post_recovery_remaining_steps = 0
        elif not self._recovery_active and self._post_recovery_remaining_steps > 0:
            self._post_recovery_remaining_steps -= 1
        if not self._recovery_active:
            return entered, exited
        if bool(np.all(chlorine > self.recovery_target)):
            self._recovery_clear_steps += 1
            if self._recovery_clear_steps >= self.recovery_clear_steps:
                exited = True
                self._recovery_active = False
                self._recovery_clear_steps = 0
                self._post_recovery_remaining_steps = (
                    self.post_recovery_cooldown_steps
                )
        else:
            self._recovery_clear_steps = 0
        return entered, exited

    def _advance_aggressive_pressure(
        self,
        chlorine: np.ndarray,
        *,
        entered: bool,
        exited: bool,
    ) -> None:
        aggressive_steps = self.recovery_aggressive_pressure_steps
        if aggressive_steps == 0:
            return
        if exited or not self._recovery_active:
            self._recovery_aggressive_pressure_remaining_steps = 0
            self._recovery_deficit_history.clear()
            return
        if entered:
            self._recovery_deficit_history.clear()

        deficit = float(
            np.max(
                np.clip(
                    (self.recovery_target - chlorine) / self.recovery_target,
                    0.0,
                    1.0,
                ),
                initial=0.0,
            )
        )
        self._recovery_deficit_history.append(deficit)
        maximum_samples = self.recovery_stagnation_window_steps + 1
        if len(self._recovery_deficit_history) > maximum_samples:
            del self._recovery_deficit_history[:-maximum_samples]
        ready = len(self._recovery_deficit_history) == maximum_samples
        improvement = (
            max(self._recovery_deficit_history[0] - deficit, 0.0)
            if ready
            else 0.0
        )
        stagnation_gate = (
            float(
                np.clip(
                    1.0
                    - improvement / self.recovery_stagnation_min_improvement,
                    0.0,
                    1.0,
                )
            )
            if ready
            else 0.0
        )
        severe = bool(np.any(chlorine < self.recovery_trigger))
        if severe or stagnation_gate > 0.0:
            self._recovery_aggressive_pressure_remaining_steps = aggressive_steps
        elif self._recovery_aggressive_pressure_remaining_steps > 0:
            self._recovery_aggressive_pressure_remaining_steps -= 1


class Provider(Protocol):
    output_shape: tuple[int, ...]
    node_ids: tuple[str, ...]
    required_history_samples: int
    ready: bool

    def reset(self) -> None: ...

    def observe(self, sample: SimulationSample) -> None: ...

    def predict(self, sample: SimulationSample) -> np.ndarray: ...


_ASSISTED_SPECS = (
    ForecastAssistedControlObservationSpec,
    LatentAssistedControlObservationSpec,
    ProjectedLatentAssistedControlObservationSpec,
    FixedPCALatentAssistedControlObservationSpec,
    SharedDynamicsEncoderControlObservationSpec,
    TrainableNextLatentControlObservationSpec,
)


def base_control_observation_spec(
    spec: ControlPolicyObservationSpec,
) -> ControlObservationSpec:
    return spec.base if isinstance(spec, _ASSISTED_SPECS) else spec


def create_control_observation(spec: ControlObservationSpec) -> ControlObservation:
    observation = spec.factory(**spec.as_dict())
    if not isinstance(observation, ControlObservation):
        raise TypeError(
            f"control observation class {spec.factory!r} returned an incompatible object"
        )
    return observation


def default_control_observation(**parameters: Any) -> ControlObservationSpec:
    """Return the factory specification for the historical base observation."""
    return ControlObservationSpec(DefaultControlObservation, parameters)


class ControlObservationBuilder:
    """Compose one experiment-owned observation with an optional provider."""

    def __init__(
        self,
        spec: ControlPolicyObservationSpec,
        node_ids: tuple[str, ...],
        max_injection: float,
        provider: Provider | None = None,
    ):
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("control observation node_ids must be non-empty and unique")
        if not np.isfinite(max_injection) or max_injection <= 0:
            raise ValueError("control observation maximum injection must be positive")
        self.spec = spec
        self.node_ids = node_ids
        self.max_injection = float(max_injection)
        self.provider = provider
        self.observation = create_control_observation(
            base_control_observation_spec(spec)
        )
        base_size = self.observation.observation_size(node_ids)
        if (
            isinstance(base_size, bool)
            or not isinstance(base_size, int)
            or base_size < 1
        ):
            raise ValueError("control observation size must be a positive integer")
        self.base_size = base_size

        if isinstance(spec, _ASSISTED_SPECS):
            if provider is None:
                raise ValueError("assisted observations require a provider")
            if tuple(getattr(provider, "node_ids", ())) != node_ids:
                raise ValueError("observation provider node order is incompatible")
            output_shape = tuple(provider.output_shape)
            if (
                not output_shape
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in output_shape
                )
            ):
                raise ValueError("observation provider output shape is invalid")
            if isinstance(
                spec,
                (
                    LatentAssistedControlObservationSpec,
                    ProjectedLatentAssistedControlObservationSpec,
                    FixedPCALatentAssistedControlObservationSpec,
                ),
            ) and output_shape != (spec.latent_size,):
                raise ValueError("latent provider shape is incompatible")
            if isinstance(
                spec,
                (
                    SharedDynamicsEncoderControlObservationSpec,
                    TrainableNextLatentControlObservationSpec,
                ),
            ) and output_shape != (
                spec.history_steps * spec.history_feature_size,
            ):
                raise ValueError("history provider shape is incompatible")
        elif provider is not None:
            raise ValueError("unassisted observations cannot have a provider")

    @property
    def shape(self) -> tuple[int, ...]:
        extra = (
            int(np.prod(self.provider.output_shape))
            if self.provider is not None
            else 0
        )
        return (self.base_size + extra,)

    @property
    def required_history_samples(self) -> int:
        if self.provider is None:
            return 1
        value = int(getattr(self.provider, "required_history_samples", 1))
        if value < 1:
            raise ValueError("provider required history samples must be positive")
        return value

    def reset(self) -> None:
        self.observation.reset()
        if self.provider is not None:
            self.provider.reset()

    def state_dict(self) -> dict[str, Any]:
        return self.observation.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.observation.load_state_dict(state)

    def prime(self, sample: SimulationSample) -> None:
        """Observe a causal pre-roll sample without exposing it to the policy."""
        self._base_observation(sample)
        if self.provider is not None:
            observe = getattr(self.provider, "observe", None)
            if observe is None:
                raise TypeError(
                    "history-backed provider does not support observation priming"
                )
            observe(sample)

    def build(self, sample: SimulationSample) -> np.ndarray:
        base = self._base_observation(sample)
        if self.provider is None:
            return base
        provided = np.asarray(self.provider.predict(sample), dtype=np.float32)
        if provided.shape != tuple(self.provider.output_shape):
            raise ValueError("observation provider returned an incompatible shape")
        if not np.all(np.isfinite(provided)):
            raise ValueError("observation provider returned non-finite values")
        return np.concatenate([base, provided.reshape(-1)], dtype=np.float32)

    def _base_observation(self, sample: SimulationSample) -> np.ndarray:
        if sample.node_ids != self.node_ids:
            raise ValueError(
                "simulation node order does not match control observation contract"
            )
        chlorine = np.asarray(sample.chlorine, dtype=np.float32)
        if chlorine.shape != (len(self.node_ids),):
            raise ValueError("simulation chlorine shape is incompatible")
        chlorine = chlorine.copy()
        chlorine.setflags(write=False)
        value = self.observation.observe(
            node_ids=self.node_ids,
            chlorine=chlorine,
            current_injection=float(sample.injection),
            maximum_injection=self.max_injection,
            time_of_year_seconds=int(
                calendar_time_seconds(
                    sample.time_seconds, sample.calendar_start_day
                )
            ),
            raw_action=float(sample.action),
        )
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (self.base_size,):
            raise ValueError(
                "control observation returned an incompatible one-dimensional shape"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("control observation returned non-finite values")
        return result

    def describe(self) -> dict[str, object]:
        base_spec = base_control_observation_spec(self.spec)
        return {
            "factory": importable_type_path(base_spec.factory),
            "shape": list(self.shape),
            "node_ids": list(self.node_ids),
        }


__all__ = [
    "ControlObservation",
    "ControlObservationBuilder",
    "DefaultControlObservation",
    "base_control_observation_spec",
    "create_control_observation",
    "default_control_observation",
]
