"""Minimum-triggered upper tracking relief with movement and jerk penalties.

The first twelve scored steps leave alpha state unchanged at zero. Steps 13
through 24 update alpha normally and linearly blend its effect from 1/12 to 1.
The initial unscored observation does not advance this schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from packages.trainer import ControlObservation, RewardInput
from packages.trainer.time import cyclic_time_features_from_time_of_year

DEFAULT_PARAMETERS: dict[str, float] = {
    "target": 0.30,
    "tracking_reference": 0.01,
    "alpha_one_threshold": 0.15,
    "alpha_zero_threshold": 0.20,
    "blend_grace_steps": 4.0,
    "blend_decay_steps": 12.0,
    "warmup_steps": 12.0,
    "blend_steps": 12.0,
    "upper_penalty_min_factor": 0.0001,
    "wasted_movement_reference": 0.01,
    "wasted_movement_weight": 0.1,
    "wasted_movement_gate_full_threshold": 0.35,
    "wasted_movement_gate_zero_threshold": 0.40,
    "action_jerk_reference": 0.05,
    "action_jerk_weight": 0.1,
}


def _validated_number(raw: object, name: str) -> float:
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"v89 {name} is invalid")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"v89 {name} is invalid") from error
    if not np.isfinite(value):
        raise ValueError(f"v89 {name} must be finite")
    return value


def _validated_parameters(parameters: dict[str, float] | None) -> dict[str, float]:
    overrides = {} if parameters is None else dict(parameters)
    unknown = set(overrides) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown v89 parameters: {sorted(unknown)}")
    p = {
        name: _validated_number(overrides.get(name, default), name)
        for name, default in DEFAULT_PARAMETERS.items()
    }
    if p["target"] <= 0.0:
        raise ValueError("v89 target must be positive")
    if any(value <= 0.0 for name, value in p.items() if name.endswith("_reference")):
        raise ValueError("v89 reference scales must be positive")
    if any(value < 0.0 for name, value in p.items() if name.endswith("_weight")):
        raise ValueError("v89 weights must be non-negative")
    if not 0.0 <= p["upper_penalty_min_factor"] <= 1.0:
        raise ValueError("v89 upper penalty minimum factor must lie in [0, 1]")
    if not (
        0.0
        <= p["wasted_movement_gate_full_threshold"]
        < p["wasted_movement_gate_zero_threshold"]
    ):
        raise ValueError(
            "v89 wasted movement gate thresholds must satisfy 0 <= full < zero"
        )
    if not (
        0.0
        <= p["alpha_one_threshold"]
        < p["alpha_zero_threshold"]
        <= p["target"]
    ):
        raise ValueError(
            "v89 alpha thresholds must satisfy 0 <= one < zero <= target"
        )
    for name, minimum in (
        ("blend_grace_steps", 0),
        ("blend_decay_steps", 1),
        ("warmup_steps", 0),
        ("blend_steps", 0),
    ):
        if (
            p[name] < minimum
            or not p[name].is_integer()
            or p[name] >= np.iinfo(np.int64).max
        ):
            raise ValueError(f"v89 {name} must be an integer >= {minimum}")
    if _release_duration(p) >= np.iinfo(np.int64).max:
        raise ValueError("v89 blend memory exceeds the supported range")
    return p


def _release_duration(p: dict[str, float]) -> int:
    return int(p["blend_grace_steps"]) + int(p["blend_decay_steps"])


def _activation_progress(step: int, p: dict[str, float]) -> float:
    if step <= p["warmup_steps"]:
        return 0.0
    if p["blend_steps"] == 0.0:
        return 1.0
    return float(np.clip((step - p["warmup_steps"]) / p["blend_steps"], 0.0, 1.0))


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _raw_alpha(minimum: float, p: dict[str, float]) -> float:
    return _smoothstep(
        (p["alpha_zero_threshold"] - minimum)
        / (p["alpha_zero_threshold"] - p["alpha_one_threshold"])
    )


def _release_remaining(release_step: int, p: dict[str, float]) -> float:
    decay_progress = np.clip(
        (release_step - int(p["blend_grace_steps"]))
        / p["blend_decay_steps"],
        0.0,
        1.0,
    )
    return 1.0 - float(decay_progress)


def _effective_alpha(
    attack_alpha: float, release_step: int, p: dict[str, float]
) -> float:
    return attack_alpha * _smoothstep(_release_remaining(release_step, p))


@dataclass(frozen=True)
class _AlphaState:
    minimum: float
    raw: float
    attack: float
    release: int
    alpha: float
    remaining: float


def _advance_alpha_state(
    chlorine: np.ndarray,
    attack_alpha: float,
    release_step: int,
    p: dict[str, float],
) -> _AlphaState:
    minimum = float(np.min(chlorine))
    raw = _raw_alpha(minimum, p)
    current = _effective_alpha(attack_alpha, release_step, p)
    activated = raw > 0.0 and raw >= current
    duration = _release_duration(p)
    if activated:
        next_attack = raw
        next_release = 0
    elif attack_alpha > 0.0:
        next_attack = attack_alpha
        next_release = min(release_step + 1, duration)
    else:
        next_attack = 0.0
        next_release = duration
    if next_release >= duration:
        next_attack = 0.0
    remaining = _release_remaining(next_release, p)
    return _AlphaState(
        minimum=minimum,
        raw=raw,
        attack=next_attack,
        release=next_release,
        alpha=_effective_alpha(next_attack, next_release, p),
        remaining=remaining,
    )


def _advance_scheduled_alpha_state(
    chlorine: np.ndarray,
    attack_alpha: float,
    release_step: int,
    step: int,
    p: dict[str, float],
) -> tuple[_AlphaState, float]:
    progress = _activation_progress(step, p)
    if step > p["warmup_steps"]:
        return _advance_alpha_state(chlorine, attack_alpha, release_step, p), progress
    minimum = float(np.min(chlorine))
    return (
        _AlphaState(
            minimum=minimum,
            raw=_raw_alpha(minimum, p),
            attack=attack_alpha,
            release=release_step,
            alpha=_effective_alpha(attack_alpha, release_step, p),
            remaining=_release_remaining(release_step, p),
        ),
        progress,
    )


def _validated_node_ids(raw: object) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("v89 node ids are invalid")
    if any(not isinstance(node, str) or not node for node in raw):
        raise ValueError("v89 node ids are invalid")
    if len(set(raw)) != len(raw):
        raise ValueError("v89 node ids must be unique")
    return tuple(raw)


def _check_nodes(
    node_ids: tuple[str, ...], count: int, previous: tuple[str, ...] | None
) -> tuple[str, ...]:
    nodes = _validated_node_ids(node_ids)
    if nodes is None or len(nodes) != count:
        raise ValueError("v89 chlorine and node counts disagree")
    if previous is not None and nodes != previous:
        raise ValueError("v89 node identity or order changed without reset")
    return nodes


def _validated_alpha_state(
    raw_attack: object,
    raw_release: object,
    p: dict[str, float],
) -> tuple[float, int]:
    attack = _validated_number(raw_attack, "attack alpha")
    if not 0.0 <= attack <= 1.0:
        raise ValueError("v89 attack alpha must lie in [0, 1]")
    if isinstance(raw_release, (bool, np.bool_)) or not isinstance(
        raw_release, (int, np.integer)
    ):
        raise ValueError("v89 release step is invalid")
    release = int(raw_release)
    duration = _release_duration(p)
    if not 0 <= release <= duration:
        raise ValueError("v89 release step is invalid")
    if attack == 0.0 and release != duration:
        raise ValueError("v89 inactive alpha state is inconsistent")
    if attack > 0.0 and release >= duration:
        raise ValueError("v89 active alpha state is inconsistent")
    return attack, release


def _validated_dynamic_state(
    state: dict[str, Any], p: dict[str, float]
) -> tuple[tuple[str, ...] | None, float, int, int]:
    nodes = _validated_node_ids(state["node_ids"])
    attack, release = _validated_alpha_state(
        state["attack_alpha"], state["release_step"], p
    )
    step = state["episode_step"]
    if (
        isinstance(step, (bool, np.bool_))
        or not isinstance(step, (int, np.integer))
        or step < 0
    ):
        raise ValueError("v89 episode step must be a non-negative integer")
    if (
        (nodes is None and (attack != 0.0 or int(step) != 0))
        or (
            step <= p["warmup_steps"]
            and (attack != 0.0 or release != _release_duration(p))
        )
    ):
        raise ValueError("v89 alpha state is inconsistent")
    return nodes, attack, release, int(step)


class V89Observation(ControlObservation):
    """Chlorine, control inputs, and alpha state (N+12)."""

    def __init__(
        self, parameters: dict[str, float] | None = None, **parameter_overrides: float
    ) -> None:
        if parameters is not None and parameter_overrides:
            raise ValueError("v89 observation parameters were supplied twice")
        self.p = _validated_parameters(
            parameters if parameters is not None else parameter_overrides
        )
        self.reset()

    def observation_size(self, node_ids: tuple[str, ...]) -> int:
        nodes = _validated_node_ids(node_ids)
        if nodes is None:
            raise ValueError("v89 node ids are invalid")
        return len(nodes) + 12

    def reset(self) -> None:
        self._previous_injection: float | None = None
        self._node_ids: tuple[str, ...] | None = None
        self._attack_alpha = 0.0
        self._release_step = _release_duration(self.p)
        self._episode_step = 0

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
        chlorine = np.asarray(chlorine, dtype=float).reshape(-1)
        nodes = _check_nodes(node_ids, chlorine.size, self._node_ids)
        if not np.all(np.isfinite(chlorine)):
            raise ValueError("v89 observation chlorine must be finite")
        if not np.isfinite(maximum_injection) or maximum_injection <= 0.0:
            raise ValueError("v89 observation maximum injection must be positive")
        if (
            not np.isfinite(current_injection)
            or not 0.0 <= current_injection <= maximum_injection
        ):
            raise ValueError(
                "v89 observation current injection must lie within its bounds"
            )
        if (
            self._previous_injection is not None
            and self._previous_injection > maximum_injection
        ):
            raise ValueError("v89 previous injection exceeds maximum injection")
        step = self._episode_step + int(self._previous_injection is not None)
        preview, activation = _advance_scheduled_alpha_state(
            V89Reward._clean_chlorine(chlorine),
            self._attack_alpha,
            self._release_step,
            step,
            self.p,
        )
        delta = (
            0.0
            if self._previous_injection is None
            else current_injection - self._previous_injection
        )
        observation = np.concatenate(
            [
                chlorine,
                [current_injection, delta],
                cyclic_time_features_from_time_of_year(
                    time_of_year_seconds
                ).reshape(-1),
                [V89Reward._clean_action(raw_action)],
                [activation * preview.alpha, preview.remaining, activation],
            ]
        ).astype(np.float32)
        if self._previous_injection is not None:
            self._attack_alpha = preview.attack
            self._release_step = preview.release
            self._episode_step = step
        self._previous_injection = float(current_injection)
        self._node_ids = nodes
        return observation

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous_injection": self._previous_injection,
            "node_ids": None if self._node_ids is None else list(self._node_ids),
            "attack_alpha": self._attack_alpha,
            "release_step": self._release_step,
            "episode_step": self._episode_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {
            "previous_injection",
            "node_ids",
            "attack_alpha",
            "release_step",
            "episode_step",
        }:
            raise ValueError("invalid v89 observation state")
        nodes, attack, release, step = _validated_dynamic_state(state, self.p)
        raw = state["previous_injection"]
        previous = None if raw is None else _validated_number(raw, "previous injection")
        if previous is not None and previous < 0.0:
            raise ValueError("v89 previous injection must be non-negative")
        if (previous is None) != (nodes is None):
            raise ValueError("v89 observation state is inconsistent")
        if previous is None and attack != 0.0:
            raise ValueError("v89 observation state is inconsistent")
        self._previous_injection = previous
        self._node_ids = nodes
        self._attack_alpha = attack
        self._release_step = release
        self._episode_step = step


class V89Reward:
    """Quadratic node tracking with movement and action jerk penalties."""

    reward_id = "v89"

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        self.p = _validated_parameters(parameters)
        self.reset()

    def reset(self) -> None:
        self._previous_chlorine: np.ndarray | None = None
        self._previous_action: float | None = None
        self._node_ids: tuple[str, ...] | None = None
        self._attack_alpha = 0.0
        self._release_step = _release_duration(self.p)
        self._episode_step = 0

    @staticmethod
    def _clean_chlorine(values: object) -> np.ndarray:
        chlorine = np.asarray(values, dtype=float).reshape(-1)
        return np.clip(
            np.nan_to_num(chlorine, nan=0.0, posinf=1_000_000.0, neginf=0.0),
            0.0,
            1_000_000.0,
        )

    @staticmethod
    def _clean_action(value: object) -> float:
        return float(
            np.clip(
                np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0),
                -1.0,
                1.0,
            )
        )

    def compute(self, state: RewardInput) -> tuple[float, dict[str, Any]]:
        p = self.p
        chlorine = self._clean_chlorine(state.nodes_quality)
        nodes = _check_nodes(state.node_ids, chlorine.size, self._node_ids)
        action = self._clean_action(state.raw_action)
        transition_available = self._previous_chlorine is not None
        action_delta = (
            0.0 if self._previous_action is None else action - self._previous_action
        )

        step = self._episode_step + 1
        alpha_state, activation = _advance_scheduled_alpha_state(
            chlorine, self._attack_alpha, self._release_step, step, p
        )
        alpha = activation * alpha_state.alpha
        lower_costs = (
            np.maximum(p["target"] - chlorine, 0.0) / p["tracking_reference"]
        ) ** 2
        upper_costs = (
            np.maximum(chlorine - p["target"], 0.0) / p["tracking_reference"]
        ) ** 2
        upper_penalty_factor = (
            (1.0 - alpha) + p["upper_penalty_min_factor"] * alpha
        )
        lower_tracking_penalty = float(np.mean(lower_costs))
        upper_tracking_penalty = upper_penalty_factor * float(np.mean(upper_costs))
        tracking_costs = lower_costs + upper_penalty_factor * upper_costs
        tracking_penalty = lower_tracking_penalty + upper_tracking_penalty
        if transition_available:
            movement = np.abs(chlorine - self._previous_chlorine)
            useful = np.maximum(
                np.abs(self._previous_chlorine - p["target"])
                - np.abs(chlorine - p["target"]),
                0.0,
            )
            wasted = np.maximum(movement - useful, 0.0)
            gate_position = np.clip(
                (
                    self._previous_chlorine
                    - p["wasted_movement_gate_full_threshold"]
                )
                / (
                    p["wasted_movement_gate_zero_threshold"]
                    - p["wasted_movement_gate_full_threshold"]
                ),
                0.0,
                1.0,
            )
            wasted_gates = 1.0 - gate_position**2 * (3.0 - 2.0 * gate_position)
        else:
            movement = np.zeros_like(chlorine)
            useful = np.zeros_like(chlorine)
            wasted = np.zeros_like(chlorine)
            wasted_gates = np.zeros_like(chlorine)
        wasted_costs = (wasted / p["wasted_movement_reference"]) ** 2
        all_node_wasted_cost = float(np.mean(wasted_costs))
        wasted_gate_sum = float(np.sum(wasted_gates))
        filtered_wasted_cost = (
            float(np.dot(wasted_gates, wasted_costs) / wasted_gate_sum)
            if wasted_gate_sum > 0.0
            else 0.0
        )
        blended_wasted_cost = (
            (1.0 - alpha) * all_node_wasted_cost
            + alpha * filtered_wasted_cost
        )
        wasted_penalty = p["wasted_movement_weight"] * blended_wasted_cost
        jerk = (
            p["action_jerk_weight"]
            * (action_delta / p["action_jerk_reference"]) ** 2
        )
        total = tracking_penalty + wasted_penalty + jerk
        reward = -float(total)

        self._previous_chlorine = chlorine.copy()
        self._previous_action = action
        self._node_ids = nodes
        self._attack_alpha = alpha_state.attack
        self._release_step = alpha_state.release
        self._episode_step = step
        return reward, {
            "reward_id": self.reward_id,
            "minimum_chlorine": alpha_state.minimum,
            "alpha_raw": alpha_state.raw,
            "attack_alpha": alpha_state.attack,
            "alpha": alpha,
            "activation_progress": activation,
            "episode_step": step,
            "release_step": alpha_state.release,
            "release_remaining": alpha_state.remaining,
            "upper_penalty_factor": upper_penalty_factor,
            "lower_tracking_cost_by_node": lower_costs.tolist(),
            "upper_tracking_cost_by_node": upper_costs.tolist(),
            "tracking_cost_by_node": tracking_costs.tolist(),
            "lower_tracking_penalty": lower_tracking_penalty,
            "upper_tracking_penalty": upper_tracking_penalty,
            "tracking_penalty": tracking_penalty,
            "transition_available": float(transition_available),
            "movement_by_node": movement.tolist(),
            "useful_movement_by_node": useful.tolist(),
            "wasted_movement_by_node": wasted.tolist(),
            "wasted_movement_cost_by_node": wasted_costs.tolist(),
            "wasted_movement_gate_by_node": wasted_gates.tolist(),
            "wasted_movement_gate_sum": wasted_gate_sum,
            "all_node_wasted_movement_cost": all_node_wasted_cost,
            "filtered_wasted_movement_cost": filtered_wasted_cost,
            "blended_wasted_movement_cost": blended_wasted_cost,
            "mean_movement": float(np.mean(movement)),
            "mean_useful_movement": float(np.mean(useful)),
            "mean_wasted_movement": float(np.mean(wasted)),
            "wasted_movement_penalty": wasted_penalty,
            "action_delta": float(action_delta),
            "action_jerk_penalty": jerk,
            "total_penalty": float(total),
            "total_reward": reward,
            "mean_chlorine_concentration": float(np.mean(chlorine)),
            "median_chlorine_concentration": float(np.median(chlorine)),
            "spatial_chlorine_std": float(np.std(chlorine)),
            "min_chlorine_concentration": float(np.min(chlorine)),
            "max_chlorine_concentration": float(np.max(chlorine)),
            "raw_action": action,
            "current_injection": float(state.current_injection),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous_chlorine": (
                None
                if self._previous_chlorine is None
                else self._previous_chlorine.tolist()
            ),
            "previous_action": self._previous_action,
            "node_ids": None if self._node_ids is None else list(self._node_ids),
            "attack_alpha": self._attack_alpha,
            "release_step": self._release_step,
            "episode_step": self._episode_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "previous_chlorine",
            "previous_action",
            "node_ids",
            "attack_alpha",
            "release_step",
            "episode_step",
        }
        if set(state) != expected:
            raise ValueError("invalid v89 reward state")
        nodes, attack, release, step = _validated_dynamic_state(state, self.p)
        raw_chlorine, raw_action = state["previous_chlorine"], state["previous_action"]
        if (
            (raw_chlorine is None) != (raw_action is None)
            or (raw_chlorine is None) != (nodes is None)
            or (raw_chlorine is None) != (step == 0)
        ):
            raise ValueError("v89 transition references and node ids must be paired")
        if raw_chlorine is None and attack != 0.0:
            raise ValueError("v89 reward alpha state requires transition references")
        chlorine, action = None, None
        if raw_chlorine is not None:
            chlorine = np.asarray(raw_chlorine, dtype=float)
            if (
                chlorine.ndim != 1
                or chlorine.size != len(nodes)
                or not np.all(np.isfinite(chlorine))
                or np.any((chlorine < 0.0) | (chlorine > 1_000_000.0))
            ):
                raise ValueError("v89 previous chlorine is invalid")
            chlorine = chlorine.copy()
            action = _validated_number(raw_action, "previous action")
            if not -1.0 <= action <= 1.0:
                raise ValueError("v89 previous action must lie in [-1, 1]")
        self._previous_chlorine, self._previous_action = chlorine, action
        self._node_ids = nodes
        self._attack_alpha = attack
        self._release_step = release
        self._episode_step = step


__all__ = ["DEFAULT_PARAMETERS", "V89Observation", "V89Reward"]
