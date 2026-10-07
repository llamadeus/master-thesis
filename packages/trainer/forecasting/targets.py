"""Generic named forecast target extraction and physical reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..simulation import Trajectory
from ..specs import ForecastHeadSpec
from .features import (_require_hydraulic_outflow, apply_transform,
                       derive_transported_arsenic, inverse_transform)


TargetProvider = Callable[[Trajectory, ForecastHeadSpec], np.ndarray]
_TARGETS: dict[str, TargetProvider] = {}


def register_target_provider(name: str, provider: TargetProvider, *, replace: bool = False):
    if not name or (name in _TARGETS and not replace): raise ValueError(f"target provider already registered: {name}")
    _TARGETS[name] = provider


def extract_target(trajectory: Trajectory, head: ForecastHeadSpec) -> np.ndarray:
    try: values = np.asarray(_TARGETS[head.target](trajectory, head), dtype=np.float32)
    except KeyError as exc: raise ValueError(f"unknown forecast target provider: {head.target}") from exc
    if values.ndim != 2 or values.shape[0] != len(trajectory.time_seconds):
        raise ValueError(f"target provider {head.target!r} returned an invalid shape")
    return values


@dataclass(frozen=True)
class HeadTarget:
    values: np.ndarray
    previous_level: np.ndarray


def target_window(trajectory: Trajectory, head: ForecastHeadSpec, start: int, end: int) -> HeadTarget:
    physical = extract_target(trajectory, head)
    levels = physical[start:end]
    previous = physical[start - 1]
    represented = levels if head.prediction == "level" else levels - physical[start - 1:end - 1]
    return HeadTarget(apply_transform(represented, head.transform, head.scale).astype(np.float32),
                      previous.astype(np.float32))


def reconstruct_physical(prediction, previous_level, head: ForecastHeadSpec):
    represented = inverse_transform(prediction, head.transform, head.scale)
    if head.prediction == "level": return np.clip(represented, 0.0, None)
    previous = np.asarray(previous_level)
    return np.clip(previous[..., None, :] + np.cumsum(represented, axis=-2), 0.0, None)


def _canonical(name):
    return lambda trajectory, _head: np.asarray(getattr(trajectory, name))


def _transported(trajectory, _head):
    values, _local = derive_transported_arsenic(
        trajectory.arsenic, trajectory.event_strength_by_node,
        _require_hydraulic_outflow(trajectory), trajectory.time_seconds,
        flow_units=trajectory.flow_units or "LPS")
    return values


register_target_provider("chlorine", _canonical("chlorine"))
register_target_provider("pressure", _canonical("pressure"))
register_target_provider("raw-demand", _canonical("demand"))
register_target_provider("raw-arsenic", _canonical("arsenic"))
register_target_provider("transported-arsenic", _transported)
