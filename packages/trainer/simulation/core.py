"""Simulator-neutral state and trajectory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class SimulationSample:
    time_seconds: int
    calendar_start_day: int
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    pressure: FloatArray
    demand: FloatArray
    flow: FloatArray
    chlorine: FloatArray
    arsenic: FloatArray
    injection: float
    action: float
    event_active: bool = False
    event_strength_by_node: FloatArray | None = None
    hydraulic_outflow: FloatArray | None = None


@dataclass(frozen=True)
class Trajectory:
    network: str
    scenario_id: str
    calendar_start_day: int
    chlorine_profile_hash: str
    arsenic_schedule_hash: str
    arsenic_schedule_seed: int
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    time_seconds: FloatArray
    pressure: FloatArray
    demand: FloatArray
    flow: FloatArray
    chlorine: FloatArray
    arsenic: FloatArray
    injection: FloatArray
    actions: FloatArray
    event_active: NDArray[np.bool_]
    event_strength_by_node: FloatArray
    hydraulic_outflow: FloatArray | None = None
    simulator_backend: str = "unknown"
    simulator_config_hash: str = ""
    flow_units: str | None = None
    reproducibility: tuple[tuple[str, Any], ...] = ()

    @property
    def raw_demand(self):
        return self.demand

    @property
    def raw_arsenic(self):
        return self.arsenic

    @property
    def applied_chlorine_injection(self):
        return self.injection


class SimulationSession(Protocol):
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]

    def reset(self, arsenic_events: tuple[object, ...]) -> SimulationSample: ...
    def step(self, injection: float, action: float) -> tuple[SimulationSample, bool]: ...
    def close(self) -> None: ...


class SimulationBackend(Protocol):
    def create_session(self, scenario_path: str, *, hydraulic_timestep_seconds: int,
                       calendar_start_day: int,
                       chlorine_source_ids: tuple[str, ...],
                       monitored_node_ids: tuple[str, ...]) -> SimulationSession: ...
