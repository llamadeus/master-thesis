"""Exact V70 policy-observation replay from cached simulator trajectories."""

from __future__ import annotations

import numpy as np

from packages.trainer.constants import V70_MAX_CHLORINE_INJECTION
from packages.trainer.observations import ControlObservationBuilder
from packages.trainer.specs import ControlObservationSpec
from packages.trainer.simulation import SimulationSample, Trajectory
from training.observations import V70_OBSERVATION


HISTORY_STEPS = 48


def simulation_sample_from_trajectory(
    trajectory: Trajectory, index: int
) -> SimulationSample:
    """Recreate the simulator sample consumed by the observation builder."""
    if index < 0 or index >= len(trajectory.time_seconds):
        raise ValueError("sample index is outside the trajectory")
    hydraulic_outflow = (
        None
        if trajectory.hydraulic_outflow is None
        else np.asarray(trajectory.hydraulic_outflow[index])
    )
    return SimulationSample(
        time_seconds=int(trajectory.time_seconds[index]),
        calendar_start_day=int(trajectory.calendar_start_day),
        node_ids=trajectory.node_ids,
        link_ids=trajectory.link_ids,
        pressure=np.asarray(trajectory.pressure[index]),
        demand=np.asarray(trajectory.demand[index]),
        flow=np.asarray(trajectory.flow[index]),
        chlorine=np.asarray(trajectory.chlorine[index]),
        arsenic=np.asarray(trajectory.arsenic[index]),
        injection=float(trajectory.injection[index]),
        action=float(trajectory.actions[index]),
        event_active=bool(trajectory.event_active[index]),
        event_strength_by_node=np.asarray(
            trajectory.event_strength_by_node[index]
        ),
        hydraulic_outflow=hydraulic_outflow,
    )


def replay_policy_observations(
    trajectory: Trajectory,
    *,
    through_index: int | None = None,
    observation_spec: ControlObservationSpec = V70_OBSERVATION,
    maximum_injection: float = V70_MAX_CHLORINE_INJECTION,
) -> np.ndarray:
    """Replay V70 recurrent observations from reset through an inclusive row."""
    final_index = (
        len(trajectory.time_seconds) - 1
        if through_index is None
        else int(through_index)
    )
    if final_index < 0 or final_index >= len(trajectory.time_seconds):
        raise ValueError("policy observation replay endpoint is outside the trajectory")
    builder = ControlObservationBuilder(
        observation_spec,
        trajectory.node_ids,
        maximum_injection,
    )
    observation_size = builder.shape[0]
    builder.reset()
    observations = np.empty(
        (final_index + 1, observation_size), dtype=np.float32
    )
    for index in range(final_index + 1):
        observations[index] = builder.build(
            simulation_sample_from_trajectory(trajectory, index)
        )
    return observations


def policy_observation_history_at(
    observations: np.ndarray, query_index: int
) -> np.ndarray:
    """Return the 48 observations ending at the pre-action query state."""
    replayed = np.asarray(observations, dtype=np.float32)
    if replayed.ndim != 2 or replayed.shape[1] <= 0:
        raise ValueError(
            "replayed policy observations must have shape (row, feature)"
        )
    start = int(query_index) - HISTORY_STEPS + 1
    if start < 0 or query_index >= replayed.shape[0]:
        raise ValueError("query cannot provide a complete policy observation history")
    history = replayed[start : query_index + 1]
    if history.shape != (HISTORY_STEPS, replayed.shape[1]):
        raise RuntimeError("policy observation history construction lost alignment")
    return history.copy()
