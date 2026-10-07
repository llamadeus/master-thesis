#!/usr/bin/env python3
"""Screen chlorine action timing at the lowest source-outflow starts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from time import monotonic
from typing import Any, Sequence
from zipfile import BadZipFile

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 9
HOURS_PER_STEP = 0.5
STEPS_PER_DAY = 48
SIMULATION_STEPS = 25 * STEPS_PER_DAY
FIRST_CORE_DAY = 2
LAST_CORE_DAY = 22
SOURCE_OUTFLOW_HOURS = 24
RESPONSE_HOURS = 48
BATCH_SEPARATION_HOURS = 72
SCENARIOS_PER_COMBINATION = 10
SCENARIO_IDS = tuple(f"scenario-{index}" for index in range(80))
NODE_IDS = tuple(str(index) for index in range(2, 33))
BASELINE_MG_MIN = 100_000.0
FIRST_ACTION_DELTA_MG_MIN = 25_000.0
MAX_INJECTION_MG_MIN = 500_000.0
MAX_STEP_DELTA_MG_MIN = 50_000.0
ABSOLUTE_THRESHOLD_MG_L = 1e-4
PRE_ACTION_MATCH_TOLERANCE_MG_L = 1e-4
HISTORY_WINDOWS_HOURS = range(12, 37, 2)
HORIZON_WINDOWS_HOURS = range(4, 17, 2)
GRACEFUL_STOP_SECONDS = 30.0
_STOP_EVENT: Any = None
_PROGRESS_PIPE: Any = None
_PROGRESS_CALLBACK: Any = None


class _RunInterrupted(Exception):
    """A requested stop, distinct from a failed diagnostic case."""


class _StopState:
    def __init__(self, event: Any) -> None:
        self.event = event
        self.requests = 0
        self.signum: int | None = None
        self.requested_at: float | None = None
        self.defer_interrupt = False

    @property
    def requested(self) -> bool:
        return self.requests > 0

    def request(self, signum: int, _frame: Any) -> None:
        self.requests += 1
        if self.signum is None:
            self.signum = signum
            self.requested_at = monotonic()
            self.event.set()
        if not self.defer_interrupt:
            raise _RunInterrupted


def _check_stop() -> None:
    if _STOP_EVENT is not None and _STOP_EVENT.is_set():
        raise _RunInterrupted


def _report_status(label: str) -> None:
    if _PROGRESS_PIPE is not None:
        _PROGRESS_PIPE.send(("status", label))
    elif _PROGRESS_CALLBACK is not None:
        _PROGRESS_CALLBACK("status", label)


def _report_progress(label: str = "") -> None:
    if _PROGRESS_PIPE is not None:
        _PROGRESS_PIPE.send(("progress", label))
    elif _PROGRESS_CALLBACK is not None:
        _PROGRESS_CALLBACK("progress", label)


@dataclass(frozen=True)
class TimeSlot:
    weekday_phase: int
    clock_hour: int
    candidate_steps: tuple[int, int, int]


@dataclass(frozen=True)
class Trial:
    scenario_id: str
    weekday_phase: int
    clock_hour: int
    day: int
    start_step: int
    selection_rank: int
    source_outflow_24h_liters: float
    hydraulic_cache_identity: str


def build_time_slots(start_interval_hours: int = 1) -> tuple[TimeSlot, ...]:
    """Give each weekday and sampled clock hour its three core-week starts."""
    if isinstance(start_interval_hours, bool) or not isinstance(start_interval_hours, int):
        raise ValueError("start interval must be a positive integer number of hours")
    if not 1 <= start_interval_hours <= 24:
        raise ValueError("start interval must be between 1 and 24 hours")
    slots = []
    for phase in range(7):
        days = tuple(day for day in range(FIRST_CORE_DAY, LAST_CORE_DAY + 1)
                     if day % 7 == phase)
        if len(days) != 3:
            raise RuntimeError("the 21 core days must contain three complete weeks")
        for hour in range(0, 24, start_interval_hours):
            starts = tuple(day * STEPS_PER_DAY + int(hour / HOURS_PER_STEP)
                           for day in days)
            if starts[-1] + int(RESPONSE_HOURS / HOURS_PER_STEP) > SIMULATION_STEPS:
                raise RuntimeError("a trial lacks a full 48-hour observation")
            slots.append(TimeSlot(phase, hour, starts))
    return tuple(slots)


def select_trials(
    slots: Sequence[TimeSlot],
    source_outflows: dict[str, tuple[np.ndarray, str]],
    *,
    top_k: int = SCENARIOS_PER_COMBINATION,
) -> tuple[Trial, ...]:
    """Rank each scenario by its smallest 24-hour source volume in three weeks."""
    if top_k < 1 or len(source_outflows) < top_k:
        raise ValueError("not enough hydraulic scenarios for the requested top-k")
    width = int(SOURCE_OUTFLOW_HOURS / HOURS_PER_STEP)
    if not slots:
        raise ValueError("no weekday-clock combinations selected")
    selected = []
    for slot in slots:
        ranking = []
        for scenario_id, (outflow, identity) in source_outflows.items():
            values = np.asarray(outflow, dtype=np.float64)
            if values.shape != (SIMULATION_STEPS + 1,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid source outflow grid for {scenario_id}")
            if np.any(values < 0):
                raise ValueError(f"negative source outflow for {scenario_id}")
            volumes = [float(np.sum(values[start:start + width]) * 30.0)
                       for start in slot.candidate_steps]
            choice = min(range(3), key=lambda index: (volumes[index], slot.candidate_steps[index]))
            ranking.append((volumes[choice], scenario_id, slot.candidate_steps[choice], identity))
        ranking.sort(key=lambda item: (item[0], int(item[1].removeprefix("scenario-"))))
        for rank, (volume, scenario_id, start, identity) in enumerate(ranking[:top_k], 1):
            selected.append(Trial(
                scenario_id=scenario_id,
                weekday_phase=slot.weekday_phase,
                clock_hour=slot.clock_hour,
                day=start // STEPS_PER_DAY,
                start_step=start,
                selection_rank=rank,
                source_outflow_24h_liters=volume,
                hydraulic_cache_identity=identity,
            ))
    return tuple(selected)


def paired_rates(start: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a constant reference and one otherwise identical first action."""
    if not FIRST_CORE_DAY * STEPS_PER_DAY <= start <= (
        LAST_CORE_DAY * STEPS_PER_DAY + int(23 / HOURS_PER_STEP)
    ):
        raise ValueError("trial start lies outside the 21 core days")
    if start + int(RESPONSE_HOURS / HOURS_PER_STEP) > SIMULATION_STEPS:
        raise ValueError("trial has less than 48 hours of follow-up")
    reference = np.full(SIMULATION_STEPS, BASELINE_MG_MIN)
    intervention = reference.copy()
    intervention[start] += FIRST_ACTION_DELTA_MG_MIN
    if np.any(intervention > MAX_INJECTION_MG_MIN):
        raise ValueError("injection outside actuator bounds")
    # The immediate 100,000 mg/min initial baseline is intentional. Later
    # transitions, including both sides of the pulse, retain the slew limit.
    if np.any(np.abs(np.diff(intervention)) > MAX_STEP_DELTA_MG_MIN):
        raise ValueError("injection change exceeds actuator limit after initialization")
    return reference, intervention


def batch_trials(trials: Sequence[Trial]) -> tuple[tuple[Trial, ...], ...]:
    """Use the fewest runs with 72 hours between pulses in each run."""
    if not trials:
        return ()
    scenario_id = trials[0].scenario_id
    if any(trial.scenario_id != scenario_id for trial in trials):
        raise ValueError("a batch cannot mix hydraulic scenarios")
    starts = [trial.start_step for trial in trials]
    if len(starts) != len(set(starts)):
        raise ValueError("a scenario cannot have two actions at one time")
    separation_steps = int(BATCH_SEPARATION_HOURS / HOURS_PER_STEP)
    batches: list[list[Trial]] = []
    last_starts: list[int] = []
    for trial in sorted(trials, key=lambda item: item.start_step):
        available = [index for index, last in enumerate(last_starts)
                     if trial.start_step - last >= separation_steps]
        if available:
            index = max(available, key=lambda candidate: last_starts[candidate])
            batches[index].append(trial)
            last_starts[index] = trial.start_step
        else:
            batches.append([trial])
            last_starts.append(trial.start_step)
    return tuple(tuple(batch) for batch in batches)


def analyze_response(response: Sequence[float]) -> dict[str, Any]:
    """Measure positive paired excess above a fixed numerical threshold."""
    values = np.asarray(response, dtype=np.float64)
    if (values.shape != (int(RESPONSE_HOURS / HOURS_PER_STEP) + 1,)
            or not np.all(np.isfinite(values))):
        raise ValueError("response must contain finite half-hour samples through 48 hours")
    positive = np.maximum(values, 0.0)
    above = positive > ABSOLUTE_THRESHOLD_MG_L
    responsive = bool(np.any(above[1:]))
    measurable = np.where(above, positive, 0.0)
    measurable[0] = 0.0
    total = float(np.sum(measurable))
    peak_index = int(np.argmax(positive)) if responsive else None
    detectable = np.flatnonzero(above[1:]) + 1
    t90_index = int(np.searchsorted(np.cumsum(measurable), 0.9 * total))
    result: dict[str, Any] = {
        "responsive": responsive,
        "tail_censored": bool(responsive and above[-1]),
        "threshold_mg_l": ABSOLUTE_THRESHOLD_MG_L,
        "peak_delta_mg_l": float(np.max(positive)),
        "onset_hours": float(detectable[0] * HOURS_PER_STEP) if responsive else None,
        "peak_hours": float(peak_index * HOURS_PER_STEP) if responsive else None,
        "t90_hours": float(t90_index * HOURS_PER_STEP) if responsive else None,
        "last_detectable_hours": float(detectable[-1] * HOURS_PER_STEP)
        if responsive else None,
        "visible_duration_hours": float(len(detectable) * HOURS_PER_STEP),
    }
    for hours in HORIZON_WINDOWS_HOURS:
        endpoint = int(hours / HOURS_PER_STEP)
        result[f"fraction_{hours}h"] = (
            float(np.sum(measurable[1:endpoint + 1]) / total) if responsive else None
        )
        result[f"peak_reached_{hours}h"] = (
            bool(result["peak_hours"] <= hours) if responsive else None
        )
    return result


def source_outflow_liters_per_minute(
    flows: np.ndarray, directions: Sequence[int], liters_per_minute_per_unit: float
) -> np.ndarray:
    """Sum flow leaving the source over its incident, oriented links."""
    values = np.asarray(flows, dtype=np.float64)
    signs = np.asarray(directions, dtype=np.int8)
    if (values.ndim != 2 or signs.shape != (values.shape[1],)
            or not np.all(np.isin(signs, (-1, 1)))
            or not np.all(np.isfinite(values))):
        raise ValueError("invalid source-link flow readings")
    return np.sum(np.maximum(values * signs, 0.0), axis=1) * liters_per_minute_per_unit


def _source_outflow(scenario: Any, cache_root: Path) -> tuple[np.ndarray, str]:
    """Read the reservoir's outward link flow directly from cached hydraulics."""
    from packages.trainer.simulation.epyt import EpytFlowBackend
    from packages.trainer.simulation.hydraulic_cache import HydraulicCache
    from packages.trainer.simulation.hydraulics import \
        FLOW_TO_LITERS_PER_MINUTE

    if len(scenario.chlorine_source_ids) != 1:
        raise ValueError(f"{scenario.scenario_id} needs exactly one chlorine source")
    source = scenario.chlorine_source_ids[0]
    backend = EpytFlowBackend(hydraulic_cache_root=cache_root)
    reference = HydraulicCache(cache_root).reference(scenario.path)
    scada = reference.load()
    if scada is None:
        backend.prepare_hydraulics(scenario.path, cache_root=cache_root)
        scada = reference.load()
    if scada is None:
        raise RuntimeError(f"could not load hydraulics for {scenario.scenario_id}")
    topology = scada.network_topo
    endpoints = {str(link): (str(a), str(b))
                 for link, (a, b) in topology.get_all_links()}
    links = [(link, 1 if a == source else -1)
             for link, (a, b) in endpoints.items() if source in (a, b)]
    if not links:
        raise ValueError(f"source {source} has no hydraulic links in {scenario.scenario_id}")
    sensors = set(str(link) for link in scada.sensor_config.flow_sensors)
    if {link for link, _ in links} - sensors:
        raise ValueError(f"source links lack flow sensors in {scenario.scenario_id}")
    # Hydraulic SCADA retains quality sensor declarations but has no MSX arrays.
    # Select only hydraulic readings, as the shared outflow loader does.
    sensor_config = scada.sensor_config
    sensor_config.quality_node_sensors = []
    sensor_config.quality_link_sensors = []
    sensor_config.bulk_species_node_sensors = {}
    sensor_config.bulk_species_link_sensors = {}
    sensor_config.surface_species_sensors = {}
    scada.sensor_config = sensor_config
    times = np.asarray(scada.sensor_readings_time, dtype=np.int64)
    expected = np.arange(0, (SIMULATION_STEPS + 1) * 1800, 1800)
    if not np.array_equal(times, expected):
        raise ValueError(f"hydraulic time grid differs in {scenario.scenario_id}")
    units = ("CFS", "GPM", "MGD", "IMGD", "AFD", "LPS", "LPM",
             "MLD", "CMH", "CMD", "CMS")
    unit = units[topology.flow_units]
    columns = [link for link, _ in links]
    flow = np.asarray(scada.get_data_flows(sensor_locations=columns), dtype=np.float64)
    if flow.shape != (len(times), len(columns)) or not np.all(np.isfinite(flow)):
        raise ValueError(f"invalid source flow readings in {scenario.scenario_id}")
    directions = [direction for _, direction in links]
    return source_outflow_liters_per_minute(
        flow, directions, FLOW_TO_LITERS_PER_MINUTE[unit]
    ), reference.identity


def _simulate(
    scenario: Any, cache_root: Path, rates: np.ndarray, *, progress_label: str
) -> tuple[np.ndarray, np.ndarray]:
    from packages.trainer.simulation.epyt import EpytFlowBackend

    session = EpytFlowBackend(hydraulic_cache_root=cache_root).create_session(
        str(scenario.path),
        hydraulic_timestep_seconds=1800,
        calendar_start_day=scenario.calendar_start_day,
        chlorine_source_ids=scenario.chlorine_source_ids,
        monitored_node_ids=scenario.monitored_node_ids,
    )
    try:
        if tuple(scenario.monitored_node_ids) != NODE_IDS:
            raise RuntimeError(f"unexpected monitored nodes in {scenario.scenario_id}")
        first = session.reset(())
        times = [first.time_seconds]
        chlorine = [first.chlorine]
        if np.any(first.arsenic > 1e-12):
            raise RuntimeError("arsenic present in chlorine-only diagnostic")
        last_progress = monotonic()
        for step, injection in enumerate(rates):
            _check_stop()
            sample, terminated = session.step(float(injection), 0.0)
            times.append(sample.time_seconds)
            chlorine.append(sample.chlorine)
            if np.any(sample.arsenic > 1e-12):
                raise RuntimeError("arsenic present in chlorine-only diagnostic")
            if terminated and step + 1 < len(rates):
                raise RuntimeError("simulation terminated before requested follow-up")
            if (step + 1) % STEPS_PER_DAY == 0 and monotonic() - last_progress >= 30:
                _report_status(f"{progress_label} {step + 1}/{len(rates)} steps")
                last_progress = monotonic()
        return np.asarray(times), np.asarray(chlorine)
    finally:
        session.close()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _baseline_path(output: Path, scenario_id: str) -> Path:
    return output / "cases" / "baselines" / f"{scenario_id}.npz"


def _baseline_case(task: tuple[Any, str, str, bool]) -> str:
    from packages.trainer.simulation.hydraulic_cache import \
        DEFAULT_HYDRAULIC_CACHE_ROOT

    scenario, hydraulic_identity, output_raw, force = task
    output = Path(output_raw)
    path = _baseline_path(output, scenario.scenario_id)
    contract = _digest({
        "schema_version": SCHEMA_VERSION,
        "scenario_content_hash": scenario.content_hash,
        "hydraulic_cache_identity": hydraulic_identity,
        "baseline_mg_min": BASELINE_MG_MIN,
        "node_ids": NODE_IDS,
    })
    if not force and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as stored:
                if (str(stored["contract_hash"]) == contract
                        and stored["chlorine"].shape == (SIMULATION_STEPS + 1, len(NODE_IDS))
                        and np.array_equal(stored["times"], np.arange(
                            0, (SIMULATION_STEPS + 1) * 1800, 1800
                        ))
                        and np.all(np.isfinite(stored["chlorine"]))):
                    _report_progress(f"{scenario.scenario_id} cached")
                    return str(path)
        except (OSError, ValueError, KeyError):
            pass
    reference, _ = paired_rates(FIRST_CORE_DAY * STEPS_PER_DAY)
    _report_status(f"{scenario.scenario_id} baseline")
    times, chlorine = _simulate(
        scenario, DEFAULT_HYDRAULIC_CACHE_ROOT, reference,
        progress_label=f"Reference {scenario.scenario_id}",
    )
    _atomic_npz(path, contract_hash=contract, times=times, chlorine=chlorine)
    _report_progress(f"{scenario.scenario_id} complete")
    return str(path)


def _trial_path(output: Path, trial: Trial) -> Path:
    return output / "cases" / "trials" / f"{trial.scenario_id}-{trial.start_step}.npz"


def _trial_digest(scenario: Any, trial: Trial) -> str:
    return _digest({
        "schema_version": SCHEMA_VERSION,
        "scenario_content_hash": scenario.content_hash,
        "scenario_id": trial.scenario_id,
        "start_step": trial.start_step,
        "hydraulic_cache_identity": trial.hydraulic_cache_identity,
        "baseline_mg_min": BASELINE_MG_MIN,
        "first_action_delta_mg_min": FIRST_ACTION_DELTA_MG_MIN,
        "response_hours": RESPONSE_HOURS,
        "timestep_seconds": int(HOURS_PER_STEP * 3600),
        "node_ids": NODE_IDS,
    })


def _case_from_trajectories(
    trial: Trial, times: np.ndarray, reference: np.ndarray, action: np.ndarray,
) -> dict[str, Any]:
    samples = int(RESPONSE_HOURS / HOURS_PER_STEP) + 1
    expected_times = np.arange(trial.start_step, trial.start_step + samples) * 1800
    if not np.array_equal(times, expected_times):
        raise ValueError(f"trial {trial.scenario_id}/{trial.start_step} has an invalid time grid")
    shape = (samples, len(NODE_IDS))
    if (reference.shape != shape or action.shape != shape
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(action))):
        raise ValueError(f"trial {trial.scenario_id}/{trial.start_step} has invalid chlorine data")
    delta = action.astype(np.float64) - reference.astype(np.float64)
    if np.max(np.abs(delta[0])) > PRE_ACTION_MATCH_TOLERANCE_MG_L:
        raise RuntimeError(
            f"paired states differ before action in {trial.scenario_id} at {trial.start_step}"
        )
    return {
        "schedule": asdict(trial),
        "rows": [
            {**asdict(trial), "node_id": node_id, **analyze_response(delta[:, column])}
            for column, node_id in enumerate(NODE_IDS)
        ],
    }


def _cached_trial(path: Path, digest: str, trial: Trial) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    samples = int(RESPONSE_HOURS / HOURS_PER_STEP) + 1
    required = {
        "schema_version", "contract_hash", "scenario_id", "start_step", "node_ids",
        "times_seconds", "reference_chlorine_mg_l", "action_chlorine_mg_l",
        "reference_injection_mg_min", "action_injection_mg_min", "batch_start_steps",
    }
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != required:
                return None
            if (int(stored["schema_version"]) != SCHEMA_VERSION
                    or str(stored["contract_hash"]) != digest
                    or str(stored["scenario_id"]) != trial.scenario_id
                    or int(stored["start_step"]) != trial.start_step
                    or not np.array_equal(stored["node_ids"], NODE_IDS)):
                return None
            times = stored["times_seconds"]
            reference = stored["reference_chlorine_mg_l"]
            action = stored["action_chlorine_mg_l"]
            starts = stored["batch_start_steps"]
            reference_rates = stored["reference_injection_mg_min"]
            action_rates = stored["action_injection_mg_min"]
        if (starts.ndim != 1 or trial.start_step not in starts
                or not np.array_equal(starts, np.sort(starts))
                or np.any(np.diff(starts) < int(BATCH_SEPARATION_HOURS / HOURS_PER_STEP))
                or reference_rates.shape != (samples - 1,)
                or action_rates.shape != (samples - 1,)
                or not np.all(reference_rates == BASELINE_MG_MIN)
                or action_rates[0] != BASELINE_MG_MIN + FIRST_ACTION_DELTA_MG_MIN
                or not np.all(action_rates[1:] == BASELINE_MG_MIN)):
            return None
        return _case_from_trajectories(trial, times, reference, action)
    except (OSError, ValueError, KeyError, BadZipFile):
        return None


def _scenario_trial_cases(task: tuple[Any, tuple[Trial, ...], str]) -> list[dict[str, Any]]:
    """Simulate separated actions together while retaining per-trial results."""
    from packages.trainer.simulation.hydraulic_cache import \
        DEFAULT_HYDRAULIC_CACHE_ROOT

    scenario, trials, output_raw = task
    if not trials:
        return []
    if any(trial.scenario_id != scenario.scenario_id for trial in trials):
        raise ValueError("trial scenario differs from its worker scenario")
    output = Path(output_raw)
    with np.load(_baseline_path(output, scenario.scenario_id), allow_pickle=False) as stored:
        reference_times = stored["times"]
        reference = stored["chlorine"]
    results: dict[Trial, dict[str, Any]] = {}
    for batch in batch_trials(trials):
        _check_stop()
        last = batch[-1]
        end = last.start_step + int(RESPONSE_HOURS / HOURS_PER_STEP)
        rates = np.full(end, BASELINE_MG_MIN)
        for trial in batch:
            if not FIRST_CORE_DAY * STEPS_PER_DAY <= trial.start_step <= (
                LAST_CORE_DAY * STEPS_PER_DAY + int(23 / HOURS_PER_STEP)
            ) or trial.start_step + int(RESPONSE_HOURS / HOURS_PER_STEP) > SIMULATION_STEPS:
                raise ValueError("trial start lacks a valid 48-hour follow-up")
            rates[trial.start_step] += FIRST_ACTION_DELTA_MG_MIN
        label = (f"{scenario.scenario_id} {len(batch)} pulses, "
                 f"d{batch[0].day} {batch[0].clock_hour:02d}:00"
                 f"–d{last.day} {last.clock_hour:02d}:00")
        _report_status(label)
        times, intervention = _simulate(
            scenario, DEFAULT_HYDRAULIC_CACHE_ROOT, rates,
            progress_label=f"Action batch {label}",
        )
        if not np.array_equal(reference_times[:end + 1], times):
            raise RuntimeError(f"paired time grids differ for {scenario.scenario_id}")
        for trial in batch:
            stop = trial.start_step + int(RESPONSE_HOURS / HOURS_PER_STEP)
            window = slice(trial.start_step, stop + 1)
            window_times = times[window]
            window_reference = reference[window]
            window_action = intervention[window]
            result = _case_from_trajectories(
                trial, window_times, window_reference, window_action,
            )
            _atomic_npz(
                _trial_path(output, trial),
                schema_version=SCHEMA_VERSION,
                contract_hash=_trial_digest(scenario, trial),
                scenario_id=trial.scenario_id,
                start_step=trial.start_step,
                node_ids=np.asarray(NODE_IDS),
                times_seconds=window_times,
                reference_chlorine_mg_l=window_reference,
                action_chlorine_mg_l=window_action,
                reference_injection_mg_min=np.full(stop - trial.start_step, BASELINE_MG_MIN),
                action_injection_mg_min=rates[trial.start_step:stop],
                batch_start_steps=np.asarray([item.start_step for item in batch]),
            )
            results[trial] = result
            _report_progress(f"{scenario.scenario_id} d{trial.day} {trial.clock_hour:02d}:00")
    return [results[trial] for trial in trials]


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    numbers = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(numbers)) if len(numbers) else None,
        "median": float(np.median(numbers)) if len(numbers) else None,
        "p05": float(np.percentile(numbers, 5)) if len(numbers) else None,
        "p95": float(np.percentile(numbers, 95)) if len(numbers) else None,
        "maximum": float(np.max(numbers)) if len(numbers) else None,
    }


def summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    schedules = [case["schedule"] for case in cases]
    scheduled_by_scenario: dict[str, list[Trial]] = {}
    for schedule in schedules:
        scheduled_by_scenario.setdefault(schedule["scenario_id"], []).append(Trial(**schedule))
    action_batches = sum(len(batch_trials(group)) for group in scheduled_by_scenario.values())
    rows = [row for case in cases for row in case["rows"]]
    responsive = [row for row in rows if row["responsive"]]
    horizons = {}
    for hours in HORIZON_WINDOWS_HOURS:
        key = str(hours)
        fractions = [row[f"fraction_{key}h"] for row in responsive]
        peaks = sum(row[f"peak_reached_{key}h"] for row in responsive)
        substantial = sum(value >= .9 for value in fractions)
        horizons[key] = {
            "responsive_count": len(responsive),
            "fraction": _quantiles(fractions),
            "at_least_90_percent": substantial,
            "at_least_90_percent_pct": 100 * substantial / len(responsive) if responsive else None,
            "peak_reached": peaks,
            "peak_reached_pct": 100 * peaks / len(responsive) if responsive else None,
            "below_half": sum(value < .5 for value in fractions),
        }
    history = {}
    for hours in HISTORY_WINDOWS_HOURS:
        covered = sum(
            not row["tail_censored"] and row["last_detectable_hours"] <= hours
            for row in responsive
        )
        history[str(hours)] = {
            "covered": covered,
            "responsive_count": len(responsive),
            "percent": 100 * covered / len(responsive) if responsive else None,
        }
    per_node = {}
    for node_id in NODE_IDS:
        node_rows = [row for row in responsive if row["node_id"] == node_id]
        fractions = [row["fraction_8h"] for row in node_rows]
        per_node[node_id] = {
            "responsive": len(node_rows),
            "eight_hour_fraction": _quantiles(fractions),
            "eight_hour_below_half": sum(value < .5 for value in fractions),
            "eight_hour_peak_reached": sum(row["peak_reached_8h"] for row in node_rows),
            "latest_detectable_hours": max(
                (row["last_detectable_hours"] for row in node_rows), default=None
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "paired first-action chlorine screening at low source outflow",
        "scenario_ids": sorted({pulse["scenario_id"] for pulse in schedules}),
        "node_ids": list(NODE_IDS),
        "trial_schedule": schedules,
        "design": {
            "baseline_mg_min": BASELINE_MG_MIN,
            "first_action_delta_mg_min": FIRST_ACTION_DELTA_MG_MIN,
            "first_action_duration_hours": HOURS_PER_STEP,
            "source_outflow_window_hours": SOURCE_OUTFLOW_HOURS,
            "observation_hours": RESPONSE_HOURS,
            "history_windows_hours": list(HISTORY_WINDOWS_HOURS),
            "horizon_windows_hours": list(HORIZON_WINDOWS_HOURS),
            "action_batch_separation_hours": BATCH_SEPARATION_HOURS,
            "absolute_threshold_mg_l": ABSOLUTE_THRESHOLD_MG_L,
            "pre_action_match_tolerance_mg_l": PRE_ACTION_MATCH_TOLERANCE_MG_L,
            "fraction_definition": "positive paired difference above the fixed threshold through horizon divided by the corresponding sum through 48 hours",
        },
        "counts": {
            "scenarios": len({pulse["scenario_id"] for pulse in schedules}),
            "combinations": len({(pulse["weekday_phase"], pulse["clock_hour"])
                                 for pulse in schedules}),
            "trials": len(schedules),
            "action_batches": action_batches,
            "node_trial_cases": len(rows),
            "responsive": len(responsive),
            "nonresponsive": len(rows) - len(responsive),
            "tail_censored": sum(row["tail_censored"] for row in responsive),
        },
        "horizons": horizons,
        "history_coverage": history,
        "t90_hours": _quantiles([row["t90_hours"] for row in responsive]),
        "last_detectable_hours": _quantiles(
            [row["last_detectable_hours"] for row in responsive]
        ),
        "visible_duration_hours": _quantiles(
            [row["visible_duration_hours"] for row in responsive]
        ),
        "per_node": per_node,
        "source_outflow_24h_liters": _quantiles(
            [pulse["source_outflow_24h_liters"] for pulse in schedules]
        ),
        "weekday_phase_trials": {
            str(phase): sum(pulse["weekday_phase"] == phase for pulse in schedules)
            for phase in range(7)
        },
        "clock_hour_trials": {
            str(hour): sum(pulse["clock_hour"] == hour for pulse in schedules)
            for hour in sorted({pulse["clock_hour"] for pulse in schedules})
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty case table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _task_process_entry(
    send: Any, function: Any, task: Any, stop_event: Any, progress_lock: Any
) -> None:
    global _STOP_EVENT, _PROGRESS_PIPE
    _STOP_EVENT = stop_event
    _PROGRESS_PIPE = send
    tqdm.set_lock(progress_lock)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
        raise _RunInterrupted

    signal.signal(signal.SIGTERM, request_stop)
    try:
        _check_stop()
        send.send(("completed", function(task)))
    except _RunInterrupted:
        send.send(("interrupted", None))
    except BaseException:
        send.send(("failed", traceback.format_exc()))
    finally:
        send.close()


def _run_tasks(
    function: Any, tasks: Sequence[Any], workers: int, *,
    stop_state: _StopState, description: str, total: int, unit: str,
) -> list[Any]:
    """Run a bounded number of scenarios and stop cooperatively on signals."""
    global _PROGRESS_CALLBACK
    if not tasks:
        return []
    results: list[Any] = [None] * len(tasks)
    with tqdm(total=total, desc=description, unit=unit, file=sys.stdout,
              dynamic_ncols=True) as progress:
        if workers == 1:
            def update_progress(kind: str, label: str) -> None:
                if kind == "progress":
                    progress.update()
                if label:
                    progress.set_postfix_str(label)

            _PROGRESS_CALLBACK = update_progress
            try:
                for index, task in enumerate(tasks):
                    _check_stop()
                    results[index] = function(task)
            finally:
                _PROGRESS_CALLBACK = None
            return results

        context = mp.get_context("spawn")
        progress_lock = tqdm.get_lock()
        active: dict[Any, tuple[Any, int]] = {}
        next_task = 0
        completed = 0
        stop_state.defer_interrupt = True
        try:
            while next_task < len(tasks) or active:
                while (next_task < len(tasks) and len(active) < workers
                       and not stop_state.requested):
                    receive, send = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_task_process_entry,
                        args=(send, function, tasks[next_task], stop_state.event,
                              progress_lock),
                        name=f"network-delay-{description}-{next_task}",
                    )
                    try:
                        process.start()
                    except BaseException:
                        receive.close()
                        send.close()
                        raise
                    send.close()
                    active[receive] = (process, next_task)
                    next_task += 1

                ready = wait_connections(list(active), timeout=0.2) if active else []
                for receive in ready:
                    process, index = active[receive]
                    try:
                        kind, payload = receive.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(
                            f"{description} worker {index} exited without a result"
                        ) from exc
                    if kind in ("progress", "status"):
                        if kind == "progress":
                            progress.update()
                        if payload:
                            progress.set_postfix_str(payload)
                        continue
                    process.join(timeout=2.0)
                    receive.close()
                    del active[receive]
                    if process.is_alive():
                        process.kill()
                        process.join()
                        raise RuntimeError(f"{description} worker {index} did not exit")
                    if kind == "completed":
                        results[index] = payload
                        completed += 1
                    elif kind == "interrupted" and stop_state.requested:
                        pass
                    elif kind == "interrupted":
                        raise RuntimeError(f"{description} worker {index} stopped unexpectedly")
                    elif kind == "failed":
                        raise RuntimeError(
                            f"{description} worker {index} failed:\n{payload}"
                        )
                    else:
                        raise RuntimeError(f"unknown {description} worker message: {kind}")

                if stop_state.requested:
                    if (stop_state.requests > 1
                            or monotonic() - stop_state.requested_at >= GRACEFUL_STOP_SECONDS):
                        raise _RunInterrupted
                    if not active:
                        raise _RunInterrupted
                elif not active and next_task == len(tasks) and completed != len(tasks):
                    raise RuntimeError(f"{description} returned incomplete results")
        finally:
            for receive, (process, _) in active.items():
                if process.is_alive():
                    process.kill()
                process.join(timeout=2.0)
                receive.close()
            stop_state.defer_interrupt = False
    if stop_state.requested:
        raise _RunInterrupted
    return results


def _main_impl(argv: Sequence[str] | None, stop_state: _StopState) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("Progress is written to stdout. Ctrl-C or SIGTERM stops new "
                "work and closes active simulations."),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/analysis/network-delays/low-source-outflow"))
    parser.add_argument("--paper-summary", type=Path)
    parser.add_argument("--start-interval-hours", type=int, default=1)
    parser.add_argument("--max-combinations", type=int)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.max_scenarios is not None and not 10 <= args.max_scenarios <= len(SCENARIO_IDS):
        parser.error("max-scenarios must be between 10 and 80")
    try:
        all_slots = build_time_slots(args.start_interval_hours)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_combinations is not None and not 1 <= args.max_combinations <= len(all_slots):
        parser.error("max-combinations lies outside the available combinations")
    if args.paper_summary is not None and (
        args.start_interval_hours != 1
        or args.max_scenarios is not None and args.max_scenarios < len(SCENARIO_IDS)
        or args.max_combinations is not None and args.max_combinations < len(all_slots)
    ):
        parser.error("a partial study cannot publish the paper summary")
    from packages.trainer.data import resolve_dataset_split
    from packages.trainer.simulation.hydraulic_cache import \
        DEFAULT_HYDRAULIC_CACHE_ROOT
    from training.datasets.hanoi import control_full, forecasting_full

    if (SCENARIO_IDS != forecasting_full.train.scenarios
            or SCENARIO_IDS != control_full.train.scenarios):
        raise RuntimeError("diagnostic scenarios differ from the forecasting training split")
    selected_ids = SCENARIO_IDS[:args.max_scenarios]
    split = replace(control_full.train, scenarios=selected_ids)
    dataset = resolve_dataset_split(
        control_full.network, split, scenarios_dir=control_full.scenarios_dir,
        uncertainty_seed=control_full.uncertainty_seed,
    )
    scenarios = dataset.scenarios_for("train")
    if dataset.hydraulic_timestep_seconds != 1800 or dataset.duration_seconds != 25 * 86400:
        raise RuntimeError("diagnostic requires a 25-day scenario with 30-minute steps")
    if any(scenario.calendar_start_day != 0 for scenario in scenarios):
        raise RuntimeError("diagnostic requires a common day-zero weekly phase")
    slots = all_slots[:args.max_combinations]
    output = args.output_dir.resolve()
    source_outflows = {}
    with tqdm(scenarios, desc="Source hydraulics", unit="scenario",
              file=sys.stdout, dynamic_ncols=True) as progress:
        for scenario in progress:
            _check_stop()
            progress.set_postfix_str(scenario.scenario_id)
            source_outflows[scenario.scenario_id] = _source_outflow(
                scenario, DEFAULT_HYDRAULIC_CACHE_ROOT
            )
            _check_stop()
    trials = select_trials(slots, source_outflows)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    case_results: list[dict[str, Any] | None] = [None] * len(trials)
    pending = []
    for index, trial in enumerate(tqdm(
        trials, desc="Check cases", unit="trial", file=sys.stdout, dynamic_ncols=True
    )):
        _check_stop()
        scenario = scenarios_by_id[trial.scenario_id]
        digest = _trial_digest(scenario, trial)
        cached = None if args.force else _cached_trial(_trial_path(output, trial), digest, trial)
        if cached is None:
            pending.append((index, scenario, trial))
        else:
            case_results[index] = cached
    needed = {trial.scenario_id for _, _, trial in pending}
    baseline_tasks = [
        (scenario, source_outflows[scenario.scenario_id][1], str(output), args.force)
        for scenario in scenarios if scenario.scenario_id in needed
    ]
    _run_tasks(
        _baseline_case, baseline_tasks, args.workers, stop_state=stop_state,
        description="References", total=len(baseline_tasks), unit="scenario",
    )
    pending_by_scenario: dict[str, list[tuple[int, Trial]]] = {}
    for index, scenario, trial in pending:
        pending_by_scenario.setdefault(scenario.scenario_id, []).append((index, trial))
    trial_tasks = [
        (scenario, tuple(trial for _, trial in pending_by_scenario[scenario.scenario_id]),
         str(output))
        for scenario in scenarios if scenario.scenario_id in pending_by_scenario
    ]
    fresh_groups = _run_tasks(
        _scenario_trial_cases, trial_tasks, args.workers, stop_state=stop_state,
        description="Actions", total=len(pending), unit="trial",
    )
    for (scenario, _, _), results in zip(trial_tasks, fresh_groups):
        for (index, _), result in zip(pending_by_scenario[scenario.scenario_id], results):
            case_results[index] = result
    cases = [case for case in case_results if case is not None]
    if len(cases) != len(trials):
        raise RuntimeError("incomplete trial results")
    schedule_rows = [case["schedule"] for case in cases]
    node_rows = [row for case in cases for row in case["rows"]]
    summary = summarize(cases)
    summary["selection"] = {
        "candidate_scenario_ids": list(selected_ids),
        "start_interval_hours": args.start_interval_hours,
        "first_core_day": FIRST_CORE_DAY,
        "last_core_day": LAST_CORE_DAY,
        "scenarios_per_combination": SCENARIOS_PER_COMBINATION,
        "source_outflow_units": "liters integrated over 24 hours",
        "weekly_occurrence_rule": "minimum 24-hour volume per scenario; earliest day wins ties",
    }
    with tqdm(total=3 + (args.paper_summary is not None), desc="Write results",
              unit="file", file=sys.stdout, dynamic_ncols=True) as progress:
        _write_csv(output / "trial_schedule.csv", schedule_rows)
        progress.update()
        _write_csv(output / "node_trial_cases.csv", node_rows)
        progress.update()
        _atomic_json(output / "summary.json", summary)
        progress.update()
        if args.paper_summary is not None:
            _atomic_json(args.paper_summary, summary)
            progress.update()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    global _STOP_EVENT
    context = mp.get_context("spawn")
    stop_state = _StopState(context.Event())
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    tqdm.set_lock(context.RLock())
    _STOP_EVENT = stop_state.event
    for signum in previous_handlers:
        signal.signal(signum, stop_state.request)
    try:
        try:
            return _main_impl(argv, stop_state)
        except _RunInterrupted:
            return 128 + (stop_state.signum or signal.SIGINT)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _STOP_EVENT = None


if __name__ == "__main__":
    raise SystemExit(main())
