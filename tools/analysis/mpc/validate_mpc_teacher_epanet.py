#!/usr/bin/env python3
"""Validate saved MPC teacher plans against independent EPANET hold rollouts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Iterable

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.control.rewards import RewardInput  # noqa: E402
from packages.trainer.data import resolve_dataset_split  # noqa: E402
from packages.trainer.forecasting.bundle import load_bundle_manifest  # noqa: E402
from packages.trainer.forecasting.materialization import (  # noqa: E402
    ForecastTrajectoryResolver,
)
from packages.trainer.serialization import canonical_data  # noqa: E402
from packages.trainer.simulation import SimulationSample, TrajectoryStore  # noqa: E402
from packages.trainer.specs import DatasetSplitSpec, ForecastTrainingSpec  # noqa: E402
from tools.mpc.generate_mpc_teacher_dataset_cem import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_TEACHER_OUTPUT,
    DEFAULT_TEACHER_CONFIGURATION_HASH,
    ForecastBundleRuntime,
    HORIZON_STEPS,
    MAX_INJECTION,
    actions_to_injections,
    reconstruct_reward_state,
    reward_from_manifest,
    resolve_device,
    score_predicted_candidates,
    seed_everything,
)
from training.experiments.forecasting_encoder_decoder_lstm import (  # noqa: E402
    build_experiment,
)
from training.datasets.hanoi import forecasting_full  # noqa: E402
from training.rewards.v89 import V89Reward  # noqa: E402


DEFAULT_DATASET_DIR = DEFAULT_TEACHER_OUTPUT
DEFAULT_DATASET_NPZ = DEFAULT_DATASET_DIR / "dataset.npz"
DEFAULT_DATASET_MANIFEST = DEFAULT_DATASET_DIR / "manifest.json"
DEFAULT_TRAJECTORY_CACHE = PROJECT_ROOT / "data/runs"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "data/analysis/mpc-teacher-epanet/"
    f"{DEFAULT_TEACHER_CONFIGURATION_HASH}-s160-seed102"
)
TARGET_CHLORINE = 0.3
RESULT_SCHEMA_VERSION = 1
INJECTION_STORAGE_TOLERANCE = 0.02

INPUT_ARRAYS = (
    "scenario_id",
    "trajectory_identity",
    "chlorine_profile_hash",
    "arsenic_schedule_hash",
    "calendar_start_day",
    "query_index",
    "time_seconds",
    "current_injection",
    "first_action",
    "best_raw_actions",
    "best_injections",
    "predicted_chlorine",
    "predicted_return",
)


@dataclass(frozen=True)
class Selection:
    dataset_index: int
    query_time_stratum: str
    predicted_return_stratum: str
    action_magnitude_stratum: str
    action_sign_stratum: str


@dataclass(frozen=True)
class WorkerTask:
    dataset_index: int
    cache_dir: str
    scenario_path: str
    calendar_start_day: int
    chlorine_source_ids: tuple[str, ...]
    monitored_node_ids: tuple[str, ...]
    arsenic_events: tuple[Any, ...]
    query_index: int
    timestep_seconds: int
    mpc_actions: np.ndarray
    mpc_injections: np.ndarray
    hold_actions: np.ndarray
    hold_injections: np.ndarray
    reward_parameters: dict[str, float]
    discount_factor: float
    replay_tolerance: float


class TerminationState:
    """Signal state shared by the parent scheduler, not worker processes."""

    def __init__(self) -> None:
        self.requests = 0
        self.requested_at: float | None = None

    @property
    def requested(self) -> bool:
        return self.requests > 0

    @property
    def forced(self) -> bool:
        return self.requests > 1

    def request(self, *_args: Any) -> None:
        self.requests += 1
        if self.requested_at is None:
            self.requested_at = time.monotonic()
            print(
                "\nTermination requested: stopping dispatch and allowing active "
                "EPANET cases to close cleanly.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "\nSecond termination request: stopping active workers now.",
                file=sys.stderr,
                flush=True,
            )


@contextmanager
def _native_worker_thread_limits():
    """Set native thread limits before spawn imports PyTorch and EPANET."""
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _stable_tie(seed: int, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()[:8], "big"
    )


def _quartile_labels(values: np.ndarray, prefix: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    edges = np.quantile(array, (0.25, 0.5, 0.75), method="linear")
    bins = np.searchsorted(edges, array, side="right")
    return np.asarray([f"{prefix}-q{int(item) + 1}" for item in bins])


def _query_time_labels(query_indexes: np.ndarray) -> np.ndarray:
    values = np.asarray(query_indexes, dtype=np.int64)
    unique = np.unique(values)
    mapping = {
        int(value): min(3, (position * 4) // len(unique))
        for position, value in enumerate(unique)
    }
    return np.asarray([f"query-time-q{mapping[int(item)] + 1}" for item in values])


def _magnitude_labels(actions: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(actions, dtype=np.float64))
    result = np.empty(len(magnitude), dtype="<U20")
    saturated = np.isclose(magnitude, 1.0, rtol=0.0, atol=1e-6)
    result[saturated] = "saturated"
    result[~saturated & (magnitude < 0.5)] = "low"
    result[~saturated & (magnitude >= 0.5) & (magnitude < 0.9)] = "medium"
    result[~saturated & (magnitude >= 0.9)] = "high"
    return result


def _sign_labels(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    return np.where(values < -1e-6, "negative", np.where(values > 1e-6, "positive", "zero"))


def select_representative_indices(
    scenario_ids: np.ndarray,
    query_indexes: np.ndarray,
    predicted_returns: np.ndarray,
    first_actions: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> tuple[Selection, ...]:
    """Scenario-round-robin selection with balanced marginal strata."""
    scenario_ids = np.asarray(scenario_ids).astype(str)
    row_count = len(scenario_ids)
    if sample_count <= 0 or sample_count > row_count:
        raise ValueError(f"--sample-count must lie in [1, {row_count}]")
    if any(len(np.asarray(item)) != row_count for item in (query_indexes, predicted_returns, first_actions)):
        raise ValueError("selection arrays have inconsistent row counts")

    time_labels = _query_time_labels(query_indexes)
    return_labels = _quartile_labels(predicted_returns, "predicted-return")
    magnitude_labels = _magnitude_labels(first_actions)
    sign_labels = _sign_labels(first_actions)
    dimensions = (time_labels, return_labels, magnitude_labels)
    counts = [{label: 0 for label in np.unique(values)} for values in dimensions]
    sign_counts = {"negative": 0, "positive": 0, "zero": 0}

    scenarios = np.unique(scenario_ids)
    rng = np.random.default_rng(seed)
    scenario_order = scenarios[rng.permutation(len(scenarios))]
    candidates = {
        scenario: [int(item) for item in np.flatnonzero(scenario_ids == scenario)]
        for scenario in scenarios
    }
    used: set[int] = set()
    selected: list[Selection] = []
    round_index = 0
    while len(selected) < sample_count:
        made_progress = False
        for scenario_position, scenario in enumerate(scenario_order):
            if len(selected) >= sample_count:
                break
            available = [item for item in candidates[scenario] if item not in used]
            if not available:
                continue
            desired = "negative" if (round_index + scenario_position) % 2 == 0 else "positive"
            sign_matched = [item for item in available if sign_labels[item] == desired]
            pool = sign_matched or available

            def key(index: int) -> tuple[float, int]:
                balance = sum(
                    1.0 / (1.0 + dimension_counts[str(labels[index])])
                    for labels, dimension_counts in zip(dimensions, counts, strict=True)
                )
                balance += 1.0 / (1.0 + sign_counts[str(sign_labels[index])])
                return balance, -_stable_tie(seed, index)

            chosen = max(pool, key=key)
            used.add(chosen)
            made_progress = True
            for labels, dimension_counts in zip(dimensions, counts, strict=True):
                dimension_counts[str(labels[chosen])] += 1
            sign_counts[str(sign_labels[chosen])] += 1
            selected.append(
                Selection(
                    chosen,
                    str(time_labels[chosen]),
                    str(return_labels[chosen]),
                    str(magnitude_labels[chosen]),
                    str(sign_labels[chosen]),
                )
            )
        if not made_progress:
            raise RuntimeError("representative selection exhausted candidates")
        round_index += 1
    return tuple(selected)


def load_teacher_arrays(path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_manifest = [name for name in INPUT_ARRAYS if name not in manifest.get("arrays", {})]
    if missing_manifest:
        raise ValueError(f"teacher manifest is missing arrays: {missing_manifest}")
    if int(manifest.get("horizon_steps", -1)) != HORIZON_STEPS:
        raise ValueError("teacher dataset must use the 16-step MPC horizon")
    reward_from_manifest(manifest)
    discount_factor = manifest.get("reward", {}).get("discount_factor")
    if not isinstance(discount_factor, int | float) or not 0.0 <= discount_factor <= 1.0:
        raise ValueError("teacher dataset has an invalid reward discount factor")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in INPUT_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"teacher archive is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in INPUT_ARRAYS}
    rows = len(arrays["scenario_id"])
    if any(len(value) != rows for value in arrays.values()):
        raise ValueError("teacher arrays have inconsistent record counts")
    if arrays["best_raw_actions"].shape != (rows, HORIZON_STEPS):
        raise ValueError("teacher raw action plans have an incompatible shape")
    if arrays["best_injections"].shape != (rows, HORIZON_STEPS):
        raise ValueError("teacher injection plans have an incompatible shape")
    if arrays["predicted_chlorine"].shape[:2] != (rows, HORIZON_STEPS):
        raise ValueError("teacher chlorine forecasts have an incompatible shape")
    return arrays, manifest


def resolve_schedule_identities(
    selected_identities: set[str],
    cache_root: Path,
    bundle_manifest: dict[str, Any],
    *,
    forecast_job: ForecastTrainingSpec | None = None,
    source_split: DatasetSplitSpec = forecasting_full.train,
) -> tuple[dict[str, tuple[Any, Any, Any, Any]], Any]:
    job = build_experiment().jobs[0] if forecast_job is None else forecast_job
    if canonical_data(job.dataset) != bundle_manifest.get("job", {}).get("dataset"):
        raise ValueError("forecast bundle dataset does not match the supplied forecast job")
    dataset = resolve_dataset_split(job.dataset.network, source_split, scenarios_dir=job.dataset.scenarios_dir,
                                    uncertainty_seed=job.dataset.uncertainty_seed, split="train")
    spec = replace(job.trajectory_data, cache_root=cache_root, materialize_missing=False)
    resolver = ForecastTrajectoryResolver(dataset, spec, backend=None)
    resolved: dict[str, tuple[Any, Any, Any, Any]] = {}
    for reference, scenario, arsenic, chlorine in resolver._prepared("train"):
        if reference.identity in selected_identities:
            if reference.identity in resolved:
                raise ValueError(f"duplicate resolved trajectory identity: {reference.identity}")
            resolved[reference.identity] = (reference, scenario, arsenic, chlorine)
    missing = sorted(selected_identities - set(resolved))
    if missing:
        raise ValueError(f"selected trajectory identities could not be resolved: {missing[:3]}")
    return resolved, dataset


def _quality_metrics(chlorine: np.ndarray) -> dict[str, float]:
    values = np.asarray(chlorine, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {"violation_below_0_2": math.nan, "violation_below_0_25": math.nan, "target_mae": math.nan}
    return {
        "violation_below_0_2": float(np.mean(values < 0.2)),
        "violation_below_0_25": float(np.mean(values < 0.25)),
        "target_mae": float(np.mean(np.abs(values - TARGET_CHLORINE))),
    }


def _plan_metrics(
    actions: np.ndarray, injections: np.ndarray, previous_action: float, current_injection: float
) -> dict[str, float]:
    action_values = np.asarray(actions, dtype=np.float64)
    injection_values = np.asarray(injections, dtype=np.float64)
    return {
        "injection_sum": float(np.sum(injection_values)),
        "injection_mean": float(np.mean(injection_values)),
        "normalized_injection_mean": float(np.mean(injection_values / MAX_INJECTION)),
        "action_smoothness": float(np.mean(np.abs(np.diff(np.r_[previous_action, action_values])))),
        "injection_smoothness": float(np.mean(np.abs(np.diff(np.r_[current_injection, injection_values])))),
    }


def _reward_input_from_sample(sample: SimulationSample, step: int) -> RewardInput:
    strengths = sample.event_strength_by_node
    if strengths is None:
        strengths = np.zeros(len(sample.node_ids), dtype=np.float32)
    return RewardInput(
        nodes_quality=np.asarray(sample.chlorine),
        contaminant_values=np.asarray(sample.arsenic),
        pressure=np.asarray(sample.pressure),
        demand=np.asarray(sample.demand),
        flow=np.asarray(sample.flow),
        current_injection=float(sample.injection),
        raw_action=float(sample.action),
        step=step,
        current_time=int(sample.time_seconds),
        node_ids=sample.node_ids,
        link_ids=sample.link_ids,
        event_active=bool(sample.event_active),
        event_strength_by_node=np.asarray(strengths),
    )


def replay_to_query(session: Any, trajectory: Any, arsenic_events: tuple[Any, ...], query_index: int) -> tuple[Any, float, float]:
    """Replay the cache convention: reset is row zero, then apply rows 1..query."""
    sample = session.reset(arsenic_events)
    if query_index == 0:
        done = False
    for index in range(1, query_index + 1):
        sample, done = session.step(float(trajectory.injection[index]), float(trajectory.actions[index]))
        if done and index < query_index:
            raise RuntimeError(f"EPANET terminated during replay at row {index}")
    if tuple(sample.node_ids) != tuple(trajectory.node_ids):
        raise ValueError("replayed node order differs from cached trajectory")
    if int(sample.time_seconds) != int(trajectory.time_seconds[query_index]):
        raise ValueError("replayed query time differs from cached trajectory")
    error = np.abs(np.asarray(sample.chlorine, dtype=np.float64) - np.asarray(trajectory.chlorine[query_index], dtype=np.float64))
    return sample, float(np.max(error)), float(np.mean(error))


def execute_branch(
    task: WorkerTask,
    trajectory: Any,
    reward_state: dict[str, Any],
    *,
    actions: np.ndarray,
    injections: np.ndarray,
    backend_factory: Callable[[], Any],
) -> dict[str, Any]:
    backend = backend_factory()
    session = backend.create_session(
        task.scenario_path,
        hydraulic_timestep_seconds=task.timestep_seconds,
        calendar_start_day=task.calendar_start_day,
        chlorine_source_ids=task.chlorine_source_ids,
        monitored_node_ids=task.monitored_node_ids,
    )
    try:
        _query, replay_max, replay_mean = replay_to_query(
            session, trajectory, task.arsenic_events, task.query_index
        )
        reward = V89Reward(task.reward_parameters)
        reward.load_state_dict(reward_state)
        chlorine: list[np.ndarray] = []
        rewards: list[float] = []
        discounts = np.power(
            task.discount_factor, np.arange(HORIZON_STEPS, dtype=np.float64)
        )
        for offset, (action, injection) in enumerate(zip(actions, injections, strict=True)):
            sample, done = session.step(float(injection), float(action))
            if tuple(sample.node_ids) != tuple(trajectory.node_ids):
                raise ValueError("branch node order differs from cached trajectory")
            expected_time = int(trajectory.time_seconds[task.query_index]) + (offset + 1) * task.timestep_seconds
            if int(sample.time_seconds) != expected_time:
                raise ValueError("branch timestamp does not match the planning horizon")
            value, _info = reward.compute(
                _reward_input_from_sample(sample, task.query_index + offset + 1)
            )
            chlorine.append(np.asarray(sample.chlorine, dtype=np.float32))
            rewards.append(float(value))
            if done and offset < HORIZON_STEPS - 1:
                raise RuntimeError(f"EPANET terminated after {offset + 1} branch steps")
        reward_array = np.asarray(rewards, dtype=np.float64)
        return {
            "complete": True,
            "replay_max_abs_error": replay_max,
            "replay_mean_abs_error": replay_mean,
            "replay_valid": replay_max <= task.replay_tolerance,
            "chlorine": np.stack(chlorine).astype(np.float32),
            "rewards": reward_array,
            "return": float(np.dot(discounts, reward_array)),
            "failure_type": "",
            "failure_message": "",
        }
    finally:
        session.close()


def run_worker_task(
    task: WorkerTask, backend_factory: Callable[[], Any] | None = None
) -> dict[str, Any]:
    if backend_factory is None:
        from packages.trainer.simulation.epyt import EpytFlowBackend

        backend_factory = EpytFlowBackend
    trajectory = TrajectoryStore().load(Path(task.cache_dir))
    reward_state = reconstruct_reward_state(
        trajectory, task.query_index, task.reward_parameters
    )
    result: dict[str, Any] = {"dataset_index": task.dataset_index}
    for name, actions, injections in (
        ("mpc", task.mpc_actions, task.mpc_injections),
        ("hold", task.hold_actions, task.hold_injections),
    ):
        try:
            branch = execute_branch(
                task,
                trajectory,
                reward_state,
                actions=actions,
                injections=injections,
                backend_factory=backend_factory,
            )
        except BaseException as exc:
            branch = {
                "complete": False,
                "replay_max_abs_error": math.nan,
                "replay_mean_abs_error": math.nan,
                "replay_valid": False,
                "chlorine": np.full((HORIZON_STEPS, len(trajectory.node_ids)), np.nan, dtype=np.float32),
                "rewards": np.full(HORIZON_STEPS, np.nan, dtype=np.float64),
                "return": math.nan,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
        result[name] = branch
    return result


def _case_process_entry(send: Any, task: WorkerTask, backend_factory: Callable[[], Any] | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        send.send((True, run_worker_task(task, backend_factory)))
    except BaseException:
        try:
            send.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send.close()


def _worker_failure(task: WorkerTask, failure_type: str, message: str, *, cancelled: bool = False) -> dict[str, Any]:
    return {
        "dataset_index": task.dataset_index,
        "worker_failure": True,
        "cancelled": cancelled,
        "failure_type": failure_type,
        "failure_message": message,
    }


def run_case_processes(
    tasks: Iterable[WorkerTask],
    *,
    workers: int,
    shutdown_timeout: float,
    termination: TerminationState | None = None,
    backend_factory: Callable[[], Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Run disposable spawned case processes with bounded graceful shutdown."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    termination = termination or TerminationState()
    pending = list(tasks)
    context = mp.get_context("spawn")
    active: dict[int, tuple[Any, Any, WorkerTask]] = {}
    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(pending), desc="EPANET validation", unit="case", dynamic_ncols=True)
    deadline: float | None = None

    def finish(process_id: int, result: dict[str, Any]) -> None:
        process, receive, _task = active.pop(process_id)
        process.join(timeout=1.0)
        receive.close()
        results.append(result)
        progress.update()

    def stop(process: Any) -> None:
        if not process.is_alive():
            return
        if hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()

    try:
        while pending or active:
            while pending and len(active) < workers and not termination.requested:
                task = pending.pop(0)
                receive, send = context.Pipe(duplex=False)
                process = context.Process(
                    target=_case_process_entry,
                    args=(send, task, backend_factory),
                    name=f"mpc-epanet-{task.dataset_index}",
                )
                # PyTorch and epanet-plus ship separate OpenMP runtimes on
                # macOS. These limits must be inherited before either native
                # extension is imported by the spawned interpreter.
                with _native_worker_thread_limits():
                    process.start()
                send.close()
                active[process.pid] = (process, receive, task)

            if termination.requested and deadline is None:
                deadline = time.monotonic() + max(0.0, shutdown_timeout)
            force = termination.forced or (
                deadline is not None and time.monotonic() >= deadline
            )
            if force:
                for process_id, (process, receive, task) in list(active.items()):
                    stop(process)
                    process.join(timeout=2.0)
                    receive.close()
                    active.pop(process_id)
                    results.append(_worker_failure(task, "Cancelled", "worker stopped during graceful shutdown", cancelled=True))
                    progress.update()
                break

            connections = [item[1] for item in active.values()]
            ready = wait_connections(connections, timeout=0.1) if connections else []
            ready_set = set(ready)
            for process_id, (process, receive, task) in list(active.items()):
                if receive in ready_set:
                    try:
                        ok, payload = receive.recv()
                    except (EOFError, OSError) as exc:
                        payload = _worker_failure(task, type(exc).__name__, "worker exited without a result payload")
                        ok = True
                    process.join(timeout=2.0)
                    if ok and process.exitcode in (0, None):
                        finish(process_id, payload)
                    elif ok:
                        finish(process_id, _worker_failure(task, "WorkerExit", f"worker exited with code {process.exitcode}"))
                    else:
                        finish(process_id, _worker_failure(task, "WorkerException", str(payload)))
                elif not process.is_alive():
                    process.join(timeout=1.0)
                    if receive.poll():
                        try:
                            ok, payload = receive.recv()
                        except (EOFError, OSError):
                            ok, payload = False, "worker closed its result pipe without a payload"
                        if ok and process.exitcode == 0:
                            finish(process_id, payload)
                        elif ok:
                            finish(process_id, _worker_failure(task, "WorkerExit", f"worker exited with code {process.exitcode}"))
                        else:
                            finish(process_id, _worker_failure(task, "WorkerException", str(payload)))
                    else:
                        finish(process_id, _worker_failure(task, "WorkerExit", f"worker exited without payload (code {process.exitcode})"))

            if termination.requested and not active:
                break

        if termination.requested:
            for task in pending:
                results.append(_worker_failure(task, "Cancelled", "case was not dispatched", cancelled=True))
                progress.update()
            pending.clear()
    finally:
        for process, receive, _task in active.values():
            stop(process)
            process.join(timeout=2.0)
            receive.close()
        progress.close()
    results.sort(key=lambda item: int(item["dataset_index"]))
    return tuple(results), termination.requested


def _base_case(arrays: dict[str, np.ndarray], selection: Selection, node_count: int) -> dict[str, Any]:
    index = selection.dataset_index
    actions = np.asarray(arrays["best_raw_actions"][index], dtype=np.float32)
    injections = np.asarray(arrays["best_injections"][index], dtype=np.float32)
    current = float(arrays["current_injection"][index])
    hold_actions = np.zeros(HORIZON_STEPS, dtype=np.float32)
    hold_injections = np.full(HORIZON_STEPS, current, dtype=np.float32)
    case: dict[str, Any] = {
        "dataset_index": index,
        "scenario_id": str(arrays["scenario_id"][index]),
        "trajectory_identity": str(arrays["trajectory_identity"][index]),
        "chlorine_profile_hash": str(arrays["chlorine_profile_hash"][index]),
        "arsenic_schedule_hash": str(arrays["arsenic_schedule_hash"][index]),
        "calendar_start_day": int(arrays["calendar_start_day"][index]),
        "query_index": int(arrays["query_index"][index]),
        "time_seconds": int(arrays["time_seconds"][index]),
        "query_time_stratum": selection.query_time_stratum,
        "predicted_return_stratum": selection.predicted_return_stratum,
        "action_magnitude_stratum": selection.action_magnitude_stratum,
        "action_sign_stratum": selection.action_sign_stratum,
        "stored_current_injection": current,
        "current_injection": current,
        "first_action": float(arrays["first_action"][index]),
        "mpc_raw_actions": actions,
        "mpc_injections": injections,
        "hold_raw_actions": hold_actions,
        "hold_injections": hold_injections,
        "predicted_mpc_chlorine": np.asarray(arrays["predicted_chlorine"][index], dtype=np.float32),
        "predicted_hold_chlorine": np.full((HORIZON_STEPS, node_count), np.nan, dtype=np.float32),
        "predicted_mpc_return": float(arrays["predicted_return"][index]),
        "predicted_hold_return": math.nan,
        "predicted_gain": math.nan,
        "realized_mpc_chlorine": np.full((HORIZON_STEPS, node_count), np.nan, dtype=np.float32),
        "realized_hold_chlorine": np.full((HORIZON_STEPS, node_count), np.nan, dtype=np.float32),
        "realized_mpc_rewards": np.full(HORIZON_STEPS, np.nan, dtype=np.float64),
        "realized_hold_rewards": np.full(HORIZON_STEPS, np.nan, dtype=np.float64),
        "realized_mpc_return": math.nan,
        "realized_hold_return": math.nan,
        "realized_gain": math.nan,
        "mpc_replay_max_abs_error": math.nan,
        "mpc_replay_mean_abs_error": math.nan,
        "hold_replay_max_abs_error": math.nan,
        "hold_replay_mean_abs_error": math.nan,
        "query_replay_error": math.nan,
        "previous_action": math.nan,
        "prediction_complete": False,
        "mpc_branch_complete": False,
        "hold_branch_complete": False,
        "replay_valid": False,
        "completed": False,
        "cancelled": False,
        "valid": False,
        "status": "pending",
        "failure_stage": "",
        "failure_type": "",
        "failure_message": "",
    }
    for domain in ("predicted", "realized"):
        for arm in ("mpc", "hold"):
            for metric in ("violation_below_0_2", "violation_below_0_25", "target_mae"):
                case[f"{domain}_{arm}_{metric}"] = math.nan
    for arm in ("mpc", "hold"):
        for metric in ("injection_sum", "injection_mean", "normalized_injection_mean", "action_smoothness", "injection_smoothness"):
            case[f"{arm}_{metric}"] = math.nan
    return case


def _append_failure(case: dict[str, Any], stage: str, failure_type: str, message: str) -> None:
    case["failure_stage"] = ";".join(filter(None, (case["failure_stage"], stage)))
    case["failure_type"] = ";".join(filter(None, (case["failure_type"], failure_type)))
    case["failure_message"] = " | ".join(filter(None, (case["failure_message"], message)))


def prepare_case(
    case: dict[str, Any],
    resolved: tuple[Any, Any, Any, Any],
    dataset: Any,
    runtime: ForecastBundleRuntime,
    teacher_manifest: dict[str, Any],
) -> WorkerTask:
    reference, scenario, arsenic, chlorine_schedule = resolved
    trajectory = reference.load()
    if (
        trajectory.scenario_id != case["scenario_id"]
        or trajectory.arsenic_schedule_hash != case["arsenic_schedule_hash"]
        or trajectory.chlorine_profile_hash != case["chlorine_profile_hash"]
        or trajectory.calendar_start_day != case["calendar_start_day"]
    ):
        raise ValueError("selected case metadata differs from its cached trajectory")
    if tuple(trajectory.node_ids) != tuple(runtime.node_ids):
        raise ValueError("cached node order differs from forecast bundle")
    if not np.all(np.diff(np.asarray(trajectory.time_seconds)) == dataset.hydraulic_timestep_seconds):
        raise ValueError("cached trajectory timestep differs from the dataset")
    schedule_values = np.asarray(chlorine_schedule.values, dtype=np.float64)
    cached_injections = np.asarray(trajectory.injection, dtype=np.float64)
    if (
        len(cached_injections) != len(schedule_values) + 1
        or cached_injections[0] != 0.0
        or not np.allclose(
            cached_injections[1:], schedule_values, rtol=0.0, atol=1e-6
        )
    ):
        raise ValueError("resolved chlorine schedule differs from cached injection history")

    query_index = int(case["query_index"])
    if int(trajectory.time_seconds[query_index]) != int(case["time_seconds"]):
        raise ValueError("selected query time differs from the cached trajectory")
    exact_current_injection = float(trajectory.injection[query_index])
    if not np.isclose(
        exact_current_injection,
        case["stored_current_injection"],
        rtol=0.0,
        atol=INJECTION_STORAGE_TOLERANCE,
    ):
        raise ValueError("selected current injection differs from the cached trajectory")
    case["current_injection"] = exact_current_injection
    case["hold_injections"] = np.full(
        HORIZON_STEPS, exact_current_injection, dtype=np.float32
    )
    if not np.isclose(case["first_action"], float(case["mpc_raw_actions"][0]), rtol=0.0, atol=1e-7):
        raise ValueError("stored first action differs from the MPC action plan")
    action_contract = teacher_manifest["action_contract"]
    reconstructed_plan = actions_to_injections(
        case["mpc_raw_actions"],
        case["current_injection"],
        max_delta=float(action_contract["max_delta"]),
        minimum=float(action_contract["minimum_injection"]),
        maximum=float(action_contract["maximum_injection"]),
    )
    if not np.allclose(
        reconstructed_plan,
        case["mpc_injections"],
        rtol=0.0,
        atol=INJECTION_STORAGE_TOLERANCE,
    ):
        raise ValueError("stored MPC injections violate the teacher action contract")
    prepared = runtime.prepare_query(trajectory, query_index)
    reward_parameters = dict(teacher_manifest["reward"]["parameters"])
    reward_from_manifest(teacher_manifest)
    discount_factor = float(teacher_manifest["reward"]["discount_factor"])
    reward_state = reconstruct_reward_state(
        trajectory, query_index, reward_parameters
    )
    hold_prediction = runtime.predict_candidates(
        prepared,
        np.asarray(case["hold_injections"], dtype=np.float32)[None, :],
        np.asarray(trajectory.chlorine[query_index], dtype=np.float32),
    )[0]
    hold_return = score_predicted_candidates(
        hold_prediction[None, ...],
        np.asarray(case["hold_raw_actions"], dtype=np.float32)[None, :],
        np.asarray(case["hold_injections"], dtype=np.float32)[None, :],
        reward_state=reward_state,
        trajectory=trajectory,
        query_index=query_index,
        timestep_seconds=dataset.hydraulic_timestep_seconds,
        discount_factor=discount_factor,
        reward_parameters=reward_parameters,
    )[0]
    case["predicted_hold_chlorine"] = hold_prediction
    case["predicted_hold_return"] = float(hold_return)
    case["predicted_gain"] = case["predicted_mpc_return"] - float(hold_return)
    case["previous_action"] = float(trajectory.actions[query_index])
    case["prediction_complete"] = True
    for arm in ("mpc", "hold"):
        for metric, value in _quality_metrics(case[f"predicted_{arm}_chlorine"]).items():
            case[f"predicted_{arm}_{metric}"] = value
        for metric, value in _plan_metrics(
            case[f"{arm}_raw_actions"], case[f"{arm}_injections"], case["previous_action"], case["current_injection"]
        ).items():
            case[f"{arm}_{metric}"] = value

    return WorkerTask(
        dataset_index=case["dataset_index"],
        cache_dir=str(reference.cache_dir),
        scenario_path=str(scenario.path),
        calendar_start_day=scenario.calendar_start_day,
        chlorine_source_ids=scenario.chlorine_source_ids,
        monitored_node_ids=scenario.monitored_node_ids,
        arsenic_events=arsenic.events,
        query_index=query_index,
        timestep_seconds=dataset.hydraulic_timestep_seconds,
        mpc_actions=case["mpc_raw_actions"],
        mpc_injections=case["mpc_injections"],
        hold_actions=case["hold_raw_actions"],
        hold_injections=case["hold_injections"],
        reward_parameters=reward_parameters,
        discount_factor=discount_factor,
        replay_tolerance=0.0,  # overwritten by the caller
    )


def merge_worker_result(case: dict[str, Any], result: dict[str, Any], replay_tolerance: float) -> None:
    if result.get("worker_failure"):
        case["cancelled"] = bool(result.get("cancelled", False))
        case["status"] = "cancelled" if case["cancelled"] else "failed-worker"
        _append_failure(case, "worker", str(result.get("failure_type", "WorkerFailure")), str(result.get("failure_message", "")))
        return
    for arm in ("mpc", "hold"):
        branch = result[arm]
        case[f"{arm}_branch_complete"] = bool(branch["complete"])
        case[f"{arm}_replay_max_abs_error"] = float(branch["replay_max_abs_error"])
        case[f"{arm}_replay_mean_abs_error"] = float(branch["replay_mean_abs_error"])
        case[f"realized_{arm}_chlorine"] = branch["chlorine"]
        case[f"realized_{arm}_rewards"] = branch["rewards"]
        case[f"realized_{arm}_return"] = float(branch["return"])
        if not branch["complete"]:
            _append_failure(case, f"epanet-{arm}", branch["failure_type"], branch["failure_message"])
        else:
            for metric, value in _quality_metrics(branch["chlorine"]).items():
                case[f"realized_{arm}_{metric}"] = value
    replay_errors = [case["mpc_replay_max_abs_error"], case["hold_replay_max_abs_error"]]
    case["query_replay_error"] = float(np.nanmax(replay_errors)) if np.any(np.isfinite(replay_errors)) else math.nan
    case["replay_valid"] = bool(
        case["mpc_branch_complete"]
        and case["hold_branch_complete"]
        and all(np.isfinite(value) and value <= replay_tolerance for value in replay_errors)
    )
    if case["mpc_branch_complete"] and case["hold_branch_complete"]:
        case["realized_gain"] = case["realized_mpc_return"] - case["realized_hold_return"]
        case["completed"] = True
    case["valid"] = bool(case["prediction_complete"] and case["completed"] and case["replay_valid"])
    if case["valid"]:
        case["status"] = "valid"
    elif case["completed"] and not case["replay_valid"]:
        case["status"] = "invalid-replay"
        _append_failure(case, "replay", "ReplayMismatch", f"query replay error exceeds {replay_tolerance:g} mg/L")
    else:
        case["status"] = "failed-case"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    if rank:
        left, right = _average_ranks(left), _average_ranks(right)
    return float(np.corrcoef(left, right)[0, 1])


def _comparison_metrics(predicted: np.ndarray, realized: np.ndarray) -> dict[str, float | None]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(realized, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_predicted_minus_realized": float(np.mean(error)),
        "pearson": _correlation(predicted, realized),
        "spearman": _correlation(predicted, realized, rank=True),
    }


def _bootstrap_interval(values: np.ndarray, statistic: Callable[[np.ndarray], float], seed: int, samples: int) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"estimate": None, "ci_95_low": None, "ci_95_high": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = statistic(values[rng.integers(0, len(values), size=len(values))])
    return {
        "estimate": float(statistic(values)),
        "ci_95_low": float(np.quantile(estimates, 0.025)),
        "ci_95_high": float(np.quantile(estimates, 0.975)),
    }


def _gain_group(cases: list[dict[str, Any]], tie_tolerance: float) -> dict[str, Any]:
    valid = [case for case in cases if case["valid"]]
    gains = np.asarray([case["realized_gain"] for case in valid], dtype=np.float64)
    predicted = np.asarray([case["predicted_gain"] for case in valid], dtype=np.float64)
    count = len(valid)
    wins = int(np.sum(gains > tie_tolerance))
    losses = int(np.sum(gains < -tie_tolerance))
    ties = count - wins - losses
    result = {
        "selected_count": len(cases),
        "valid_count": count,
        "failed_count": len(cases) - count,
        "win_fraction": wins / count if count else None,
        "tie_fraction": ties / count if count else None,
        "loss_fraction": losses / count if count else None,
        "mean_realized_gain": float(np.mean(gains)) if count else None,
        "median_realized_gain": float(np.median(gains)) if count else None,
        "mean_predicted_gain": float(np.mean(predicted)) if count else None,
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(gains))) if count else None,
    }
    for domain in ("predicted", "realized"):
        for metric in ("violation_below_0_2", "violation_below_0_25", "target_mae"):
            differences = [
                case[f"{domain}_mpc_{metric}"] - case[f"{domain}_hold_{metric}"]
                for case in valid
            ]
            result[f"mean_{domain}_mpc_minus_hold_{metric}"] = (
                float(np.mean(differences)) if differences else None
            )
    return result


def summarize_cases(
    cases: list[dict[str, Any]], *, seed: int, bootstrap_samples: int, tie_tolerance: float, interrupted: bool
) -> dict[str, Any]:
    valid = [case for case in cases if case["valid"]]
    gains = np.asarray([case["realized_gain"] for case in valid], dtype=np.float64)
    group = _gain_group(cases, tie_tolerance)
    predicted_vs_realized: dict[str, Any] = {}
    for name in ("mpc_return", "hold_return", "gain"):
        predicted = np.asarray([case[f"predicted_{name}"] for case in valid], dtype=np.float64)
        realized = np.asarray([case[f"realized_{name}"] for case in valid], dtype=np.float64)
        predicted_vs_realized[name] = _comparison_metrics(predicted, realized) if len(valid) else {
            "mae": None, "rmse": None, "bias_predicted_minus_realized": None, "pearson": None, "spearman": None
        }
    quality_differences: dict[str, float | None] = {}
    for domain in ("predicted", "realized"):
        for metric in ("violation_below_0_2", "violation_below_0_25", "target_mae"):
            values = [case[f"{domain}_mpc_{metric}"] - case[f"{domain}_hold_{metric}"] for case in valid]
            quality_differences[f"{domain}_mpc_minus_hold_{metric}"] = float(np.mean(values)) if values else None
    for metric in ("injection_sum", "injection_mean", "normalized_injection_mean", "action_smoothness", "injection_smoothness"):
        values = [case[f"mpc_{metric}"] - case[f"hold_{metric}"] for case in valid]
        quality_differences[f"mpc_minus_hold_{metric}"] = float(np.mean(values)) if values else None

    breakdowns: dict[str, dict[str, Any]] = {}
    for field in ("query_time_stratum", "predicted_return_stratum", "action_magnitude_stratum"):
        breakdowns[field] = {
            label: _gain_group([case for case in cases if case[field] == label], tie_tolerance)
            for label in sorted({case[field] for case in cases})
        }
    status_counts = {status: sum(case["status"] == status for case in cases) for status in sorted({case["status"] for case in cases})}
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "interrupted": interrupted,
        "selected_count": len(cases),
        "completed_count": sum(bool(case["completed"]) for case in cases),
        "valid_count": len(valid),
        "failed_count": sum(not case["valid"] and not case["cancelled"] for case in cases),
        "cancelled_count": sum(bool(case["cancelled"]) for case in cases),
        "status_counts": status_counts,
        "realized_outcome": group,
        "realized_gain_bootstrap": {
            "samples": bootstrap_samples,
            "mean": _bootstrap_interval(gains, lambda value: float(np.mean(value)), seed, bootstrap_samples),
            "median": _bootstrap_interval(gains, lambda value: float(np.median(value)), seed + 1, bootstrap_samples),
        },
        "predicted_vs_realized": predicted_vs_realized,
        "predicted_realized_gain_sign_agreement": group["sign_agreement"],
        "aggregate_differences": quality_differences,
        "breakdowns": breakdowns,
        "failures": [
            {
                "dataset_index": case["dataset_index"],
                "scenario_id": case["scenario_id"],
                "trajectory_identity": case["trajectory_identity"],
                "status": case["status"],
                "stage": case["failure_stage"],
                "type": case["failure_type"],
                "message": case["failure_message"],
            }
            for case in cases if not case["valid"]
        ],
    }
    return summary


def cases_to_arrays(cases: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not cases:
        raise ValueError("cannot write an empty case set")
    keys = tuple(cases[0])
    if any(tuple(case) != keys for case in cases):
        raise ValueError("case result schemas differ")
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [case[key] for case in cases]
        if isinstance(values[0], np.ndarray):
            arrays[key] = np.stack(values)
        elif isinstance(values[0], (bool, np.bool_)):
            arrays[key] = np.asarray(values, dtype=np.bool_)
        elif isinstance(values[0], (int, np.integer)):
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif isinstance(values[0], (float, np.floating)):
            arrays[key] = np.asarray(values, dtype=np.float64)
        else:
            arrays[key] = np.asarray([str(value) for value in values], dtype=np.str_)
    return arrays


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_plots(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [case for case in cases if case["valid"]]
    predicted = np.asarray([case["predicted_gain"] for case in valid], dtype=np.float64)
    realized = np.asarray([case["realized_gain"] for case in valid], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    if len(valid):
        axis.scatter(predicted, realized, s=22, alpha=0.7, edgecolors="none")
        lo = float(min(np.min(predicted), np.min(realized)))
        hi = float(max(np.max(predicted), np.max(realized)))
        axis.plot((lo, hi), (lo, hi), color="black", linestyle="--", linewidth=1)
        pearson = _correlation(predicted, realized)
        spearman = _correlation(predicted, realized, rank=True)
        axis.text(0.02, 0.98, f"n={len(valid)}\nPearson={pearson if pearson is not None else 'n/a'}\nSpearman={spearman if spearman is not None else 'n/a'}", transform=axis.transAxes, va="top", fontsize=8)
    else:
        axis.text(0.5, 0.5, "No valid completed cases", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Predicted MPC − hold return")
    axis.set_ylabel("Realized EPANET MPC − hold return")
    axis.set_title("Predicted vs. realized MPC gain")
    figure.savefig(output_dir / "predicted_vs_realized_gain.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    if len(valid):
        axis.hist(realized, bins=min(20, max(5, int(np.sqrt(len(realized))))), color="#4477AA", alpha=0.85)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.axvline(float(np.mean(realized)), color="#CC3311", linestyle="--", label="mean")
        axis.axvline(float(np.median(realized)), color="#009988", linestyle=":", label="median")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No valid completed cases", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Realized EPANET MPC − hold return")
    axis.set_ylabel("Cases")
    axis.set_title("Realized MPC gain distribution")
    figure.savefig(output_dir / "realized_gain_distribution.png", dpi=160)
    plt.close(figure)


def ensure_fresh_output(output_dir: Path) -> None:
    targets = (
        output_dir / "validation_cases.npz",
        output_dir / "summary.json",
        output_dir / "predicted_vs_realized_gain.png",
        output_dir / "realized_gain_distribution.png",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"validation output already exists: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)


def run_validation(
    args: argparse.Namespace,
    termination: TerminationState,
    *,
    forecast_job: ForecastTrainingSpec | None = None,
    source_split: DatasetSplitSpec = forecasting_full.train,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    arrays, teacher_manifest = load_teacher_arrays(args.dataset_npz, args.dataset_manifest)
    saved_forecast_bundle = Path(teacher_manifest["forecasting_bundle"])
    forecast_bundle = (
        saved_forecast_bundle
        if args.forecast_bundle is None
        else Path(args.forecast_bundle)
    )
    if forecast_bundle.resolve() != saved_forecast_bundle.resolve():
        raise ValueError(
            "forecast bundle override disagrees with the saved teacher manifest"
        )
    bundle_manifest = load_bundle_manifest(forecast_bundle)
    selections = select_representative_indices(
        arrays["scenario_id"], arrays["query_index"], arrays["predicted_return"], arrays["first_action"],
        sample_count=args.sample_count, seed=args.seed,
    )
    node_count = arrays["predicted_chlorine"].shape[2]
    cases = [_base_case(arrays, selection, node_count) for selection in selections]
    identities = {case["trajectory_identity"] for case in cases}
    resolved, dataset = resolve_schedule_identities(
        identities,
        args.trajectory_cache,
        bundle_manifest,
        forecast_job=forecast_job,
        source_split=source_split,
    )
    runtime = ForecastBundleRuntime(forecast_bundle, resolve_device(args.device))
    seed_everything(args.seed)

    tasks: list[WorkerTask] = []
    progress = tqdm(cases, desc="Hold forecasts", unit="case", dynamic_ncols=True)
    try:
        for case in progress:
            if termination.requested:
                case["cancelled"] = True
                case["status"] = "cancelled"
                _append_failure(case, "forecast", "Cancelled", "case preparation was cancelled")
                continue
            try:
                task = prepare_case(
                    case,
                    resolved[case["trajectory_identity"]],
                    dataset,
                    runtime,
                    teacher_manifest,
                )
                tasks.append(replace(task, replay_tolerance=args.replay_tolerance))
            except BaseException as exc:
                case["status"] = "failed-preparation"
                _append_failure(case, "preparation", type(exc).__name__, str(exc))
    finally:
        progress.close()

    worker_results: tuple[dict[str, Any], ...] = ()
    interrupted = termination.requested
    if tasks and not termination.forced:
        worker_results, worker_interrupted = run_case_processes(
            tasks,
            workers=args.workers,
            shutdown_timeout=args.shutdown_timeout,
            termination=termination,
        )
        interrupted = interrupted or worker_interrupted
    elif termination.requested:
        for task in tasks:
            worker_results += (_worker_failure(task, "Cancelled", "case was not dispatched", cancelled=True),)
    by_index = {case["dataset_index"]: case for case in cases}
    for result in worker_results:
        merge_worker_result(by_index[int(result["dataset_index"])], result, args.replay_tolerance)
    for case in cases:
        if case["status"] == "pending":
            case["status"] = "cancelled" if interrupted else "failed-worker"
            case["cancelled"] = interrupted
            _append_failure(case, "worker", "Cancelled" if interrupted else "MissingResult", "no worker result was produced")
    cases.sort(key=lambda case: case["dataset_index"])
    summary = summarize_cases(
        cases,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        tie_tolerance=args.tie_tolerance,
        interrupted=interrupted,
    )
    summary["configuration"] = {
        "dataset_npz": str(args.dataset_npz),
        "dataset_manifest": str(args.dataset_manifest),
        "trajectory_cache": str(args.trajectory_cache),
        "forecast_bundle": str(forecast_bundle),
        "output_dir": str(args.output_dir),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "workers": args.workers,
        "device": str(args.device),
        "discount_factor": float(teacher_manifest["reward"]["discount_factor"]),
        "replay_tolerance_mg_per_l": args.replay_tolerance,
        "tie_tolerance": args.tie_tolerance,
        "shutdown_timeout_seconds": args.shutdown_timeout,
        "teacher_kind": teacher_manifest.get("kind"),
    }
    summary["method"] = {
        "selection": "scenario-round-robin with alternating action sign and balanced query-time/return/magnitude strata",
        "query_replay": "reset is cache row 0; cached injections/actions rows 1 through query are replayed",
        "branch_sessions": "separate fresh spawned-process EPANET sessions for MPC and hold",
        "predicted_hold": "frozen forecast bundle; non-chlorine reward inputs held at query as in teacher generation",
    }
    return cases, summary, interrupted


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.replay_tolerance < 0.0 or args.tie_tolerance < 0.0 or args.shutdown_timeout < 0.0:
        raise ValueError("tolerances and shutdown timeout must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--trajectory-cache", type=Path, default=DEFAULT_TRAJECTORY_CACHE)
    parser.add_argument(
        "--forecast-bundle",
        type=Path,
        default=None,
        help=(
            "optional consistency-checked override; by default the forecast "
            "bundle is read from the teacher manifest"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-tolerance", type=float, default=1e-5)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("dataset_npz", "dataset_manifest", "trajectory_cache", "output_dir"):
        setattr(args, name, _project_path(getattr(args, name)).resolve())
    if args.forecast_bundle is not None:
        args.forecast_bundle = _project_path(args.forecast_bundle).resolve()
    termination = TerminationState()
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, termination.request)
    try:
        _validate_args(args)
        ensure_fresh_output(args.output_dir)
        cases, summary, interrupted = run_validation(args, termination)
        _atomic_npz(args.output_dir / "validation_cases.npz", cases_to_arrays(cases))
        _atomic_json(args.output_dir / "summary.json", summary)
        write_plots(args.output_dir, cases)
        print(
            f"Wrote {len(cases)} cases to {args.output_dir}; "
            f"valid={summary['valid_count']}, failed={summary['failed_count']}, "
            f"cancelled={summary['cancelled_count']}"
        )
        if interrupted:
            return 130
        return 0 if summary["valid_count"] == len(cases) else 2
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
