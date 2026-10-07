"""Lazy, identity-keyed forecast trajectory materialization."""

from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import ResolvedDataset, ResolvedDatasetSplit, ScenarioEntry
from ..progress import tqdm
from ..schedules import (ArsenicSchedule, FixedChlorineInjectionScheduleIterator,
                         resolve_schedule_source)
from ..serialization import canonical_data
from ..simulation import (Trajectory, TrajectoryStore,
                          prepare_hydraulic_outflow_cache)
from ..specs import ForecastTrajectoryDataSpec


@dataclass(frozen=True)
class TrajectoryReference:
    identity: str
    cache_dir: Path
    scenario_id: str
    arsenic_schedule_hash: str
    chlorine_schedule_hash: str

    def load(self) -> Trajectory: return TrajectoryStore().load(self.cache_dir)


def lazy_schedule_combinations(arsenic, chlorine, strategy: str):
    """Combine sources without exhausting either input as a prerequisite."""
    if strategy in ("zip", "broadcast"):
        yield from zip(arsenic, chlorine)
        return
    if strategy != "cartesian": raise ValueError(f"unsupported schedule pairing: {strategy}")
    left, right, left_done, right_done = [], [], False, False
    diagonal = 0
    while True:
        while len(left) <= diagonal and not left_done:
            try: left.append(next(arsenic))
            except StopIteration: left_done = True
        while len(right) <= diagonal and not right_done:
            try: right.append(next(chlorine))
            except StopIteration: right_done = True
        yielded = False
        for left_index in range(diagonal + 1):
            right_index = diagonal - left_index
            if left_index < len(left) and right_index < len(right):
                yielded = True
                yield left[left_index], right[right_index]
        if left_done and right_done and diagonal >= len(left) + len(right) - 2: return
        if (left_done and not left) or (right_done and not right): return
        if not yielded and left_done and right_done: return
        diagonal += 1


class ForecastTrajectoryResolver:
    def __init__(self, dataset: ResolvedDataset | ResolvedDatasetSplit, spec: ForecastTrajectoryDataSpec, backend,
                 hydraulic_catalog=None):
        self.dataset, self.spec, self.backend = dataset, spec, backend
        self.hydraulic_catalog = hydraulic_catalog
        self.store = TrajectoryStore()

    def references(self, split: str, *, workers: int = 1, description: str | None = None):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._ensure_hydraulics(split, workers)
        prepared = list(self._prepared(split))
        missing = [item for item in prepared if not self._cache_is_valid(item[0])]
        label = description or f"{split} trajectories"
        if missing:
            progress = tqdm(total=len(missing), desc=label, unit="trajectory", dynamic_ncols=True)
            try:
                with _defer_sigint() as interrupted:
                    if workers == 1:
                        for reference, scenario, arsenic_schedule, chlorine_schedule in missing:
                            if interrupted(): break
                            self._materialize(reference, scenario, arsenic_schedule, chlorine_schedule)
                            progress.update()
                    else:
                        self._materialize_parallel(missing, workers, progress, interrupted)
                if interrupted(): raise KeyboardInterrupt
            finally:
                progress.close()
        progress = tqdm(total=len(prepared), desc=f"{label} cache validation",
                        unit="trajectory", dynamic_ncols=True)
        try:
            for reference, _scenario, _arsenic, _chlorine in prepared:
                self._validate_trajectory(reference.load(), reference)
                progress.update()
                yield reference
        finally:
            progress.close()

    def _prepared(self, split: str):
        self._ensure_hydraulics(split, 1)
        arsenic_spec = self.dataset.schedule_for(split)
        chlorine_spec = self.dataset.chlorine_schedule_for(split)
        for scenario in self.dataset.scenarios_for(split):
            if chlorine_spec is None:
                chlorine = iter((_ZeroChlorineSchedule(self.dataset.duration_seconds // self.dataset.hydraulic_timestep_seconds,
                                                       self.dataset.hydraulic_timestep_seconds),))
            else:
                chlorine = FixedChlorineInjectionScheduleIterator(
                    chlorine_spec, scenario.scenario_id,
                    step_count=self.dataset.duration_seconds // self.dataset.hydraulic_timestep_seconds,
                    timestep_seconds=self.dataset.hydraulic_timestep_seconds)
            arsenic = resolve_schedule_source(arsenic_spec, scenario.scenario_id, scenario.eligible_arsenic_node_ids,
                                              warmup_days=self.dataset.warmup_days, main_days=self.dataset.main_days,
                                              hydraulic_profile=(self.hydraulic_catalog.profile(scenario.scenario_id)
                                                  if self.hydraulic_catalog is not None else None))
            for arsenic_schedule, chlorine_schedule in lazy_schedule_combinations(iter(arsenic), iter(chlorine),
                                                                                   self.spec.schedule_pairing):
                identity = trajectory_cache_identity(self.dataset, scenario, arsenic_schedule.hash,
                                                     chlorine_schedule.hash, self.spec)
                cache_dir = Path(self.spec.cache_root) / self.dataset.network / scenario.scenario_id / identity
                reference = TrajectoryReference(identity, cache_dir, scenario.scenario_id,
                                                arsenic_schedule.hash, chlorine_schedule.hash)
                if not (cache_dir / "metadata.json").is_file() and not self.spec.materialize_missing:
                    raise FileNotFoundError(f"forecast trajectory cache is missing: {cache_dir}")
                yield reference, scenario, arsenic_schedule, chlorine_schedule

    def _ensure_hydraulics(self, split: str, workers: int) -> None:
        if self.hydraulic_catalog is not None:
            return
        backend = self.backend
        if backend is not None and not hasattr(backend, "create_session"):
            candidate = getattr(backend, "backend", None)
            backend = candidate if hasattr(candidate, "create_session") else None
        self.hydraulic_catalog = prepare_hydraulic_outflow_cache(
            self.dataset,
            backend,
            workers=workers,
            splits=(split,),
            description=f"{split} hydraulics",
        )

    def _resolve(self, scenario, arsenic_schedule, chlorine_schedule):
        identity = trajectory_cache_identity(self.dataset, scenario, arsenic_schedule.hash,
                                             chlorine_schedule.hash, self.spec)
        cache_dir = Path(self.spec.cache_root) / self.dataset.network / scenario.scenario_id / identity
        reference = TrajectoryReference(identity, cache_dir, scenario.scenario_id,
                                        arsenic_schedule.hash, chlorine_schedule.hash)
        if (cache_dir / "metadata.json").is_file():
            if self._cache_is_valid(reference):
                return reference
        if not self.spec.materialize_missing:
            raise FileNotFoundError(f"forecast trajectory cache is missing: {cache_dir}")
        self._materialize(reference, scenario, arsenic_schedule, chlorine_schedule)
        return reference

    def _materialize_parallel(self, missing, workers, progress, interrupted):
        worker_count = min(workers, len(missing))
        context = mp.get_context("spawn")
        pool = context.Pool(worker_count, initializer=_parallel_materialization_init,
                            initargs=(self.dataset, self.spec, self.backend))
        observed_workers = list(pool._pool)
        try:
            tasks = iter((index, scenario, arsenic, chlorine)
                         for index, (_reference, scenario, arsenic, chlorine) in enumerate(missing))
            pending = []
            for _ in range(worker_count):
                if interrupted(): break
                try: task = next(tasks)
                except StopIteration: break
                pending.append(pool.apply_async(_parallel_materialize, (task,)))
            completed = 0
            while pending:
                dead = next((worker for worker in observed_workers
                             if worker.exitcode not in (None, 0)), None)
                if dead is not None:
                    raise RuntimeError(
                        "forecast trajectory worker exited unexpectedly "
                        f"(exit code {dead.exitcode})"
                    )
                observed_pids = {worker.pid for worker in observed_workers}
                observed_workers.extend(
                    worker for worker in pool._pool
                    if worker.pid not in observed_pids
                )
                ready = next((result for result in pending if result.ready()), None)
                if ready is None:
                    time.sleep(.05)
                    continue
                ready.get()
                pending.remove(ready)
                completed += 1
                progress.update()
                if not interrupted():
                    try: task = next(tasks)
                    except StopIteration: pass
                    else: pending.append(pool.apply_async(_parallel_materialize, (task,)))
            if not interrupted() and completed != len(missing):
                raise RuntimeError("forecast trajectory workers returned incomplete results")
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()

    def _materialize(self, reference, scenario, arsenic_schedule, chlorine_schedule):
        reference.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = reference.cache_dir / ".materializing"
        owner = False
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor); owner = True
        except FileExistsError:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (reference.cache_dir / "metadata.json").is_file():
                    if self._cache_is_valid(reference): return
                time.sleep(.01)
            raise TimeoutError(f"timed out waiting for trajectory publication: {reference.identity}")
        try:
            if self._cache_is_valid(reference): return
            # Metadata is the commit marker. Remove only an invalid marker;
            # the following atomic save replaces any stale array archive.
            (reference.cache_dir / "metadata.json").unlink(missing_ok=True)
            if hasattr(self.backend, "simulate"):
                trajectory = self.backend.simulate(scenario, arsenic_schedule, chlorine_schedule,
                                                   hydraulic_timestep_seconds=self.dataset.hydraulic_timestep_seconds,
                                                   simulator_parameters=dict(self.spec.simulator_parameters))
            else:
                trajectory = self.backend(scenario, arsenic_schedule, chlorine_schedule,
                                          self.dataset.hydraulic_timestep_seconds,
                                          dict(self.spec.simulator_parameters))
            if not isinstance(trajectory, Trajectory): raise TypeError("forecast backend must return Trajectory")
            self._validate_trajectory(trajectory, reference)
            self.store.save(reference.cache_dir, trajectory)
        finally:
            if owner: lock.unlink(missing_ok=True)

    @staticmethod
    def _validate_identity(trajectory, reference):
        if (trajectory.scenario_id != reference.scenario_id or
                trajectory.arsenic_schedule_hash != reference.arsenic_schedule_hash or
                trajectory.chlorine_profile_hash != reference.chlorine_schedule_hash):
            raise ValueError("trajectory content collides with its cache identity")

    def _validate_trajectory(self, trajectory, reference):
        self._validate_identity(trajectory, reference)
        expected = np.arange(
            0,
            self.dataset.duration_seconds + self.dataset.hydraulic_timestep_seconds,
            self.dataset.hydraulic_timestep_seconds,
            dtype=np.int64,
        )
        actual = np.asarray(trajectory.time_seconds)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"trajectory {trajectory.scenario_id} does not cover the configured "
                f"duration ({len(actual)}/{len(expected)} samples)"
            )

    def _cache_is_valid(self, reference):
        if not (reference.cache_dir / "metadata.json").is_file():
            return False
        try:
            self._validate_trajectory(reference.load(), reference)
            return True
        except (EOFError, KeyError, OSError, TypeError, ValueError):
            return False


def trajectory_cache_identity(dataset: ResolvedDataset | ResolvedDatasetSplit, scenario: ScenarioEntry,
                              arsenic_hash: str, chlorine_hash: str,
                              spec: ForecastTrajectoryDataSpec) -> str:
    payload = {
        "trajectory_contract_version": 4,
        "network": dataset.network,
        "scenario": {"id": scenario.scenario_id, "content_hash": scenario.content_hash,
                     "calendar_start_day": scenario.calendar_start_day,
                     "chlorine_source_ids": scenario.chlorine_source_ids,
                     "monitored_node_ids": scenario.monitored_node_ids,
                     "eligible_arsenic_node_ids": scenario.eligible_arsenic_node_ids},
        "hydraulic_timestep_seconds": dataset.hydraulic_timestep_seconds,
        "duration_seconds": dataset.duration_seconds,
        "arsenic_schedule_hash": arsenic_hash, "chlorine_schedule_hash": chlorine_hash,
        "backend": spec.backend, "simulator_parameters": dict(spec.simulator_parameters),
    }
    return hashlib.sha256(json.dumps(canonical_data(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class _ZeroChlorineSchedule:
    step_count: int
    timestep_seconds: int
    @property
    def values(self): return (0.0,) * self.step_count
    @property
    def hash(self):
        return hashlib.sha256(f"zero:{self.step_count}:{self.timestep_seconds}".encode()).hexdigest()[:16]


_PARALLEL_RESOLVER = None


@contextmanager
def _defer_sigint():
    """Let active materializations publish before delivering KeyboardInterrupt."""
    requested = False

    def request_stop(_signum, _frame):
        nonlocal requested
        requested = True

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        yield lambda: requested
    finally:
        signal.signal(signal.SIGINT, previous)


def _parallel_materialization_init(dataset, spec, backend):
    global _PARALLEL_RESOLVER
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from ..simulation.native_temp import isolate_native_worker_temp
    isolate_native_worker_temp()
    _PARALLEL_RESOLVER = ForecastTrajectoryResolver(dataset, spec, backend)


def _parallel_materialize(task):
    index, scenario, arsenic_schedule, chlorine_schedule = task
    if _PARALLEL_RESOLVER is None:
        raise RuntimeError("forecast trajectory worker was not initialized")
    _PARALLEL_RESOLVER._resolve(scenario, arsenic_schedule, chlorine_schedule)
    return index
