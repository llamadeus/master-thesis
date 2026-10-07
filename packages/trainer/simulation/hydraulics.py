"""Scenario-level hydraulic outflow caching for calibrated arsenic schedules."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..data import ResolvedDataset, ResolvedDatasetSplit
from ..progress import tqdm
from ..schedules.arsenic import schedule_source_requires_hydraulics
from .hydraulic_cache import DEFAULT_HYDRAULIC_CACHE_ROOT, HydraulicCache, HydraulicCacheReference

FLOW_TO_LITERS_PER_MINUTE = {
    "LPS": 60.0,
    "LPM": 1.0,
    "GPM": 3.785411784,
    "CFS": 1699.01082,
    "CMH": 1000.0 / 60.0,
    "CMD": 1000.0 / 1440.0,
    "MGD": 1_000_000.0 * 3.785411784 / 1440.0,
    "IMGD": 1_000_000.0 * 4.54609 / 1440.0,
    "AFD": 1_233_481.83754752 / 1440.0,
    "MLD": 1_000_000.0 / 1440.0,
    "CMS": 60_000.0,
}


@dataclass(frozen=True)
class HydraulicOutflowProfile:
    identity: str
    scenario_id: str
    time_seconds: np.ndarray
    node_ids: tuple[str, ...]
    outflow_liters_per_minute: np.ndarray

    def minimum_outflow_liters_per_minute(
        self,
        node_id: str,
        start_time_seconds: int,
        end_time_seconds: int,
    ) -> float:
        minimum, _maximum = self.outflow_range_liters_per_minute(
            node_id, start_time_seconds, end_time_seconds
        )
        return minimum

    def outflow_range_liters_per_minute(
        self,
        node_id: str,
        start_time_seconds: int,
        end_time_seconds: int,
    ) -> tuple[float, float]:
        if end_time_seconds <= start_time_seconds:
            raise ValueError("arsenic event end must follow its start")
        try:
            column = self.node_ids.index(str(node_id))
        except ValueError as exc:
            raise ValueError(
                f"hydraulic profile {self.scenario_id} does not contain event node {node_id}"
            ) from exc
        mask = (
            (self.time_seconds >= start_time_seconds)
            & (self.time_seconds < end_time_seconds)
        )
        if not np.any(mask):
            raise ValueError(
                f"hydraulic profile {self.scenario_id} has no samples in the event window"
            )
        values = self.outflow_liters_per_minute[mask, column]
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"hydraulic profile {self.scenario_id} has nonpositive or nonfinite "
                f"outflow for event node {node_id}"
            )
        return float(np.min(values)), float(np.max(values))


@dataclass(frozen=True)
class HydraulicOutflowReference:
    identity: str
    scenario_id: str
    cache_dir: Path
    source: HydraulicCacheReference
    node_ids: tuple[str, ...]

    def load(self) -> HydraulicOutflowProfile:
        from .epyt import EpytFlowBackend
        return EpytFlowBackend.load_hydraulic_outflow(self.source, self.node_ids)


class HydraulicOutflowCatalog:
    """Pickle-safe, lazy scenario-to-profile lookup."""

    def __init__(self, references: Iterable[HydraulicOutflowReference]):
        values = tuple(references)
        self.references = {item.scenario_id: item for item in values}
        if len(self.references) != len(values):
            raise ValueError("hydraulic catalog contains duplicate scenarios")
        payload = [(key, self.references[key].identity) for key in sorted(self.references)]
        self.catalog_hash = hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self._profiles: dict[str, HydraulicOutflowProfile] = {}

    def profile(self, scenario_id: str) -> HydraulicOutflowProfile:
        try:
            reference = self.references[scenario_id]
        except KeyError as exc:
            raise ValueError(
                f"hydraulic catalog does not contain scenario {scenario_id}"
            ) from exc
        if scenario_id not in self._profiles:
            self._profiles[scenario_id] = reference.load()
        return self._profiles[scenario_id]

    def __getstate__(self):
        return {"references": self.references, "catalog_hash": self.catalog_hash}

    def __setstate__(self, state):
        self.references = state["references"]
        self.catalog_hash = state["catalog_hash"]
        self._profiles = {}


def prepare_hydraulic_outflow_cache(
    dataset: ResolvedDataset | ResolvedDatasetSplit,
    backend=None,
    *,
    workers: int = 1,
    splits: Sequence[str] = ("train", "evaluation", "test"),
    cache_root: Path | None = None,
    description: str = "hydraulic outflow",
) -> HydraulicOutflowCatalog | None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("hydraulic cache workers must be a positive integer")
    required = {
        scenario.scenario_id: scenario
        for split in splits
        if schedule_source_requires_hydraulics(dataset.schedule_for(split))
        for scenario in dataset.scenarios_for(split)
    }
    if not required:
        return None
    if backend is None:
        from .epyt import EpytFlowBackend
        backend = EpytFlowBackend()

    root = Path(cache_root if cache_root is not None else
                getattr(backend, "hydraulic_cache_root", None) or DEFAULT_HYDRAULIC_CACHE_ROOT).resolve()
    references = [
        _reference(dataset, required[key], root)
        for key in sorted(required)
    ]
    missing = [
        (reference, required[reference.scenario_id])
        for reference in references
        if not _reference_is_valid(reference)
    ]
    hits = len(references) - len(missing)
    print(
        f"{description}: {hits}/{len(references)} cache hits; "
        f"generating {len(missing)}"
    )
    if missing:
        progress = tqdm(
            total=len(missing), desc=description, unit="scenario", dynamic_ncols=True
        )
        try:
            if workers == 1:
                for reference, scenario in missing:
                    _materialize(dataset, backend, reference, scenario)
                    progress.update()
            else:
                _materialize_parallel(
                    dataset, backend, missing, workers, progress
                )
        finally:
            progress.close()

    validation = tqdm(
        references,
        total=len(references),
        desc=f"{description} cache validation",
        unit="scenario",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for reference in validation:
            profile = reference.load()
            scenario = required[reference.scenario_id]
            _validate_profile(profile, dataset, scenario)
    finally:
        validation.close()
    return HydraulicOutflowCatalog(references)


def _reference(dataset, scenario, root: Path) -> HydraulicOutflowReference:
    source = HydraulicCache(root).reference(scenario.path)
    contract = source.contract
    if (contract["network"] != dataset.network
            or contract["scenario_id"] != scenario.scenario_id
            or contract["hydraulic_timestep_seconds"] != dataset.hydraulic_timestep_seconds
            or contract["duration_seconds"] != dataset.duration_seconds
            or contract["calendar"]["start_day"] != scenario.calendar_start_day):
        raise ValueError(f"hydraulic source settings disagree with dataset: {scenario.scenario_id}")
    return HydraulicOutflowReference(source.identity, scenario.scenario_id, source.cache_dir,
                                     source, scenario.monitored_node_ids)


def _reference_is_valid(reference: HydraulicOutflowReference) -> bool:
    return reference.source.load() is not None


def _materialize(dataset, backend, reference, scenario) -> None:
    source = backend.prepare_hydraulics(scenario.path, cache_root=reference.cache_dir.parents[2])
    if source.identity != reference.identity:
        raise ValueError(f"hydraulic scenario changed during preparation: {scenario.scenario_id}")


def _validate_profile(profile, dataset, scenario) -> None:
    times = np.asarray(profile.time_seconds)
    values = np.asarray(profile.outflow_liters_per_minute)
    if profile.scenario_id != scenario.scenario_id:
        raise ValueError("hydraulic profile scenario identity mismatch")
    if profile.node_ids != scenario.monitored_node_ids:
        raise ValueError(
            f"hydraulic profile node order mismatch for {scenario.scenario_id}"
        )
    if not set(scenario.eligible_arsenic_node_ids).issubset(profile.node_ids):
        raise ValueError(
            f"eligible arsenic nodes are absent from hydraulic profile {scenario.scenario_id}"
        )
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("hydraulic profile timestamps must be increasing")
    expected = np.arange(
        0,
        dataset.duration_seconds + dataset.hydraulic_timestep_seconds,
        dataset.hydraulic_timestep_seconds,
        dtype=np.int64,
    )
    if not np.array_equal(times, expected):
        raise ValueError(
            f"hydraulic profile does not cover the configured duration for {scenario.scenario_id}"
        )
    if values.shape != (len(times), len(profile.node_ids)):
        raise ValueError("hydraulic profile outflow shape does not match node order")
    indexes = [profile.node_ids.index(node) for node in scenario.eligible_arsenic_node_ids]
    eligible = values[:, indexes]
    if not np.all(np.isfinite(eligible)) or np.any(eligible <= 0.0):
        raise ValueError(
            f"hydraulic profile has nonpositive or nonfinite eligible-node outflow: "
            f"{scenario.scenario_id}"
        )


_PARALLEL_STATE = None


def _materialize_parallel(dataset, backend, missing, workers, progress):
    worker_count = min(workers, len(missing))
    context = mp.get_context("spawn")
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        pool = context.Pool(
            worker_count,
            initializer=_parallel_init,
            initargs=(dataset, backend),
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    try:
        tasks = ((reference, scenario) for reference, scenario in missing)
        for _scenario_id in pool.imap_unordered(_parallel_materialize, tasks, chunksize=1):
            progress.update()
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def _parallel_init(dataset, backend):
    global _PARALLEL_STATE
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from .native_temp import isolate_native_worker_temp
    isolate_native_worker_temp()
    _PARALLEL_STATE = (dataset, backend)


def _parallel_materialize(task):
    if _PARALLEL_STATE is None:
        raise RuntimeError("hydraulic materialization worker was not initialized")
    reference, scenario = task
    dataset, backend = _PARALLEL_STATE
    _materialize(dataset, backend, reference, scenario)
    return scenario.scenario_id
