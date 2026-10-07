"""Ordered forecast windows and train-only normalization."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..progress import tqdm
from ..simulation import Trajectory
from ..specs import ForecastHeadSpec, ForecastInputSpec
from .features import RegisteredForecastFeatureBuilder
from .features import apply_transform
from .targets import extract_target, target_window


@dataclass(frozen=True)
class NamedWindows:
    history: np.ndarray
    future_known: np.ndarray
    targets: dict[str, np.ndarray]
    previous_levels: dict[str, np.ndarray]
    feature_contract: dict[str, object]


@dataclass(frozen=True)
class NamedNormalization:
    history_mean: np.ndarray
    history_std: np.ndarray
    future_mean: np.ndarray
    future_std: np.ndarray
    targets: dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class _PreparedTrajectory:
    trajectory: Trajectory
    history: np.ndarray
    future_known: np.ndarray
    targets: dict[str, np.ndarray]
    auxiliary_history: np.ndarray | None = None
    auxiliary_future_known: np.ndarray | None = None


class ForecastWindowDataset(Dataset):
    """A bounded-memory view over all forecast windows in trajectory stores.

    Sources may be canonical ``TrajectoryReference`` objects or in-memory
    trajectories. Only a small LRU of decoded and feature-expanded trajectories
    is retained; individual normalized windows are created on demand.
    """

    def __init__(self, sources, inputs: ForecastInputSpec, heads: tuple[ForecastHeadSpec, ...],
                 history_steps: int, horizon_steps: int,
                 normalization: NamedNormalization | None = None, *,
                 include_future_histories: bool = False,
                 auxiliary_inputs: ForecastInputSpec | None = None,
                 auxiliary_normalization: NamedNormalization | None = None,
                 cache_size: int = 8, description: str | None = None):
        if not sources: raise ValueError("forecast window dataset requires trajectories")
        if cache_size < 1: raise ValueError("forecast trajectory cache size must be positive")
        self.sources = tuple(sources); self.inputs = inputs; self.heads = heads
        self.history_steps = history_steps; self.horizon_steps = horizon_steps
        self.normalization = normalization
        self.include_future_histories = include_future_histories
        self.auxiliary_inputs = auxiliary_inputs
        self.auxiliary_normalization = auxiliary_normalization
        self.cache_size = cache_size; self._cache: OrderedDict[int, _PreparedTrajectory] = OrderedDict()
        self.window_counts = []
        self.scenario_ids = []
        contract = None; node_ids = link_ids = source_ids = None
        progress = tqdm(self.sources, desc=description, unit="trajectory", dynamic_ncols=True,
                        disable=description is None)
        try:
            for index, source in enumerate(progress):
                trajectory = _load_source(source)
                count = len(trajectory.time_seconds) - history_steps - horizon_steps + 1
                if count <= 0: raise ValueError(f"trajectory {trajectory.scenario_id} is too short")
                self.window_counts.append(count); self.scenario_ids.append(trajectory.scenario_id)
                identity = (trajectory.node_ids, trajectory.link_ids, trajectory.source_ids)
                if index == 0:
                    node_ids, link_ids, source_ids = identity
                    features = RegisteredForecastFeatureBuilder(inputs).build(trajectory)
                    contract = _feature_contract(features)
                    self.history_size = features.history.values.shape[1]
                    self.future_known_size = features.future_known.values.shape[1]
                    self.head_sizes = {head.name: extract_target(trajectory, head).shape[1] for head in heads}
                elif identity != (node_ids, link_ids, source_ids):
                    raise ValueError("forecast trajectories have incompatible node/link/source order")
        finally:
            progress.close()
        self.node_ids = node_ids or (); self.link_ids = link_ids or (); self.source_ids = source_ids or ()
        self.feature_contract = contract or {}
        self._starts = np.concatenate((np.asarray([0], dtype=np.int64),
                                       np.cumsum(self.window_counts, dtype=np.int64)))

    def __len__(self): return int(self._starts[-1])

    def __getitem__(self, index):
        trajectory_index, offset = self.locate(index)
        item = self._prepared(trajectory_index)
        start = offset + self.history_steps; end = start + self.horizon_steps
        history = item.history[offset:start]; future = item.future_known[start:end]
        targets, previous = {}, {}
        for head in self.heads:
            physical = item.targets[head.name]
            represented = physical[start:end] if head.prediction == "level" else physical[start:end] - physical[start - 1:end - 1]
            represented = apply_transform(represented, head.transform, head.scale).astype(np.float32)
            targets[head.name] = represented; previous[head.name] = physical[start - 1].astype(np.float32)
        if self.normalization is not None:
            history = _normalize(history, self.normalization.history_mean, self.normalization.history_std)
            future = _normalize(future, self.normalization.future_mean, self.normalization.future_std)
            targets = {name: _normalize(value, *self.normalization.targets[name])
                       for name, value in targets.items()}
        result = {"history": torch.from_numpy(np.asarray(history, dtype=np.float32)),
                  "future_known": torch.from_numpy(np.asarray(future, dtype=np.float32)),
                  "targets": {name: torch.from_numpy(value) for name, value in targets.items()},
                  "previous_levels": {name: torch.from_numpy(value) for name, value in previous.items()},
                  "trajectory_index": torch.tensor(trajectory_index, dtype=torch.int64)}
        if self.include_future_histories:
            future_histories = np.stack([
                item.history[offset + step + 1:start + step + 1]
                for step in range(self.horizon_steps)
            ])
            if self.normalization is not None:
                future_histories = _normalize(
                    future_histories, self.normalization.history_mean, self.normalization.history_std)
            result["future_histories"] = torch.from_numpy(
                np.asarray(future_histories, dtype=np.float32))
        if item.auxiliary_history is not None and item.auxiliary_future_known is not None:
            auxiliary_history = item.auxiliary_history[offset:start]
            auxiliary_future = item.auxiliary_future_known[start:end]
            if self.auxiliary_normalization is not None:
                auxiliary_history = _normalize(auxiliary_history, self.auxiliary_normalization.history_mean,
                                               self.auxiliary_normalization.history_std)
                auxiliary_future = _normalize(auxiliary_future, self.auxiliary_normalization.future_mean,
                                              self.auxiliary_normalization.future_std)
            result["auxiliary_history"] = torch.from_numpy(np.asarray(auxiliary_history, dtype=np.float32))
            result["auxiliary_future_known"] = torch.from_numpy(np.asarray(auxiliary_future, dtype=np.float32))
        return result

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0: index += len(self)
        if index < 0 or index >= len(self): raise IndexError(index)
        trajectory_index = bisect.bisect_right(self._starts, index) - 1
        return trajectory_index, int(index - self._starts[trajectory_index])

    def iter_prepared(self) -> Iterator[_PreparedTrajectory]:
        for index in range(len(self.sources)): yield self._prepare(index)

    def _prepared(self, index: int) -> _PreparedTrajectory:
        if index in self._cache:
            value = self._cache.pop(index); self._cache[index] = value; return value
        value = self._prepare(index); self._cache[index] = value
        while len(self._cache) > self.cache_size: self._cache.popitem(last=False)
        return value

    def _prepare(self, index: int) -> _PreparedTrajectory:
        trajectory = _load_source(self.sources[index])
        features = RegisteredForecastFeatureBuilder(self.inputs).build(trajectory)
        if _feature_contract(features) != self.feature_contract:
            raise ValueError("forecast trajectories produced incompatible feature contracts")
        auxiliary_history = auxiliary_future = None
        if self.auxiliary_inputs is not None:
            auxiliary = RegisteredForecastFeatureBuilder(self.auxiliary_inputs).build(trajectory)
            auxiliary_history, auxiliary_future = auxiliary.history.values, auxiliary.future_known.values
        return _PreparedTrajectory(trajectory, features.history.values, features.future_known.values,
                                   {head.name: extract_target(trajectory, head) for head in self.heads},
                                   auxiliary_history, auxiliary_future)


class TrajectoryBatchSampler(Sampler[list[int]]):
    """Shuffle deterministically while keeping trajectory reads local."""

    def __init__(self, dataset: ForecastWindowDataset, batch_size: int, seed: int):
        self.dataset = dataset; self.batch_size = batch_size; self.seed = seed; self.epoch = 0

    def __len__(self): return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.dataset.window_counts))
        pending = []
        for trajectory_index in order:
            count = self.dataset.window_counts[int(trajectory_index)]
            offsets = rng.permutation(count)
            base = int(self.dataset._starts[int(trajectory_index)])
            indexes = (base + offsets).tolist()
            if pending:
                needed = self.batch_size - len(pending)
                pending.extend(indexes[:needed]); indexes = indexes[needed:]
                if len(pending) == self.batch_size: yield pending; pending = []
            complete = len(indexes) - len(indexes) % self.batch_size
            for start in range(0, complete, self.batch_size):
                yield indexes[start:start + self.batch_size]
            pending.extend(indexes[complete:])
        if pending: yield pending


def fit_streaming_normalization(dataset: ForecastWindowDataset, *, independently: bool,
                                input_mode: str = "fitted",
                                description: str | None = None) -> NamedNormalization:
    """Fit exact window-weighted statistics without constructing any windows."""
    if input_mode not in ("fitted", "fixed"):
        raise ValueError("forecast input normalization must be 'fitted' or 'fixed'")
    history = _Sums(); future = _Sums(); targets = {head.name: _Sums() for head in dataset.heads}
    prepared = tqdm(dataset.iter_prepared(), total=len(dataset.sources), desc=description,
                    unit="trajectory", dynamic_ncols=True, disable=description is None)
    try:
        for item, count in zip(prepared, dataset.window_counts, strict=True):
            history.add(item.history, _range_weights(len(item.history), count, 0, dataset.history_steps))
            future_weights = _range_weights(len(item.future_known), count, dataset.history_steps,
                                            dataset.horizon_steps)
            future.add(item.future_known, future_weights)
            for head in dataset.heads:
                physical = item.targets[head.name]
                if head.prediction == "level": represented = physical
                else:
                    represented = np.zeros_like(physical)
                    represented[1:] = physical[1:] - physical[:-1]
                represented = apply_transform(represented, head.transform, head.scale)
                targets[head.name].add(represented, future_weights)
    finally:
        prepared.close()
    if independently:
        target_stats = {name: value.finish() for name, value in targets.items()}
    else:
        widths = {value.total.shape[0] for value in targets.values() if value.total is not None}
        if len(widths) != 1: raise ValueError("shared target normalization requires equal head widths")
        shared = _Sums()
        for value in targets.values(): shared.merge(value)
        target_stats = {name: shared.finish() for name in targets}
    history_mean, history_std = history.finish()
    future_mean, future_std = future.finish()
    if input_mode == "fixed":
        history_mean = np.zeros_like(history_mean)
        history_std = np.ones_like(history_std)
        future_mean = np.zeros_like(future_mean)
        future_std = np.ones_like(future_std)
    return NamedNormalization(
        history_mean, history_std, future_mean, future_std, target_stats
    )


class _Sums:
    def __init__(self): self.total = None; self.squares = None; self.count = 0.0
    def add(self, values, weights):
        values = np.asarray(values, dtype=np.float64); weights = np.asarray(weights, dtype=np.float64)
        selected = weights > 0; values = values[selected]; weights = weights[selected, None]
        total = np.sum(values * weights, axis=0); squares = np.sum(np.square(values) * weights, axis=0)
        self.total = total if self.total is None else self.total + total
        self.squares = squares if self.squares is None else self.squares + squares
        self.count += float(weights.sum())
    def merge(self, other):
        self.total = other.total.copy() if self.total is None else self.total + other.total
        self.squares = other.squares.copy() if self.squares is None else self.squares + other.squares
        self.count += other.count
    def finish(self):
        if self.total is None or self.squares is None or not self.count: raise ValueError("normalization input is empty")
        mean = self.total / self.count
        variance = np.maximum(self.squares / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return mean.astype(np.float32), np.where(std < 1e-8, 1.0, std).astype(np.float32)


def _range_weights(length: int, count: int, start: int, width: int):
    difference = np.zeros(length + 1, dtype=np.int64)
    offsets = np.arange(count, dtype=np.int64)
    np.add.at(difference, offsets + start, 1); np.add.at(difference, offsets + start + width, -1)
    return np.cumsum(difference[:-1])


def _load_source(source: Any) -> Trajectory:
    value = source.load() if hasattr(source, "load") else source
    if not isinstance(value, Trajectory): raise TypeError("forecast trajectory source did not load a Trajectory")
    return value


def _feature_contract(features):
    return {"history_names": features.history.names, "future_known_names": features.future_known.names,
            "history_expansions": features.history.expansions, "future_known_expansions": features.future_known.expansions,
            "history_transforms": features.history.transforms, "future_known_transforms": features.future_known.transforms}


def _normalize(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def build_named_windows(trajectories: list[Trajectory], inputs: ForecastInputSpec,
                        heads: tuple[ForecastHeadSpec, ...], history_steps: int,
                        horizon_steps: int) -> NamedWindows:
    builder = RegisteredForecastFeatureBuilder(inputs)
    histories, futures = [], []
    targets = {head.name: [] for head in heads}; previous = {head.name: [] for head in heads}
    contract = None
    for trajectory in trajectories:
        features = builder.build(trajectory)
        current = {
            "history_names": features.history.names, "future_known_names": features.future_known.names,
            "history_expansions": features.history.expansions, "future_known_expansions": features.future_known.expansions,
            "history_transforms": features.history.transforms, "future_known_transforms": features.future_known.transforms,
        }
        if contract is None: contract = current
        elif current != contract: raise ValueError("forecast trajectories produced incompatible feature contracts")
        count = len(trajectory.time_seconds) - history_steps - horizon_steps + 1
        if count <= 0: raise ValueError(f"trajectory {trajectory.scenario_id} is too short")
        for offset in range(count):
            start, end = offset + history_steps, offset + history_steps + horizon_steps
            histories.append(features.history.values[offset:start]); futures.append(features.future_known.values[start:end])
            for head in heads:
                window = target_window(trajectory, head, start, end)
                targets[head.name].append(window.values); previous[head.name].append(window.previous_level)
    return NamedWindows(np.asarray(histories, dtype=np.float32), np.asarray(futures, dtype=np.float32),
                        {key: np.asarray(value, dtype=np.float32) for key, value in targets.items()},
                        {key: np.asarray(value, dtype=np.float32) for key, value in previous.items()}, contract or {})


def fit_named_normalization(windows: NamedWindows, *, independently: bool) -> NamedNormalization:
    if independently:
        target_stats = {name: _mean_std(values, axis=(0, 1)) for name, values in windows.targets.items()}
    else:
        widths = {values.shape[-1] for values in windows.targets.values()}
        if len(widths) != 1: raise ValueError("shared target normalization requires equal head widths")
        shared = _mean_std(np.concatenate(tuple(windows.targets.values()), axis=0), axis=(0, 1))
        target_stats = {name: shared for name in windows.targets}
    return NamedNormalization(*_mean_std(windows.history, axis=(0, 1)),
                              *_mean_std(windows.future_known, axis=(0, 1)), target_stats)


def normalize_named_windows(windows: NamedWindows, stats: NamedNormalization):
    return {
        "history": ((windows.history - stats.history_mean) / stats.history_std).astype(np.float32),
        "future_known": ((windows.future_known - stats.future_mean) / stats.future_std).astype(np.float32),
        "targets": {name: ((values - stats.targets[name][0]) / stats.targets[name][1]).astype(np.float32)
                    for name, values in windows.targets.items()},
        "previous_levels": windows.previous_levels,
    }


def _mean_std(values: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=axis, dtype=np.float64).astype(np.float32)
    std = values.std(axis=axis, dtype=np.float64).astype(np.float32)
    return mean, np.where(std < 1e-8, 1.0, std).astype(np.float32)
