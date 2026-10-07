"""Registered forecasting feature and target providers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..serialization import canonical_data
from ..simulation import FLOW_TO_LITERS_PER_MINUTE, Trajectory
from ..specs import ForecastFeatureSpec, ForecastInputSpec
from ..time import cyclic_time_features


DERIVED_ARSENIC_VERSION = 2
_Provider = Callable[[Trajectory, ForecastFeatureSpec], tuple[np.ndarray, tuple[str, ...]]]
_PROVIDERS: dict[str, _Provider] = {}


def register_feature_provider(name: str, provider: _Provider, *, replace: bool = False) -> None:
    if not name or (name in _PROVIDERS and not replace):
        raise ValueError(f"forecast feature provider already registered or invalid: {name!r}")
    _PROVIDERS[name] = provider


def feature_provider(name: str) -> _Provider:
    try: return _PROVIDERS[name]
    except KeyError as exc: raise ValueError(f"unknown forecast feature provider: {name}") from exc


@dataclass(frozen=True)
class FeatureTensor:
    values: np.ndarray
    names: tuple[str, ...]
    expansions: tuple[tuple[str, tuple[str, ...]], ...]
    transforms: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ForecastInputTensors:
    history: FeatureTensor
    future_known: FeatureTensor


class RegisteredForecastFeatureBuilder:
    def __init__(self, spec: ForecastInputSpec): self.spec = spec

    def build(self, trajectory: Trajectory) -> ForecastInputTensors:
        return ForecastInputTensors(self._build(self.spec.history, trajectory),
                                    self._build(self.spec.future_known, trajectory))

    @staticmethod
    def _build(specs, trajectory):
        parts, names, expansions, transforms = [], [], [], []
        for spec in specs:
            values, expanded = feature_provider(spec.name)(trajectory, spec)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim == 1: values = values[:, None]
            if values.shape != (len(trajectory.time_seconds), len(expanded)):
                raise ValueError(f"feature provider {spec.name!r} returned an invalid shape")
            values = apply_transform(values, spec.transform, spec.scale)
            parts.append(values); names.extend(expanded); expansions.append((spec.name, expanded))
            transforms.extend({"feature": name, "transform": spec.transform, "scale": spec.scale} for name in expanded)
        rows = len(trajectory.time_seconds)
        result = np.concatenate(parts, axis=1) if parts else np.empty((rows, 0), dtype=np.float32)
        return FeatureTensor(result.astype(np.float32), tuple(names), tuple(expansions), tuple(transforms))


def apply_transform(values, transform: str, scale: float | None = None):
    values = np.asarray(values)
    if transform == "identity": return values
    if transform == "log1p":
        if np.any(values < 0): raise ValueError("log1p feature transform requires nonnegative values")
        return np.log1p(values)
    if transform == "divide-by":
        if scale is None or scale <= 0: raise ValueError("divide-by feature transform requires a positive scale")
        return values / scale
    raise ValueError(f"unsupported feature transform: {transform}")


def inverse_transform(values, transform: str, scale: float | None = None):
    if transform == "identity": return np.asarray(values)
    if transform == "log1p": return np.expm1(values)
    if transform == "divide-by": return np.asarray(values) * float(scale)
    raise ValueError(f"unsupported feature transform: {transform}")


def _field(attribute: str, prefix: str, order: str):
    def provider(trajectory, spec):
        values = np.asarray(getattr(trajectory, attribute))
        ids = getattr(trajectory, order)
        expected_scope = {"node_ids": "node", "link_ids": "link", "source_ids": "source"}[order]
        if spec.scope != expected_scope:
            raise ValueError(f"feature {spec.name!r} requires {expected_scope} scope")
        return values, tuple(f"{prefix}:{value}" for value in ids)
    return provider


def _injection(trajectory, spec):
    if spec.scope not in ("source", "global"): raise ValueError("chlorine-injection requires source or global scope")
    values = np.asarray(trajectory.injection)[:, None]
    if spec.scope == "source" and len(trajectory.source_ids) > 1:
        if values.shape[1] != len(trajectory.source_ids):
            raise ValueError("scalar trajectory injection is incompatible with multiple source-scoped outputs")
    names = tuple(f"chlorine-injection:{item}" for item in trajectory.source_ids) if spec.scope == "source" else ("chlorine-injection",)
    return values, names


def _cyclic(trajectory, spec):
    if spec.scope != "global": raise ValueError("cyclic-time requires global scope")
    all_values = cyclic_time_features(trajectory.time_seconds, trajectory.calendar_start_day)
    indexes = {"day": (0, 1), "week": (2, 3), "year": (4, 5)}
    selected = tuple(index for cycle in spec.cycles for index in indexes[cycle])
    names = tuple(f"{cycle}_{part}" for cycle in spec.cycles for part in ("sin", "cos"))
    return all_values[:, selected], names


def _local_arsenic(trajectory, spec):
    if spec.scope != "node": raise ValueError("local-injected-arsenic requires node scope")
    parameters = spec.parameters_dict()
    outflow = _require_hydraulic_outflow(trajectory)
    transported, _hydraulic_local = derive_transported_arsenic(
        trajectory.arsenic, trajectory.event_strength_by_node, outflow, trajectory.time_seconds,
        flow_units=trajectory.flow_units or str(parameters.get("flow_units", "LPS")),
    )
    local = np.clip(np.asarray(trajectory.arsenic) - transported, 0.0, None)
    return local, tuple(f"local-injected-arsenic:{n}" for n in trajectory.node_ids)


def _transported(trajectory, spec):
    if spec.scope != "node": raise ValueError("transported-arsenic requires node scope")
    parameters = spec.parameters_dict()
    outflow = _require_hydraulic_outflow(trajectory)
    values, _local = derive_transported_arsenic(
        trajectory.arsenic, trajectory.event_strength_by_node, outflow,
        trajectory.time_seconds, flow_units=trajectory.flow_units or str(parameters.get("flow_units", "LPS")),
    )
    return values, tuple(f"transported-arsenic:{n}" for n in trajectory.node_ids)


def _require_hydraulic_outflow(trajectory: Trajectory) -> np.ndarray:
    if trajectory.hydraulic_outflow is None:
        raise ValueError(
            "transported arsenic requires canonical hydraulic_outflow; "
            "rematerialize this trajectory with a compatible simulator backend"
        )
    return np.asarray(trajectory.hydraulic_outflow)


def previous_sample_times(time_seconds) -> np.ndarray:
    times = np.asarray(time_seconds)
    if times.ndim != 1 or not len(times): raise ValueError("time_seconds must be a non-empty vector")
    return np.concatenate((times[:1], times[:-1]))


def derive_transported_arsenic(raw_arsenic, event_strength_by_node, hydraulic_outflow,
                                time_seconds, *, flow_units: str = "LPS", tolerance: float = 1e-9):
    """Return transported and local arsenic using report-end alignment.

    Strength is MASS units/minute and outflow is converted to liters/minute.
    For isolated source-only reports the raw source value is authoritative.
    """
    raw = np.asarray(raw_arsenic, dtype=np.float64)
    strengths = np.asarray(event_strength_by_node, dtype=np.float64)
    outflow = np.asarray(hydraulic_outflow, dtype=np.float64)
    if raw.ndim != 2 or strengths.shape != raw.shape or outflow.shape != raw.shape:
        raise ValueError("arsenic, event strengths, and hydraulic outflow must share (time, node) shape")
    previous_sample_times(time_seconds)  # validates and documents alignment
    try: conversion = FLOW_TO_LITERS_PER_MINUTE[flow_units.upper()]
    except KeyError as exc: raise ValueError(f"unsupported EPANET flow units: {flow_units}") from exc
    aligned_strength = np.zeros_like(strengths); aligned_flow = np.zeros_like(outflow)
    aligned_strength[1:] = strengths[:-1]; aligned_flow[1:] = outflow[:-1]
    local = np.divide(aligned_strength, aligned_flow * conversion, out=np.zeros_like(raw),
                      where=aligned_flow > tolerance)
    # Match the authoritative offline correction: replace the estimate with
    # raw source readings only when a complete contiguous event interval is
    # isolated (no concurrent event and no arsenic at another monitored node).
    for source in range(raw.shape[1]):
        active = aligned_strength[:, source] > tolerance
        changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        for start, end in zip(np.flatnonzero(changes == 1),
                              np.flatnonzero(changes == -1), strict=True):
            other_sources = np.delete(aligned_strength[start:end], source, axis=1)
            other_raw = np.delete(raw[start:end], source, axis=1)
            if ((not other_sources.size or np.max(other_sources) <= tolerance) and
                    (not other_raw.size or np.max(other_raw) <= tolerance)):
                local[start:end, source] = raw[start:end, source]
    return np.clip(raw - local, 0.0, None).astype(np.float32), local.astype(np.float32)


def canonical_trajectory_identity(trajectory: Trajectory) -> str:
    payload = {"network": trajectory.network, "scenario": trajectory.scenario_id,
               "arsenic": trajectory.arsenic_schedule_hash, "chlorine": trajectory.chlorine_profile_hash,
               "backend": trajectory.simulator_backend, "simulator": trajectory.simulator_config_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class DerivedFeatureCache:
    def __init__(self, root: Path): self.root = Path(root)
    def path(self, trajectory: Trajectory, name: str, version: int) -> Path:
        return self.root / canonical_trajectory_identity(trajectory) / f"{name}-v{version}.npz"
    def load_or_compute(self, trajectory, name, version, compute):
        path = self.path(trajectory, name, version)
        if path.is_file():
            with np.load(path, allow_pickle=False) as data: return np.asarray(data["values"])
        values = np.asarray(compute())
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{id(values)}.tmp.npz")
        np.savez_compressed(temporary, values=values)
        temporary.replace(path)
        return values


register_feature_provider("chlorine", _field("chlorine", "chlorine", "node_ids"))
register_feature_provider("pressure", _field("pressure", "pressure", "node_ids"))
register_feature_provider("raw-demand", _field("demand", "raw-demand", "node_ids"))
register_feature_provider("link-flow", _field("flow", "link-flow", "link_ids"))
register_feature_provider("raw-arsenic", _field("arsenic", "raw-arsenic", "node_ids"))
register_feature_provider("chlorine-injection", _injection)
register_feature_provider("cyclic-time", _cyclic)
register_feature_provider("local-injected-arsenic", _local_arsenic)
register_feature_provider("transported-arsenic", _transported)
