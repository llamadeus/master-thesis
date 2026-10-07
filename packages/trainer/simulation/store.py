"""Canonical trajectory.npz + metadata.json storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .core import Trajectory


TRAJECTORY_SCHEMA_VERSION = 1
_ARRAYS = (
    "time_seconds", "pressure", "demand", "flow", "chlorine", "arsenic",
    "injection", "actions", "event_active", "event_strength_by_node",
)
_OPTIONAL_ARRAYS = ("hydraulic_outflow",)


class TrajectoryStore:
    def save(self, directory: Path, trajectory: Trajectory) -> None:
        if (isinstance(trajectory.calendar_start_day,bool) or not isinstance(trajectory.calendar_start_day,int)
                or not 0<=trajectory.calendar_start_day<365):
            raise ValueError("trajectory calendar metadata is invalid")
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "response.json").exists():
            raise ValueError("trainer2 refuses cache directories containing legacy response.json")
        _validate(trajectory)
        array_names = _ARRAYS + tuple(
            name for name in _OPTIONAL_ARRAYS if getattr(trajectory, name) is not None
        )
        metadata = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "network": trajectory.network,
            "scenario_id": trajectory.scenario_id,
            "calendar_start_day": trajectory.calendar_start_day,
            "chlorine_profile_hash": trajectory.chlorine_profile_hash,
            "arsenic_schedule_hash": trajectory.arsenic_schedule_hash,
            "arsenic_schedule_seed": trajectory.arsenic_schedule_seed,
            "node_ids": list(trajectory.node_ids),
            "link_ids": list(trajectory.link_ids),
            "source_ids": list(trajectory.source_ids),
            "simulator_backend": trajectory.simulator_backend,
            "simulator_config_hash": trajectory.simulator_config_hash,
            "flow_units": trajectory.flow_units,
            "reproducibility": dict(trajectory.reproducibility),
            "arrays": {name: list(np.asarray(getattr(trajectory, name)).shape) for name in array_names},
        }
        # Metadata is the commit marker.  Publishing it last means readers can
        # never accept an entry whose array archive is only partially written.
        fd, raw_path = tempfile.mkstemp(prefix=".trajectory-", suffix=".npz", dir=directory)
        os.close(fd)
        temporary = Path(raw_path)
        metadata_temporary = directory / f".metadata-{os.getpid()}-{id(trajectory)}.json"
        try:
            np.savez_compressed(temporary, **{name: getattr(trajectory, name) for name in array_names})
            metadata_temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (directory / "metadata.json").unlink(missing_ok=True)
            os.replace(temporary, directory / "trajectory.npz")
            os.replace(metadata_temporary, directory / "metadata.json")
        finally:
            temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)

    def load(self, directory: Path) -> Trajectory:
        if (directory / "response.json").exists():
            raise ValueError("legacy response.json caches are not supported")
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trajectory metadata in {directory}") from exc
        if metadata.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema in {directory}")
        calendar_start_day = metadata.get("calendar_start_day")
        if (isinstance(calendar_start_day,bool) or not isinstance(calendar_start_day,int)
                or not 0<=calendar_start_day<365):
            raise ValueError(f"invalid trajectory calendar metadata in {directory}")
        with np.load(directory / "trajectory.npz", allow_pickle=False) as values:
            missing = set(_ARRAYS) - set(values.files)
            if missing:
                raise ValueError(f"trajectory is missing arrays: {sorted(missing)}")
            arrays = {name: np.asarray(values[name]) for name in _ARRAYS}
            optional_arrays = {
                name: np.asarray(values[name]) for name in _OPTIONAL_ARRAYS if name in values.files
            }
        row_count = len(arrays["time_seconds"])
        if any(len(array) != row_count for array in arrays.values()):
            raise ValueError("trajectory arrays have inconsistent time dimensions")
        node_ids = tuple(str(item) for item in metadata.get("node_ids", []))
        link_ids = tuple(str(item) for item in metadata.get("link_ids", []))
        source_ids = tuple(str(item) for item in metadata.get("source_ids", []))
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("trajectory source order must be non-empty and unique")
        for name in ("pressure", "demand", "chlorine", "arsenic", "event_strength_by_node"):
            if arrays[name].shape != (row_count, len(node_ids)):
                raise ValueError(f"trajectory {name} shape does not match node order")
        if arrays["flow"].shape != (row_count, len(link_ids)):
            raise ValueError("trajectory flow shape does not match link order")
        trajectory = Trajectory(
            network=str(metadata["network"]), scenario_id=str(metadata["scenario_id"]),
            calendar_start_day=calendar_start_day,
            chlorine_profile_hash=str(metadata["chlorine_profile_hash"]),
            arsenic_schedule_hash=str(metadata["arsenic_schedule_hash"]),
            arsenic_schedule_seed=int(metadata["arsenic_schedule_seed"]),
            node_ids=node_ids, link_ids=link_ids, source_ids=source_ids, **arrays, **optional_arrays,
            simulator_backend=str(metadata.get("simulator_backend", "unknown")),
            simulator_config_hash=str(metadata.get("simulator_config_hash", "")),
            flow_units=metadata.get("flow_units"),
            reproducibility=tuple(sorted(dict(metadata.get("reproducibility", {})).items())),
        )
        _validate(trajectory)
        expected_shapes = metadata.get("arrays")
        actual_shapes = {
            name: list(np.asarray(getattr(trajectory, name)).shape)
            for name in _ARRAYS + tuple(optional_arrays)
        }
        if expected_shapes != actual_shapes:
            raise ValueError("trajectory array shapes disagree with metadata")
        return trajectory


def _validate(trajectory: Trajectory) -> None:
    rows = len(np.asarray(trajectory.time_seconds))
    if rows == 0:
        raise ValueError("trajectory must contain samples")
    if len(set(trajectory.node_ids)) != len(trajectory.node_ids) or not trajectory.node_ids:
        raise ValueError("trajectory node order must be non-empty and unique")
    if len(set(trajectory.link_ids)) != len(trajectory.link_ids):
        raise ValueError("trajectory link order must be unique")
    if not trajectory.source_ids or len(set(trajectory.source_ids)) != len(trajectory.source_ids):
        raise ValueError("trajectory source order must be non-empty and unique")
    if np.asarray(trajectory.time_seconds).ndim != 1:
        raise ValueError("trajectory timestamps must be one-dimensional")
    if np.any(np.diff(np.asarray(trajectory.time_seconds, dtype=np.float64)) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    for name in ("pressure", "demand", "chlorine", "arsenic", "event_strength_by_node"):
        if np.asarray(getattr(trajectory, name)).shape != (rows, len(trajectory.node_ids)):
            raise ValueError(f"trajectory {name} shape does not match node order")
    if np.asarray(trajectory.flow).shape != (rows, len(trajectory.link_ids)):
        raise ValueError("trajectory flow shape does not match link order")
    if (trajectory.hydraulic_outflow is not None and
            np.asarray(trajectory.hydraulic_outflow).shape != (rows, len(trajectory.node_ids))):
        raise ValueError("trajectory hydraulic_outflow shape does not match node order")
    for name in ("injection", "actions", "event_active"):
        if np.asarray(getattr(trajectory, name)).shape != (rows,):
            raise ValueError(f"trajectory {name} must have one value per timestamp")
