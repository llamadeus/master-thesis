#!/usr/bin/env python3
"""Generate MPC teacher labels with CEM and a frozen forecast bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import random
import signal
import sys
import zlib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.constants import MAX_CHLORINE_INJECTION  # noqa: E402
from packages.trainer.control.rewards import RewardInput  # noqa: E402
from packages.trainer.data import resolve_dataset_split  # noqa: E402
from packages.trainer.forecasting.bundle import \
    load_bundle_manifest  # noqa: E402
from packages.trainer.forecasting.features import (  # noqa: E402
    RegisteredForecastFeatureBuilder, apply_transform, inverse_transform)
from packages.trainer.forecasting.materialization import (  # noqa: E402
    ForecastTrajectoryResolver, TrajectoryReference)
from packages.trainer.forecasting.model import EncoderDecoderLSTM  # noqa: E402
from packages.trainer.observations import \
    ControlObservationBuilder  # noqa: E402
from packages.trainer.serialization import canonical_data  # noqa: E402
from packages.trainer.serialization import config_hash
from packages.trainer.simulation import Trajectory
from packages.trainer.specs import ControlObservationSpec  # noqa: E402
from packages.trainer.specs import (DatasetSplitSpec,  # noqa: E402
                                    ForecastFeatureSpec, ForecastInputSpec,
                                    ForecastTrainingSpec)
from tools.utils.policy_observation_replay import (  # noqa: E402
    policy_observation_history_at, replay_policy_observations,
    simulation_sample_from_trajectory)
from training.datasets.hanoi import forecasting_full  # noqa: E402
from training.experiments.forecasting_encoder_decoder_lstm import \
    build_experiment  # noqa: E402
from training.rewards.v89 import \
    DEFAULT_PARAMETERS as V89_PARAMETERS  # noqa: E402
from training.rewards.v89 import V89Observation, V89Reward  # noqa: E402

HISTORY_STEPS = 48
HORIZON_STEPS = 16
ACTION_CONTROL_POINT_OFFSETS = (0, 1, 2, 3, 5, 8, 11, 15)
ACTION_SEQUENCE_LENGTH = len(ACTION_CONTROL_POINT_OFFSETS)
V89_OBSERVATION = ControlObservationSpec(V89Observation, V89_PARAMETERS)
# Default width for Hanoi; runtime validation derives it from the node order.
POLICY_OBSERVATION_SIZE = V89_OBSERVATION.factory(
    **V89_OBSERVATION.as_dict()
).observation_size(tuple(f"node-{index}" for index in range(31)))
HALF_HOUR_SECONDS = 1_800
HALF_HOUR_SLOTS_PER_DAY = 48
MAX_DELTA = 50_000.0
MIN_INJECTION = 0.0
MAX_INJECTION = MAX_CHLORINE_INJECTION
INITIAL_INJECTION = 0.0
DATASET_SCHEMA_VERSION = 2
INITIAL_CEM_STD = 1.0
MIN_CEM_STD = 1.0 / 60.0

DEFAULT_QUERIES_PER_TRAJECTORY = 24
DEFAULT_CANDIDATES = 64
DEFAULT_ITERATIONS = 4
DEFAULT_ELITE_FRACTION = 0.1
DEFAULT_SEED = 102
DEFAULT_DISCOUNT_FACTOR = 0.99

FORECASTING_JOB_HASH = config_hash(build_experiment().jobs[0])
DEFAULT_FORECASTING_RUN = PROJECT_ROOT / (
    "data/weights/forecasting/encoder-decoder-lstm/"
    "forecasting-encoder-decoder-lstm/encoder-decoder-lstm/"
    f"{FORECASTING_JOB_HASH}"
)
DEFAULT_TRAJECTORY_CACHE = PROJECT_ROOT / "data/runs"
DEFAULT_SOURCE_DEFINITION = "training.datasets.hanoi.forecasting_full.train"


def action_contract() -> dict[str, Any]:
    return {
        "raw_minimum": -1.0,
        "raw_maximum": 1.0,
        "max_delta": MAX_DELTA,
        "minimum_injection": MIN_INJECTION,
        "maximum_injection": MAX_INJECTION,
        "initial_injection": INITIAL_INJECTION,
        "cumulative": True,
    }


def reward_contract(discount_factor: float) -> dict[str, Any]:
    return {
        "id": V89Reward.reward_id,
        "parameters": dict(V89_PARAMETERS),
        "discount_factor": float(discount_factor),
        "query_state_reconstruction": (
            "replay-recorded-trajectory-rows-1-through-query"
        ),
        "stored_state_encoding": "base64(zlib(canonical-json-utf8))",
    }


def reward_from_manifest(manifest: dict[str, Any]) -> V89Reward:
    """Reconstruct the reward only from a complete artifact contract."""
    contract = manifest.get("reward", {})
    parameters = contract.get("parameters")
    if contract.get("id") != V89Reward.reward_id or not isinstance(parameters, dict):
        raise ValueError("artifact does not contain a complete V89 reward contract")
    if set(parameters) != set(V89_PARAMETERS):
        raise ValueError("artifact reward parameters are incomplete or unknown")
    return V89Reward({str(key): float(value) for key, value in parameters.items()})


def teacher_configuration(
    *,
    forecast_job_hash: str,
    state_count: int | None,
    queries_per_trajectory: int,
    seed: int,
    discount_factor: float,
    optimizer_kind: str,
    optimizer: dict[str, Any],
    source_split: DatasetSplitSpec = forecasting_full.train,
    source_network: str = forecasting_full.network,
    observation_size: int | None = None,
) -> dict[str, Any]:
    """Return every semantic input that identifies a teacher dataset."""
    observation_spec = V89_OBSERVATION
    if observation_size is None:
        observation_size = observation_spec.factory(
            **observation_spec.as_dict()
        ).observation_size(tuple(f"node-{index}" for index in range(31)))
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset": {
            "network": source_network,
            "split": canonical_data(source_split),
        },
        "forecast": {
            "job_hash": str(forecast_job_hash),
            "history_steps": HISTORY_STEPS,
            "horizon_steps": HORIZON_STEPS,
        },
        "selection": {
            "state_count": state_count,
            "queries_per_trajectory": int(queries_per_trajectory),
            "seed": int(seed),
        },
        "action_contract": action_contract(),
        "policy_observation_contract": {
            "spec": canonical_data(observation_spec),
            "observation_size": observation_size,
            "max_injection": MAX_INJECTION,
        },
        "reward": reward_contract(discount_factor),
        "optimizer": {
            "kind": optimizer_kind,
            "settings": canonical_data(optimizer),
        },
    }


def cem_optimizer_settings(
    candidates: int,
    iterations: int,
    elite_fraction: float,
) -> dict[str, Any]:
    return {
        "candidates": int(candidates),
        "iterations": int(iterations),
        "elite_fraction": float(elite_fraction),
        "initial_action_mean": 0.0,
        "initial_action_std": INITIAL_CEM_STD,
        "minimum_action_std": MIN_CEM_STD,
        "hold_candidate_every_iteration": True,
        "preserve_global_best": True,
        "action_parameterization": {
            "kind": "fixed-piecewise-linear",
            "parameter_space": "raw-action-delta",
            "control_point_offsets": list(ACTION_CONTROL_POINT_OFFSETS),
            "decision_steps": ACTION_SEQUENCE_LENGTH,
            "expanded_horizon_steps": HORIZON_STEPS,
            "interpolation": "linear",
            "stored_plan_representation": "expanded-per-step",
            "diagnostic_representation": "expanded-marginal-mean-and-std",
        },
    }


DEFAULT_TEACHER_CONFIGURATION = teacher_configuration(
    forecast_job_hash=FORECASTING_JOB_HASH,
    state_count=None,
    queries_per_trajectory=DEFAULT_QUERIES_PER_TRAJECTORY,
    seed=DEFAULT_SEED,
    discount_factor=DEFAULT_DISCOUNT_FACTOR,
    optimizer_kind="cem",
    optimizer=cem_optimizer_settings(
        DEFAULT_CANDIDATES,
        DEFAULT_ITERATIONS,
        DEFAULT_ELITE_FRACTION,
    ),
)
DEFAULT_TEACHER_CONFIGURATION_HASH = config_hash(DEFAULT_TEACHER_CONFIGURATION)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    f"data/datasets/mpc-teacher-cem/{DEFAULT_TEACHER_CONFIGURATION_HASH}"
)

REQUIRED_DATASET_ARRAYS = (
    "scenario_id",
    "trajectory_identity",
    "chlorine_profile_hash",
    "arsenic_schedule_hash",
    "calendar_start_day",
    "query_index",
    "time_seconds",
    "raw_causal_history",
    "policy_observation_history",
    "current_injection",
    "best_raw_actions",
    "best_injections",
    "first_action",
    "predicted_chlorine",
    "predicted_return",
    "reward_state_zlib_base64",
    "cem_seed",
    "cem_iteration_population_max",
    "cem_iteration_population_mean",
    "cem_iteration_elite_mean",
    "cem_iteration_best_so_far",
    "cem_iteration_action_mean",
    "cem_iteration_action_std",
)


@dataclass(frozen=True)
class PreparedForecastQuery:
    raw_history: np.ndarray
    normalized_history: np.ndarray
    base_future_known: np.ndarray
    future_injection_index: int


@dataclass(frozen=True)
class PreparedForecastTrajectory:
    raw_history: np.ndarray
    transformed_history: np.ndarray
    transformed_future_known: np.ndarray
    future_injection_index: int


@dataclass(frozen=True)
class CEMBatchEvaluation:
    returns: np.ndarray
    predicted_chlorine: np.ndarray


@dataclass(frozen=True)
class CEMDiagnostics:
    population_max: np.ndarray
    population_mean: np.ndarray
    elite_mean: np.ndarray
    best_so_far: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


@dataclass(frozen=True)
class CEMResult:
    best_actions: np.ndarray
    best_return: float
    best_predicted_chlorine: np.ndarray
    diagnostics: CEMDiagnostics


@dataclass(frozen=True)
class QuerySelection:
    reference: TrajectoryReference
    query_index: int


@dataclass(frozen=True)
class TrajectoryQueryGroup:
    reference: TrajectoryReference
    query_indexes: tuple[int, ...]


@dataclass(frozen=True)
class GenerationConfig:
    candidates: int
    iterations: int
    elite_fraction: float
    discount_factor: float
    seed: int


@dataclass(frozen=True)
class TeacherRecord:
    scenario_id: str
    trajectory_identity: str
    chlorine_profile_hash: str
    arsenic_schedule_hash: str
    calendar_start_day: int
    query_index: int
    time_seconds: int
    raw_causal_history: np.ndarray
    policy_observation_history: np.ndarray
    current_injection: float
    best_raw_actions: np.ndarray
    best_injections: np.ndarray
    first_action: float
    predicted_chlorine: np.ndarray
    predicted_return: float
    reward_state_zlib_base64: bytes
    cem_seed: int
    diagnostics: CEMDiagnostics


class ForecastBundleRuntime:
    """Bundle loader with a candidate-aware batched decoder interface."""

    def __init__(self, bundle: Path, device: torch.device):
        self.bundle = require_completed_best_bundle(bundle)
        self.manifest = load_bundle_manifest(self.bundle)
        self.device = device
        self._validate_manifest()

        self.inputs = _input_spec(self.manifest["job"]["observation"])
        self.raw_inputs = ForecastInputSpec(
            history=tuple(
                replace(feature, transform="identity", scale=None)
                for feature in self.inputs.history
            ),
            future_known=self.inputs.future_known,
        )
        self.builder = RegisteredForecastFeatureBuilder(self.inputs)
        self.raw_builder = RegisteredForecastFeatureBuilder(self.raw_inputs)

        model_kwargs = dict(self.manifest["model_kwargs"])
        model_kwargs["head_sizes"] = tuple(
            tuple(item) for item in model_kwargs["head_sizes"]
        )
        self.model = EncoderDecoderLSTM(**model_kwargs).to(device)
        self.model.load_state_dict(
            torch.load(
                self.bundle / "model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        self.model.eval()
        self.model.requires_grad_(False)

        raw_normalization = json.loads(
            (self.bundle / "normalization.json").read_text(encoding="utf-8")
        )
        self.history_mean = np.asarray(
            raw_normalization["history_mean"], dtype=np.float32
        )
        self.history_std = np.asarray(
            raw_normalization["history_std"], dtype=np.float32
        )
        self.future_mean = np.asarray(
            raw_normalization["future_mean"], dtype=np.float32
        )
        self.future_std = np.asarray(
            raw_normalization["future_std"], dtype=np.float32
        )
        target = raw_normalization["targets"]["chlorine"]
        self.target_mean = np.asarray(target["mean"], dtype=np.float32)
        self.target_std = np.asarray(target["std"], dtype=np.float32)
        self._validate_normalization()

        self.node_ids = tuple(str(item) for item in self.manifest["node_ids"])
        self.source_ids = tuple(str(item) for item in self.manifest["source_ids"])
        self.timestep_seconds = int(self.manifest["hydraulic_timestep_seconds"])
        self.head = next(
            item for item in self.manifest["heads"] if item["name"] == "chlorine"
        )
        self.raw_history_names = tuple(
            self.manifest["history_feature_contract"]["names"]
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("kind") != "encoder-decoder-lstm":
            raise ValueError("MPC teacher generation requires an encoder-decoder-lstm bundle")
        node_ids = tuple(manifest.get("node_ids", ()))
        source_ids = tuple(manifest.get("source_ids", ()))
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("forecast bundle node order must be non-empty and unique")
        model = manifest.get("job", {}).get("model", {})
        if (
            int(model.get("history_steps", -1)) != HISTORY_STEPS
            or int(model.get("horizon_steps", -1)) != HORIZON_STEPS
            or int(manifest["model_kwargs"].get("horizon_steps", -1))
            != HORIZON_STEPS
        ):
            raise ValueError("forecast bundle must use 48 history steps and a 16-step horizon")
        if len(source_ids) != 1:
            raise ValueError("MPC teacher generation requires exactly one chlorine source")
        heads = [item for item in manifest.get("heads", ()) if item.get("name") == "chlorine"]
        if len(heads) != 1:
            raise ValueError("forecast bundle must contain exactly one chlorine head")
        observation = manifest.get("job", {}).get("observation", {})
        features = (*observation.get("history", ()), *observation.get("future_known", ()))
        allowed = {"chlorine", "chlorine-injection", "cyclic-time"}
        if any(item.get("name") not in allowed for item in features):
            raise ValueError("forecast bundle contains unsupported non-causal features")
        if not any(
            item.get("name") == "chlorine-injection"
            for item in observation.get("future_known", ())
        ):
            raise ValueError("forecast bundle future-known contract lacks chlorine injection")
        supported_transforms = {"identity", "log1p", "divide-by"}
        if any(item.get("transform", "identity") not in supported_transforms for item in features):
            raise ValueError("forecast bundle contains an unsupported feature transform")
        if int(manifest.get("hydraulic_timestep_seconds", -1)) != 1_800:
            raise ValueError("forecast bundle must use the Hanoi 1,800-second timestep")
        history_width = len(manifest["history_feature_contract"]["names"])
        future_width = len(manifest["future_known_feature_contract"]["names"])
        model_kwargs = manifest["model_kwargs"]
        if (
            int(model_kwargs.get("input_size", -1)) != history_width
            or int(model_kwargs.get("decoder_input_size", -1)) != future_width
        ):
            raise ValueError("forecast model dimensions disagree with feature contracts")
        head_sizes = dict(model_kwargs.get("head_sizes", ()))
        if head_sizes.get("chlorine") != len(node_ids):
            raise ValueError("forecast chlorine head width disagrees with node order")

    def _validate_normalization(self) -> None:
        history_width = len(self.manifest["history_feature_contract"]["names"])
        future_width = len(self.manifest["future_known_feature_contract"]["names"])
        node_count = len(self.manifest["node_ids"])
        values = (
            (self.history_mean, (history_width,), "history mean"),
            (self.history_std, (history_width,), "history std"),
            (self.future_mean, (future_width,), "future mean"),
            (self.future_std, (future_width,), "future std"),
            (self.target_mean, (node_count,), "chlorine target mean"),
            (self.target_std, (node_count,), "chlorine target std"),
        )
        for value, shape, label in values:
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"forecast bundle has incompatible {label} normalization")
        if np.any(self.history_std <= 0) or np.any(self.future_std <= 0) or np.any(
            self.target_std <= 0
        ):
            raise ValueError("forecast bundle normalization standard deviations must be positive")

    def prepare_query(
        self, trajectory: Trajectory, query_index: int
    ) -> PreparedForecastQuery:
        return self.query_from_prepared(
            self.prepare_trajectory(trajectory), query_index
        )

    def prepare_trajectory(
        self, trajectory: Trajectory
    ) -> PreparedForecastTrajectory:
        if tuple(trajectory.node_ids) != self.node_ids:
            raise ValueError("trajectory node order does not match the forecast bundle")
        if tuple(trajectory.source_ids) != self.source_ids:
            raise ValueError("trajectory source order does not match the forecast bundle")

        transformed = self.builder.build(trajectory)
        raw = self.raw_builder.build(trajectory)
        if transformed.history.names != raw.history.names:
            raise ValueError("physical and transformed history contracts disagree")
        if tuple(transformed.history.names) != tuple(
            self.manifest["history_feature_contract"]["names"]
        ):
            raise ValueError("trajectory history feature contract does not match bundle")
        if tuple(transformed.future_known.names) != tuple(
            self.manifest["future_known_feature_contract"]["names"]
        ):
            raise ValueError("trajectory future feature contract does not match bundle")

        future_injection_indexes = [
            index
            for index, name in enumerate(transformed.future_known.names)
            if name == "chlorine-injection" or name.startswith("chlorine-injection:")
        ]
        if len(future_injection_indexes) != 1:
            raise ValueError("future-known contract must expand to one injection feature")
        injection_index = future_injection_indexes[0]
        transform = transformed.future_known.transforms[injection_index]
        if transform["feature"] != transformed.future_known.names[injection_index]:
            raise ValueError("future injection transform contract is inconsistent")

        return PreparedForecastTrajectory(
            raw_history=np.asarray(raw.history.values, dtype=np.float32),
            transformed_history=np.asarray(
                transformed.history.values, dtype=np.float32
            ),
            transformed_future_known=np.asarray(
                transformed.future_known.values, dtype=np.float32
            ),
            future_injection_index=injection_index,
        )

    def query_from_prepared(
        self,
        prepared: PreparedForecastTrajectory,
        query_index: int,
    ) -> PreparedForecastQuery:
        if query_index < HISTORY_STEPS - 1:
            raise ValueError("query does not have a complete 48-step history")
        if query_index + HORIZON_STEPS >= len(prepared.raw_history):
            raise ValueError("query does not have a complete in-scenario planning horizon")

        start = query_index - HISTORY_STEPS + 1
        end = query_index + 1
        normalized_history = (
            prepared.transformed_history[start:end] - self.history_mean
        ) / self.history_std
        future = prepared.transformed_future_known[
            query_index + 1 : query_index + HORIZON_STEPS + 1
        ]
        return PreparedForecastQuery(
            raw_history=np.asarray(prepared.raw_history[start:end], dtype=np.float32),
            normalized_history=np.asarray(normalized_history, dtype=np.float32),
            base_future_known=np.asarray(future, dtype=np.float32),
            future_injection_index=prepared.future_injection_index,
        )

    def predict_candidates(
        self,
        prepared: PreparedForecastQuery,
        candidate_injections: np.ndarray,
        current_chlorine: np.ndarray,
    ) -> np.ndarray:
        injections = np.asarray(candidate_injections, dtype=np.float32)
        if injections.ndim != 2 or injections.shape[1] != HORIZON_STEPS:
            raise ValueError("candidate injections must have shape (candidate, 16)")
        candidate_count = injections.shape[0]
        history = np.broadcast_to(
            prepared.normalized_history,
            (candidate_count, *prepared.normalized_history.shape),
        ).copy()
        future = np.broadcast_to(
            prepared.base_future_known,
            (candidate_count, *prepared.base_future_known.shape),
        ).copy()
        index = prepared.future_injection_index
        transform = self.manifest["future_known_feature_contract"]["transforms"][index]
        future[..., index] = apply_transform(
            injections,
            str(transform["transform"]),
            transform.get("scale"),
        )
        future = (future - self.future_mean) / self.future_std

        with torch.inference_mode():
            outputs = self.model(
                torch.from_numpy(history.astype(np.float32)).to(self.device),
                torch.from_numpy(future.astype(np.float32)).to(self.device),
            )["chlorine"]
        normalized = outputs.detach().cpu().numpy()
        represented = normalized * self.target_std + self.target_mean
        physical = inverse_transform(
            represented,
            str(self.head["transform"]),
            self.head.get("scale"),
        )
        if self.head["prediction"] == "delta":
            physical = np.asarray(current_chlorine, dtype=np.float32)[None, None, :] + np.cumsum(
                physical, axis=1
            )
        elif self.head["prediction"] != "level":
            raise ValueError("forecast chlorine head has an unsupported prediction mode")
        expected = (candidate_count, HORIZON_STEPS, len(self.node_ids))
        if physical.shape != expected:
            raise ValueError(
                f"forecast output shape {physical.shape} does not match expected {expected}"
            )
        if not np.all(np.isfinite(physical)):
            raise ValueError("forecast model produced non-finite chlorine predictions")
        return np.maximum(physical, 0.0).astype(np.float32)


@dataclass(frozen=True)
class EnsembleForecastQuery:
    members: tuple[PreparedForecastQuery, ...]

    @property
    def raw_history(self) -> np.ndarray:
        return self.members[0].raw_history


class EnsembleForecastRuntime:
    """Keep each fold's preprocessing and prediction separate throughout scoring."""

    def __init__(self, bundles: tuple[Path, ...], device: torch.device):
        if len(bundles) != 5:
            raise ValueError("validation teacher requires exactly five fold forecasters")
        self.bundles = bundles
        self.members = tuple(ForecastBundleRuntime(path, device) for path in bundles)
        first = self.members[0]
        for name in ("bundle", "manifest", "node_ids", "source_ids",
                     "timestep_seconds", "raw_history_names"):
            setattr(self, name, getattr(first, name))
        for member in self.members[1:]:
            for name in ("node_ids", "source_ids", "timestep_seconds", "raw_history_names"):
                if getattr(member, name) != getattr(first, name):
                    raise ValueError(f"ensemble forecast contract differs: {name}")

    def prepare_trajectory(self, trajectory: Trajectory) -> tuple[PreparedForecastTrajectory, ...]:
        return tuple(member.prepare_trajectory(trajectory) for member in self.members)

    def query_from_prepared(
        self, prepared: tuple[PreparedForecastTrajectory, ...], query_index: int,
    ) -> EnsembleForecastQuery:
        return EnsembleForecastQuery(tuple(
            member.query_from_prepared(item, query_index)
            for member, item in zip(self.members, prepared, strict=True)
        ))

    def predict_candidates(
        self, prepared: EnsembleForecastQuery, injections: np.ndarray,
        current_chlorine: np.ndarray,
    ) -> np.ndarray:
        # Shape: candidate, member, horizon, node. No trajectory averaging.
        return np.stack([
            member.predict_candidates(query, injections, current_chlorine)
            for member, query in zip(self.members, prepared.members, strict=True)
        ], axis=1)


def score_forecast_predictions(
    predicted: np.ndarray, raw_actions: np.ndarray, injections: np.ndarray, **context: Any,
) -> np.ndarray:
    if predicted.ndim == 3:
        return score_predicted_candidates(predicted, raw_actions, injections, **context)
    if predicted.ndim != 4 or predicted.shape[1] != 5:
        raise ValueError("ensemble predictions require five separate member trajectories")
    returns = [
        score_predicted_candidates(predicted[:, index], raw_actions, injections, **context)
        for index in range(predicted.shape[1])
    ]
    return np.mean(np.stack(returns), axis=0, dtype=np.float64)


def require_completed_best_bundle(path: Path) -> Path:
    """Require bundle artifacts and deliberately ignore checkpoint fallbacks."""
    bundle = Path(path)
    required = ("manifest.json", "normalization.json", "model.pt")
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "completed forecasting best/ bundle is required because manifest and "
            "normalization artifacts cannot be reconstructed from a checkpoint; "
            f"missing {missing} in {bundle}"
        )
    return bundle


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_generation_device(value: str, workers: int) -> torch.device:
    """Use accelerators only for the single-process execution path."""
    if workers == 1:
        return resolve_device(value)
    if value == "auto":
        return torch.device("cpu")
    requested = torch.device(value)
    if requested.type != "cpu":
        raise ValueError(
            "--workers greater than one requires --device cpu; independent "
            "worker processes must not contend for one accelerator"
        )
    return requested


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_action_interpolation_weights() -> np.ndarray:
    offsets = np.asarray(ACTION_CONTROL_POINT_OFFSETS, dtype=np.int64)
    if (
        offsets.shape != (ACTION_SEQUENCE_LENGTH,)
        or offsets[0] != 0
        or offsets[-1] != HORIZON_STEPS - 1
        or np.any(np.diff(offsets) <= 0)
    ):
        raise RuntimeError("action control points must strictly span the horizon")
    weights = np.zeros(
        (HORIZON_STEPS, ACTION_SEQUENCE_LENGTH), dtype=np.float64
    )
    for step in range(HORIZON_STEPS):
        exact = np.flatnonzero(offsets == step)
        if exact.size:
            weights[step, int(exact[0])] = 1.0
            continue
        right = int(np.searchsorted(offsets, step, side="right"))
        left = right - 1
        fraction = (step - offsets[left]) / (offsets[right] - offsets[left])
        weights[step, left] = 1.0 - fraction
        weights[step, right] = fraction
    return weights


ACTION_INTERPOLATION_WEIGHTS = _build_action_interpolation_weights()


def expand_action_control_points(control_points: np.ndarray) -> np.ndarray:
    """Linearly expand eight raw-action control points to the 16-step horizon."""
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim not in (1, 2) or points.shape[-1] != ACTION_SEQUENCE_LENGTH:
        raise ValueError(
            "action control points must have shape (8,) or (batch, 8)"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("action control points must be finite")
    bounded = np.clip(points, -1.0, 1.0)
    expanded = bounded @ ACTION_INTERPOLATION_WEIGHTS.T
    return expanded.astype(np.float32)


def expand_action_control_point_std(control_point_std: np.ndarray) -> np.ndarray:
    """Project independent control-point standard deviations onto all 16 steps."""
    standard_deviations = np.asarray(control_point_std, dtype=np.float64)
    if (
        standard_deviations.ndim not in (1, 2)
        or standard_deviations.shape[-1] != ACTION_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "action control-point standard deviations must have shape "
            "(8,) or (batch, 8)"
        )
    if not np.all(np.isfinite(standard_deviations)) or np.any(
        standard_deviations < 0.0
    ):
        raise ValueError(
            "action control-point standard deviations must be finite and non-negative"
        )
    variances = np.square(standard_deviations) @ np.square(
        ACTION_INTERPOLATION_WEIGHTS
    ).T
    return np.sqrt(np.maximum(variances, 0.0)).astype(np.float32)


def actions_to_injections(
    raw_actions: np.ndarray,
    current_injection: float,
    *,
    max_delta: float = MAX_DELTA,
    minimum: float = MIN_INJECTION,
    maximum: float = MAX_INJECTION,
) -> np.ndarray:
    """Convert raw controls to bounded cumulative absolute injections."""
    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.ndim not in (1, 2):
        raise ValueError("raw actions must have shape (horizon,) or (batch, horizon)")
    if not np.all(np.isfinite(actions)) or not np.isfinite(current_injection):
        raise ValueError("actions and current injection must be finite")
    if max_delta <= 0 or maximum <= minimum:
        raise ValueError("injection bounds and maximum delta are invalid")
    batched = actions[None, :] if actions.ndim == 1 else actions
    result = np.empty_like(batched, dtype=np.float64)
    previous = np.full(batched.shape[0], np.clip(current_injection, minimum, maximum))
    for step in range(batched.shape[1]):
        previous = np.clip(
            previous + np.clip(batched[:, step], -1.0, 1.0) * max_delta,
            minimum,
            maximum,
        )
        result[:, step] = previous
    result = result.astype(np.float32)
    return result[0] if actions.ndim == 1 else result


def optimize_cem(
    evaluator: Callable[[np.ndarray], CEMBatchEvaluation],
    *,
    horizon: int,
    candidates: int,
    iterations: int,
    elite_fraction: float,
    seed: int,
    initial_std: float = INITIAL_CEM_STD,
    min_std: float = MIN_CEM_STD,
) -> CEMResult:
    """Optimize a raw action sequence while retaining the global best sample."""
    if horizon <= 0 or candidates < 2 or iterations <= 0:
        raise ValueError("CEM requires a positive horizon/iterations and at least two candidates")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite fraction must lie in (0, 1]")
    if initial_std <= 0 or min_std <= 0:
        raise ValueError("CEM standard deviations must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("CEM seed must be a non-negative integer")

    rng = np.random.default_rng(seed)
    mean = np.zeros(horizon, dtype=np.float64)
    std = np.full(horizon, initial_std, dtype=np.float64)
    elite_count = min(candidates, max(2, int(np.ceil(candidates * elite_fraction))))
    best_return = -np.inf
    best_actions: np.ndarray | None = None
    best_prediction: np.ndarray | None = None
    population_max: list[float] = []
    population_mean: list[float] = []
    elite_mean: list[float] = []
    best_so_far: list[float] = []
    action_mean: list[np.ndarray] = []
    action_std: list[np.ndarray] = []

    for _iteration in range(iterations):
        sampled = rng.normal(mean, std, size=(candidates - 1, horizon))
        population = np.concatenate(
            (np.zeros((1, horizon), dtype=np.float64), np.clip(sampled, -1.0, 1.0)),
            axis=0,
        ).astype(np.float32)
        evaluation = evaluator(population)
        returns = np.asarray(evaluation.returns, dtype=np.float64)
        predictions = np.asarray(evaluation.predicted_chlorine)
        if returns.shape != (candidates,) or not np.all(np.isfinite(returns)):
            raise ValueError("CEM evaluator must return one finite return per candidate")
        if predictions.shape[0] != candidates:
            raise ValueError("CEM evaluator prediction batch does not match candidates")

        order = np.argsort(-returns, kind="stable")
        iteration_best = int(order[0])
        if returns[iteration_best] > best_return:
            best_return = float(returns[iteration_best])
            best_actions = population[iteration_best].copy()
            best_prediction = predictions[iteration_best].copy()

        elite = population[order[:elite_count]].astype(np.float64)
        mean = np.mean(elite, axis=0)
        std = np.maximum(np.std(elite, axis=0), min_std)
        population_max.append(float(returns[iteration_best]))
        population_mean.append(float(np.mean(returns)))
        elite_mean.append(float(np.mean(returns[order[:elite_count]])))
        best_so_far.append(best_return)
        action_mean.append(mean.astype(np.float32))
        action_std.append(std.astype(np.float32))

    if best_actions is None or best_prediction is None:
        raise RuntimeError("CEM produced no candidate result")
    diagnostics = CEMDiagnostics(
        population_max=np.asarray(population_max, dtype=np.float32),
        population_mean=np.asarray(population_mean, dtype=np.float32),
        elite_mean=np.asarray(elite_mean, dtype=np.float32),
        best_so_far=np.asarray(best_so_far, dtype=np.float32),
        action_mean=np.stack(action_mean).astype(np.float32),
        action_std=np.stack(action_std).astype(np.float32),
    )
    return CEMResult(
        best_actions=best_actions.astype(np.float32),
        best_return=best_return,
        best_predicted_chlorine=np.asarray(best_prediction, dtype=np.float32),
        diagnostics=diagnostics,
    )


def optimize_cem_action_sequence(
    evaluator: Callable[[np.ndarray], CEMBatchEvaluation],
    *,
    candidates: int,
    iterations: int,
    elite_fraction: float,
    seed: int,
    initial_std: float = INITIAL_CEM_STD,
    min_std: float = MIN_CEM_STD,
) -> CEMResult:
    """Optimize eight control points while evaluating expanded 16-step plans."""

    def evaluate_control_points(control_points: np.ndarray) -> CEMBatchEvaluation:
        return evaluator(expand_action_control_points(control_points))

    result = optimize_cem(
        evaluate_control_points,
        horizon=ACTION_SEQUENCE_LENGTH,
        candidates=candidates,
        iterations=iterations,
        elite_fraction=elite_fraction,
        seed=seed,
        initial_std=initial_std,
        min_std=min_std,
    )
    diagnostics = CEMDiagnostics(
        population_max=result.diagnostics.population_max,
        population_mean=result.diagnostics.population_mean,
        elite_mean=result.diagnostics.elite_mean,
        best_so_far=result.diagnostics.best_so_far,
        action_mean=expand_action_control_points(result.diagnostics.action_mean),
        action_std=expand_action_control_point_std(result.diagnostics.action_std),
    )
    return CEMResult(
        best_actions=expand_action_control_points(result.best_actions),
        best_return=result.best_return,
        best_predicted_chlorine=result.best_predicted_chlorine,
        diagnostics=diagnostics,
    )


def reward_input_from_trajectory(trajectory: Trajectory, index: int) -> RewardInput:
    event_strength = np.asarray(trajectory.event_strength_by_node[index])
    return RewardInput(
        nodes_quality=np.asarray(trajectory.chlorine[index]),
        contaminant_values=np.asarray(trajectory.arsenic[index]),
        pressure=np.asarray(trajectory.pressure[index]),
        demand=np.asarray(trajectory.demand[index]),
        flow=np.asarray(trajectory.flow[index]),
        current_injection=float(trajectory.injection[index]),
        raw_action=float(trajectory.actions[index]),
        step=index,
        current_time=int(trajectory.time_seconds[index]),
        node_ids=trajectory.node_ids,
        link_ids=trajectory.link_ids,
        event_active=bool(trajectory.event_active[index]),
        event_strength_by_node=event_strength,
    )


def reconstruct_reward_state(
    trajectory: Trajectory,
    query_index: int,
    reward_parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay scored rows; row zero is the environment reset observation."""
    if query_index < 0 or query_index >= len(trajectory.time_seconds):
        raise ValueError("query index is outside the trajectory")
    reward = V89Reward(V89_PARAMETERS if reward_parameters is None else reward_parameters)
    for index in range(1, query_index + 1):
        reward.compute(reward_input_from_trajectory(trajectory, index))
    return reward.state_dict()


def incremental_reward_states(
    trajectory: Trajectory,
    query_indexes: tuple[int, ...],
):
    """Yield exact query reward states while replaying each trajectory row once."""
    reward = V89Reward(V89_PARAMETERS)
    replayed_through = 0
    for query_index in query_indexes:
        if query_index < replayed_through:
            raise ValueError("incremental reward queries must be sorted")
        for index in range(replayed_through + 1, query_index + 1):
            reward.compute(reward_input_from_trajectory(trajectory, index))
        replayed_through = query_index
        yield query_index, reward.state_dict()


def score_predicted_candidates(
    predicted_chlorine: np.ndarray,
    raw_actions: np.ndarray,
    injections: np.ndarray,
    *,
    reward_state: dict[str, Any],
    trajectory: Trajectory,
    query_index: int,
    timestep_seconds: int,
    discount_factor: float,
    reward_parameters: dict[str, float] | None = None,
) -> np.ndarray:
    predictions = np.asarray(predicted_chlorine, dtype=np.float32)
    actions = np.asarray(raw_actions, dtype=np.float32)
    absolute = np.asarray(injections, dtype=np.float32)
    expected = (actions.shape[0], HORIZON_STEPS, len(trajectory.node_ids))
    if predictions.shape != expected or absolute.shape != actions.shape:
        raise ValueError("candidate reward inputs have incompatible shapes")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount factor must lie in [0, 1]")

    query = reward_input_from_trajectory(trajectory, query_index)
    discounts = np.power(discount_factor, np.arange(HORIZON_STEPS, dtype=np.float64))
    returns = np.empty(actions.shape[0], dtype=np.float64)
    for candidate in range(actions.shape[0]):
        reward = V89Reward(V89_PARAMETERS if reward_parameters is None else reward_parameters)
        reward.load_state_dict(reward_state)
        total = 0.0
        for step in range(HORIZON_STEPS):
            value, _info = reward.compute(
                RewardInput(
                    nodes_quality=predictions[candidate, step],
                    contaminant_values=query.contaminant_values,
                    pressure=query.pressure,
                    demand=query.demand,
                    flow=query.flow,
                    current_injection=float(absolute[candidate, step]),
                    raw_action=float(actions[candidate, step]),
                    step=query_index + step + 1,
                    current_time=int(
                        trajectory.time_seconds[query_index]
                        + (step + 1) * timestep_seconds
                    ),
                    node_ids=query.node_ids,
                    link_ids=query.link_ids,
                    event_active=query.event_active,
                    event_strength_by_node=query.event_strength_by_node,
                )
            )
            total += discounts[step] * value
        returns[candidate] = total
    return returns.astype(np.float32)


def stable_query_seed(seed: int, selection: QuerySelection) -> int:
    payload = f"{seed}:{selection.reference.identity}:{selection.query_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def half_hour_slot_for_query_index(query_index: int) -> int:
    """Return the episode-clock half-hour slot for zero-based cached rows."""
    if query_index < 0:
        raise ValueError("query index must be non-negative")
    return int(query_index % HALF_HOUR_SLOTS_PER_DAY)


def temporally_stratified_query_pool(
    available: list[tuple[TrajectoryReference, int]],
    queries_per_trajectory: int,
    seed: int,
) -> list[QuerySelection]:
    """Choose random queries across each episode while balancing clock slots."""
    if queries_per_trajectory <= 0 or seed < 0:
        raise ValueError("temporal query sampling configuration is invalid")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54535452]))
    slot_loads = np.zeros(HALF_HOUR_SLOTS_PER_DAY, dtype=np.int64)
    selected: list[QuerySelection] = []

    # Randomize trajectory and stratum processing so balance tie-breaking does
    # not correlate with scenario, contaminant profile, or episode position.
    for trajectory_value in rng.permutation(len(available)):
        trajectory_index = int(trajectory_value)
        reference, row_count = available[trajectory_index]
        valid_indexes = np.arange(
            HISTORY_STEPS - 1,
            row_count - HORIZON_STEPS,
            dtype=np.int64,
        )
        if valid_indexes.size < queries_per_trajectory:
            raise ValueError(
                f"training trajectory {reference.identity} has only "
                f"{valid_indexes.size} valid queries; "
                f"{queries_per_trajectory} are required"
            )
        strata = np.array_split(valid_indexes, queries_per_trajectory)
        trajectory_queries = np.empty(queries_per_trajectory, dtype=np.int64)
        for stratum_value in rng.permutation(queries_per_trajectory):
            stratum_index = int(stratum_value)
            candidates = strata[stratum_index]
            candidate_slots = candidates % HALF_HOUR_SLOTS_PER_DAY
            minimum_load = int(slot_loads[candidate_slots].min())
            least_loaded = candidates[
                slot_loads[candidate_slots] == minimum_load
            ]
            query_index = int(least_loaded[int(rng.integers(len(least_loaded)))])
            trajectory_queries[stratum_index] = query_index
            slot_loads[half_hour_slot_for_query_index(query_index)] += 1
        selected.extend(
            QuerySelection(reference, int(query_index))
            for query_index in trajectory_queries
        )
    return selected


def stratified_query_sample(
    eligible: list[QuerySelection], state_count: int, seed: int
) -> list[QuerySelection]:
    """Sample without replacement while balancing episode-clock slots."""
    if state_count <= 0 or state_count > len(eligible):
        raise ValueError("state sample count is outside the eligible query pool")
    buckets: list[list[QuerySelection]] = [
        [] for _ in range(HALF_HOUR_SLOTS_PER_DAY)
    ]
    for selection in eligible:
        buckets[half_hour_slot_for_query_index(selection.query_index)].append(
            selection
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D50]))
    for bucket in buckets:
        rng.shuffle(bucket)
    slot_order = rng.permutation(HALF_HOUR_SLOTS_PER_DAY)
    selected: list[QuerySelection] = []
    next_index = np.zeros(HALF_HOUR_SLOTS_PER_DAY, dtype=np.int64)
    while len(selected) < state_count:
        added = False
        for slot_value in slot_order:
            slot = int(slot_value)
            index = int(next_index[slot])
            if index >= len(buckets[slot]):
                continue
            selected.append(buckets[slot][index])
            next_index[slot] += 1
            added = True
            if len(selected) == state_count:
                break
        if not added:
            raise RuntimeError("stratified query sampling exhausted its eligible pool")
    return selected


def select_training_queries(
    *,
    cache_root: Path,
    state_count: int | None,
    queries_per_trajectory: int,
    seed: int,
    bundle_manifest: dict[str, Any],
    forecast_job: ForecastTrainingSpec | None = None,
    source_split: DatasetSplitSpec = forecasting_full.train,
    require_out_of_fold: bool = False,
    source_phase: str = "train",
) -> tuple[QuerySelection, ...]:
    """Select temporally stratified states from exact training cache identities."""
    job = build_experiment().jobs[0] if forecast_job is None else forecast_job
    if canonical_data(job.dataset) != bundle_manifest.get("job", {}).get("dataset"):
        raise ValueError(
            "forecast bundle dataset does not match the supplied forecast job"
        )
    if job.dataset.network != forecasting_full.network:
        raise ValueError("forecast job network does not match forecasting_full")
    if require_out_of_fold:
        source_scenarios = set(source_split.scenarios)
        forecast_training = set(job.dataset.train.scenarios)
        reference_training = set(forecasting_full.train.scenarios)
        if not source_scenarios <= reference_training:
            raise ValueError(
                "out-of-fold source scenarios are not a subset of forecasting_full.train"
            )
        overlap = source_scenarios & forecast_training
        if overlap:
            raise ValueError(
                "out-of-fold source scenarios occur in forecast training: "
                f"{sorted(overlap)}"
            )
        if (
            canonical_data(source_split.arsenic)
            != canonical_data(forecasting_full.train.arsenic)
            or canonical_data(source_split.chlorine)
            != canonical_data(forecasting_full.train.chlorine)
        ):
            raise ValueError(
                "out-of-fold source schedules must match forecasting_full.train"
            )
    dataset = resolve_dataset_split(job.dataset.network, source_split, scenarios_dir=job.dataset.scenarios_dir,
                                    uncertainty_seed=job.dataset.uncertainty_seed, split=source_phase)
    trajectory_spec = replace(
        job.trajectory_data,
        cache_root=Path(cache_root),
        materialize_missing=not require_out_of_fold and source_phase == "train",
    )
    resolver = ForecastTrajectoryResolver(dataset, trajectory_spec, backend=None)
    available: list[tuple[TrajectoryReference, int]] = []

    # _prepared is deliberately used instead of references(): the latter eagerly
    # decompresses every training trajectory. Here metadata indexes the pool and
    # only selected trajectories are decoded and validated later.
    for reference, _scenario, _arsenic, _chlorine in resolver._prepared(source_phase):
        metadata_path = reference.cache_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"required read-only training trajectory cache is missing or invalid: {metadata_path}"
            ) from exc
        if metadata.get("scenario_id") != reference.scenario_id:
            raise ValueError(f"trajectory cache scenario identity mismatch: {metadata_path}")
        time_shape = metadata.get("arrays", {}).get("time_seconds")
        if not isinstance(time_shape, list) or len(time_shape) != 1:
            raise ValueError(f"trajectory cache has invalid time metadata: {metadata_path}")
        available.append((reference, int(time_shape[0])))

    available.sort(key=lambda item: (item[0].scenario_id, item[0].identity))
    eligible = temporally_stratified_query_pool(
        available,
        queries_per_trajectory,
        seed,
    )

    if state_count is not None and state_count > len(eligible):
        raise ValueError(
            f"requested {state_count} states but the stratified training pool "
            f"contains only {len(eligible)}"
        )
    if state_count is None:
        selected = eligible
    else:
        selected = stratified_query_sample(eligible, state_count, seed)
    selected.sort(
        key=lambda item: (
            item.reference.scenario_id,
            item.reference.identity,
            item.query_index,
        )
    )
    return tuple(selected)


def group_query_selections(
    selections: tuple[QuerySelection, ...],
) -> tuple[TrajectoryQueryGroup, ...]:
    grouped: dict[str, tuple[TrajectoryReference, list[int]]] = {}
    for selection in selections:
        identity = selection.reference.identity
        if identity not in grouped:
            grouped[identity] = (selection.reference, [])
        reference, indexes = grouped[identity]
        if reference != selection.reference:
            raise ValueError("trajectory cache identity maps to conflicting references")
        indexes.append(selection.query_index)
    return tuple(
        TrajectoryQueryGroup(reference, tuple(sorted(indexes)))
        for reference, indexes in grouped.values()
    )


def _validate_selected_trajectory(
    runtime: ForecastBundleRuntime,
    reference: TrajectoryReference,
    trajectory: Trajectory,
) -> None:
    if (
        trajectory.scenario_id != reference.scenario_id
        or trajectory.chlorine_profile_hash
        != reference.chlorine_schedule_hash
        or trajectory.arsenic_schedule_hash
        != reference.arsenic_schedule_hash
    ):
        raise ValueError(
            f"selected training trajectory content mismatches cache identity: "
            f"{reference.cache_dir}"
        )
    time_seconds = np.asarray(trajectory.time_seconds, dtype=np.int64)
    if time_seconds[0] != 0:
        raise ValueError(
            "selected trajectory must use zero-based episode time for query phases"
        )
    if not np.all(np.diff(time_seconds) == runtime.timestep_seconds):
        raise ValueError("selected trajectory timestep does not match forecast bundle")


def _generate_prepared_record(
    runtime: ForecastBundleRuntime,
    selection: QuerySelection,
    trajectory: Trajectory,
    prepared_trajectory: PreparedForecastTrajectory,
    policy_observations: np.ndarray,
    reward_state: dict[str, Any],
    config: GenerationConfig,
) -> TeacherRecord:
    query_index = selection.query_index
    prepared = runtime.query_from_prepared(prepared_trajectory, query_index)
    current_injection = float(trajectory.injection[query_index])
    current_chlorine = np.asarray(trajectory.chlorine[query_index], dtype=np.float32)

    def evaluate(raw_actions: np.ndarray) -> CEMBatchEvaluation:
        injections = actions_to_injections(raw_actions, current_injection)
        predicted = runtime.predict_candidates(prepared, injections, current_chlorine)
        returns = score_forecast_predictions(
            predicted,
            raw_actions,
            injections,
            reward_state=reward_state,
            trajectory=trajectory,
            query_index=query_index,
            timestep_seconds=runtime.timestep_seconds,
            discount_factor=config.discount_factor,
        )
        return CEMBatchEvaluation(returns=returns, predicted_chlorine=predicted)

    cem_seed = stable_query_seed(config.seed, selection)
    result = optimize_cem_action_sequence(
        evaluate,
        candidates=config.candidates,
        iterations=config.iterations,
        elite_fraction=config.elite_fraction,
        seed=cem_seed,
    )
    best_injections = actions_to_injections(result.best_actions, current_injection)
    return TeacherRecord(
        scenario_id=trajectory.scenario_id,
        trajectory_identity=selection.reference.identity,
        chlorine_profile_hash=trajectory.chlorine_profile_hash,
        arsenic_schedule_hash=trajectory.arsenic_schedule_hash,
        calendar_start_day=trajectory.calendar_start_day,
        query_index=query_index,
        time_seconds=int(trajectory.time_seconds[query_index]),
        raw_causal_history=prepared.raw_history,
        policy_observation_history=policy_observation_history_at(
            policy_observations, query_index
        ),
        current_injection=current_injection,
        best_raw_actions=result.best_actions,
        best_injections=best_injections,
        first_action=float(result.best_actions[0]),
        predicted_chlorine=result.best_predicted_chlorine,
        predicted_return=result.best_return,
        reward_state_zlib_base64=base64.b64encode(
            zlib.compress(
                json.dumps(
                    reward_state, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
                level=9,
            )
        ),
        cem_seed=cem_seed,
        diagnostics=result.diagnostics,
    )


def generate_trajectory_records(
    runtime: ForecastBundleRuntime,
    group: TrajectoryQueryGroup,
    config: GenerationConfig,
) -> tuple[TeacherRecord, ...]:
    """Load and featurize once, then advance reward state across query points."""
    trajectory = group.reference.load()
    _validate_selected_trajectory(runtime, group.reference, trajectory)
    prepared_trajectory = runtime.prepare_trajectory(trajectory)
    policy_observations = replay_policy_observations(
        trajectory, through_index=max(group.query_indexes),
        observation_spec=V89_OBSERVATION,
    )
    records: list[TeacherRecord] = []
    for query_index, reward_state in incremental_reward_states(
        trajectory, group.query_indexes
    ):
        selection = QuerySelection(group.reference, query_index)
        records.append(
            _generate_prepared_record(
                runtime,
                selection,
                trajectory,
                prepared_trajectory,
                policy_observations,
                reward_state,
                config,
            )
        )
    return tuple(records)


def generate_record(
    runtime: ForecastBundleRuntime,
    selection: QuerySelection,
    *,
    candidates: int,
    iterations: int,
    elite_fraction: float,
    discount_factor: float,
    seed: int,
) -> TeacherRecord:
    """Compatibility wrapper for a single query using grouped generation."""
    records = generate_trajectory_records(
        runtime,
        TrajectoryQueryGroup(selection.reference, (selection.query_index,)),
        GenerationConfig(
            candidates,
            iterations,
            elite_fraction,
            discount_factor,
            seed,
        ),
    )
    return records[0]


def records_to_arrays(records: tuple[TeacherRecord, ...]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("teacher dataset requires at least one record")
    return {
        "scenario_id": np.asarray([item.scenario_id for item in records], dtype=np.str_),
        "trajectory_identity": np.asarray(
            [item.trajectory_identity for item in records], dtype=np.str_
        ),
        "chlorine_profile_hash": np.asarray(
            [item.chlorine_profile_hash for item in records], dtype=np.str_
        ),
        "arsenic_schedule_hash": np.asarray(
            [item.arsenic_schedule_hash for item in records], dtype=np.str_
        ),
        "calendar_start_day": np.asarray(
            [item.calendar_start_day for item in records], dtype=np.int16
        ),
        "query_index": np.asarray([item.query_index for item in records], dtype=np.int64),
        "time_seconds": np.asarray([item.time_seconds for item in records], dtype=np.int64),
        "raw_causal_history": np.stack(
            [item.raw_causal_history for item in records]
        ).astype(np.float32),
        "policy_observation_history": np.stack(
            [item.policy_observation_history for item in records]
        ).astype(np.float32),
        "current_injection": np.asarray(
            [item.current_injection for item in records], dtype=np.float32
        ),
        "best_raw_actions": np.stack(
            [item.best_raw_actions for item in records]
        ).astype(np.float32),
        "best_injections": np.stack(
            [item.best_injections for item in records]
        ).astype(np.float32),
        "first_action": np.asarray([item.first_action for item in records], dtype=np.float32),
        "predicted_chlorine": np.stack(
            [item.predicted_chlorine for item in records]
        ).astype(np.float32),
        "predicted_return": np.asarray(
            [item.predicted_return for item in records], dtype=np.float32
        ),
        "reward_state_zlib_base64": np.asarray(
            [item.reward_state_zlib_base64 for item in records], dtype=np.bytes_
        ),
        "cem_seed": np.asarray([item.cem_seed for item in records], dtype=np.uint32),
        "cem_iteration_population_max": np.stack(
            [item.diagnostics.population_max for item in records]
        ).astype(np.float32),
        "cem_iteration_population_mean": np.stack(
            [item.diagnostics.population_mean for item in records]
        ).astype(np.float32),
        "cem_iteration_elite_mean": np.stack(
            [item.diagnostics.elite_mean for item in records]
        ).astype(np.float32),
        "cem_iteration_best_so_far": np.stack(
            [item.diagnostics.best_so_far for item in records]
        ).astype(np.float32),
        "cem_iteration_action_mean": np.stack(
            [item.diagnostics.action_mean for item in records]
        ).astype(np.float32),
        "cem_iteration_action_std": np.stack(
            [item.diagnostics.action_std for item in records]
        ).astype(np.float32),
    }


def write_dataset(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    missing = set(REQUIRED_DATASET_ARRAYS) - set(arrays)
    extra = set(arrays) - set(REQUIRED_DATASET_ARRAYS)
    if missing or extra:
        raise ValueError(
            f"dataset arrays do not match schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in converted.values()):
        raise ValueError("dataset arrays must be pickle-free numeric or Unicode arrays")
    counts = {value.shape[0] for value in converted.values() if value.ndim > 0}
    if len(counts) != 1:
        raise ValueError("all dataset arrays must share their leading record dimension")
    count = next(iter(counts))
    scalar_fields = {
        "scenario_id",
        "trajectory_identity",
        "chlorine_profile_hash",
        "arsenic_schedule_hash",
        "calendar_start_day",
        "query_index",
        "time_seconds",
        "current_injection",
        "first_action",
        "predicted_return",
        "reward_state_zlib_base64",
        "cem_seed",
    }
    if any(converted[name].shape != (count,) for name in scalar_fields):
        raise ValueError("dataset scalar fields must have shape (record,)")
    if converted["raw_causal_history"].ndim != 3 or converted[
        "raw_causal_history"
    ].shape[1] != HISTORY_STEPS:
        raise ValueError("raw causal history must have shape (record, 48, feature)")
    if (
        converted["policy_observation_history"].ndim != 3
        or converted["policy_observation_history"].shape[:2]
        != (count, HISTORY_STEPS)
        or converted["policy_observation_history"].shape[2] <= 0
    ):
        raise ValueError(
            "policy observation history must have shape (record, 48, feature)"
        )
    for name in ("best_raw_actions", "best_injections"):
        if converted[name].shape != (count, HORIZON_STEPS):
            raise ValueError(f"{name} must have shape (record, 16)")
    prediction_prefix = (
        (count, 5, HORIZON_STEPS)
        if manifest.get("kind") == "mpc-teacher-cem-ensemble"
        else (count, HORIZON_STEPS)
    )
    if (converted["predicted_chlorine"].ndim != len(prediction_prefix) + 1
            or converted["predicted_chlorine"].shape[:-1] != prediction_prefix):
        raise ValueError(
            f"predicted chlorine must have leading dimensions {prediction_prefix} and a node axis"
        )
    iteration_fields = (
        "cem_iteration_population_max",
        "cem_iteration_population_mean",
        "cem_iteration_elite_mean",
        "cem_iteration_best_so_far",
    )
    iteration_counts = {
        converted[name].shape[1]
        for name in iteration_fields
        if converted[name].ndim == 2 and converted[name].shape[0] == count
    }
    if len(iteration_counts) != 1 or any(
        converted[name].ndim != 2 or converted[name].shape[0] != count
        for name in iteration_fields
    ):
        raise ValueError("CEM scalar diagnostics must share shape (record, iteration)")
    iteration_count = next(iter(iteration_counts))
    for name in ("cem_iteration_action_mean", "cem_iteration_action_std"):
        if converted[name].shape != (count, iteration_count, HORIZON_STEPS):
            raise ValueError(
                f"{name} must have shape (record, iteration, 16)"
            )
    output = Path(output_dir)
    dataset_path = output / "dataset.npz"
    manifest_path = output / "manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing MPC teacher output in {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    temporary_dataset = output / ".dataset.tmp.npz"
    temporary_manifest = output / ".manifest.tmp.json"
    complete_manifest = dict(manifest)
    complete_manifest["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in converted.items()
    }
    try:
        np.savez_compressed(temporary_dataset, **converted)
        temporary_manifest.write_text(
            json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_dataset.replace(dataset_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_dataset.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return dataset_path, manifest_path


def require_fresh_output(output_dir: Path) -> None:
    output = Path(output_dir)
    if (output / "dataset.npz").exists() or (output / "manifest.json").exists():
        raise FileExistsError(
            f"refusing to overwrite existing MPC teacher output in {output}"
        )


_WORKER_RUNTIME: ForecastBundleRuntime | EnsembleForecastRuntime | None = None
_WORKER_CONFIG: GenerationConfig | None = None


def _generation_worker_init(bundle: Path | tuple[Path, ...], config: GenerationConfig) -> None:
    """Create one CPU forecast runtime per worker and leave SIGINT to the parent."""
    global _WORKER_RUNTIME, _WORKER_CONFIG
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(config.seed)
    _WORKER_RUNTIME = (
        EnsembleForecastRuntime(bundle, torch.device("cpu"))
        if isinstance(bundle, tuple)
        else ForecastBundleRuntime(Path(bundle), torch.device("cpu"))
    )
    _WORKER_CONFIG = config


def _generation_worker_run(
    group: TrajectoryQueryGroup,
) -> tuple[TeacherRecord, ...]:
    if _WORKER_RUNTIME is None or _WORKER_CONFIG is None:
        raise RuntimeError("MPC generation worker was not initialized")
    return generate_trajectory_records(_WORKER_RUNTIME, group, _WORKER_CONFIG)


def generate_grouped_records(
    runtime: ForecastBundleRuntime,
    groups: tuple[TrajectoryQueryGroup, ...],
    config: GenerationConfig,
    *,
    workers: int,
) -> tuple[TeacherRecord, ...]:
    """Generate grouped records with deterministic output order and state progress."""
    total_states = sum(len(group.query_indexes) for group in groups)
    records: list[TeacherRecord] = []
    progress = tqdm(
        total=total_states,
        desc="CEM MPC teacher labels",
        unit="state",
        dynamic_ncols=True,
    )
    try:
        if workers == 1:
            for group in groups:
                generated = generate_trajectory_records(runtime, group, config)
                records.extend(generated)
                progress.update(len(generated))
                progress.set_postfix(
                    predicted_return=f"{generated[-1].predicted_return:.4g}"
                )
        else:
            context = mp.get_context("spawn")
            pool = context.Pool(
                workers,
                initializer=_generation_worker_init,
                initargs=(runtime.bundles if isinstance(runtime, EnsembleForecastRuntime) else runtime.bundle, config),
            )
            try:
                for generated in pool.imap_unordered(
                    _generation_worker_run, groups, chunksize=1
                ):
                    records.extend(generated)
                    progress.update(len(generated))
                    progress.set_postfix(
                        predicted_return=f"{generated[-1].predicted_return:.4g}"
                    )
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
    finally:
        progress.close()
    records.sort(
        key=lambda item: (
            item.scenario_id,
            item.trajectory_identity,
            item.query_index,
        )
    )
    return tuple(records)


def build_output_manifest(
    runtime: ForecastBundleRuntime,
    *,
    forecasting_run: Path,
    trajectory_cache: Path,
    state_count: int,
    all_states: bool,
    trajectory_count: int,
    queries_per_trajectory: int,
    candidates: int,
    iterations: int,
    elite_fraction: float,
    seed: int,
    device: torch.device,
    workers: int,
    discount_factor: float,
    query_half_hour_slot_counts: tuple[int, ...],
    forecast_job: ForecastTrainingSpec | None = None,
    source_split: DatasetSplitSpec = forecasting_full.train,
    source_definition: str = DEFAULT_SOURCE_DEFINITION,
    require_out_of_fold: bool = False,
) -> dict[str, Any]:
    if runtime.raw_history_names is None:
        raise RuntimeError("raw history contract is unavailable before query preparation")
    job = build_experiment().jobs[0] if forecast_job is None else forecast_job
    policy_spec = V89_OBSERVATION
    policy_builder = ControlObservationBuilder(
        policy_spec, runtime.node_ids, MAX_INJECTION
    )
    observation_size = policy_builder.shape[0]
    if (
        len(query_half_hour_slot_counts) != HALF_HOUR_SLOTS_PER_DAY
        or any(count < 0 for count in query_half_hour_slot_counts)
        or sum(query_half_hour_slot_counts) != state_count
    ):
        raise ValueError("query half-hour slot counts do not match dataset state count")
    optimizer = cem_optimizer_settings(candidates, iterations, elite_fraction)
    identity = teacher_configuration(
        forecast_job_hash=str(runtime.manifest.get("job_hash")),
        state_count=None if all_states else state_count,
        queries_per_trajectory=queries_per_trajectory,
        seed=seed,
        discount_factor=discount_factor,
        optimizer_kind="cem",
        optimizer=optimizer,
        source_split=source_split,
        source_network=job.dataset.network,
        observation_size=observation_size,
    )
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": "mpc-teacher-cem",
        "teacher_configuration": identity,
        "teacher_configuration_hash": config_hash(identity),
        "source_split": "train",
        "source_dataset": {
            "definition": source_definition,
            "network": job.dataset.network,
            "contract": canonical_data(source_split),
        },
        "forecasting_run": str(Path(forecasting_run).resolve()),
        "forecasting_bundle": str(runtime.bundle.resolve()),
        "forecast_job_hash": runtime.manifest.get("job_hash"),
        "forecast_training_dataset_contract": runtime.manifest.get("job", {}).get(
            "dataset"
        ),
        "trajectory_cache_root": str(Path(trajectory_cache).resolve()),
        "node_ids": list(runtime.node_ids),
        "source_ids": list(runtime.source_ids),
        "hydraulic_timestep_seconds": runtime.timestep_seconds,
        "history_steps": HISTORY_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "raw_history_feature_contract": {
            "names": list(runtime.raw_history_names),
            "transforms": [
                {"feature": name, "transform": "identity", "scale": None}
                for name in runtime.raw_history_names
            ],
            "stored_representation": "raw-feature-values",
            "forecast_model_preprocessing": {
                "method": "forecast-training-split-mean-std",
                "stored": "raw-physical",
                "applied_at": "forecast-model-input-boundary",
                "statistics": "forecasting_bundle/normalization.json",
            },
        },
        "policy_observation_contract": {
            "array": "policy_observation_history",
            "builder": policy_builder.describe(),
            "spec": canonical_data(policy_spec),
            "history_steps": HISTORY_STEPS,
            "observation_size": observation_size,
            "max_injection": MAX_INJECTION,
            "normalized": False,
            "stored_representation": "v89-builder-output",
            "replay": (
                "reset builder once per cached trajectory, then build rows 0 "
                "through query inclusive in simulator order"
            ),
            "alignment": (
                "history[-1] is the query-state observation and first_action "
                "is the action applied after that observation"
            ),
            "stateful_features_preserved": [
                "previous_injection",
                "attention_steps_remaining",
            ],
        },
        "forecast_history_feature_contract": runtime.manifest[
            "history_feature_contract"
        ],
        "forecast_future_known_feature_contract": runtime.manifest[
            "future_known_feature_contract"
        ],
        "selection": {
            "strategy": (
                "per-trajectory-temporal-stratified-training-states"
                if all_states
                else "temporal-stratified-pool-then-half-hour-stratified-subset"
            ),
            "state_count": state_count,
            "trajectory_count": trajectory_count,
            "queries_per_trajectory": queries_per_trajectory,
            "selection_seed": seed,
            "per_trajectory_sampling": {
                "strategy": "one-seeded-query-per-equal-valid-time-stratum",
                "temporal_strata": queries_per_trajectory,
                "fixed_spacing": False,
                "valid_query_minimum_index": HISTORY_STEPS - 1,
                "valid_query_maximum": "trajectory_rows - horizon_steps - 1",
                "clock_slot_balance": (
                    "random choice among candidates in the globally "
                    "least-represented available half-hour slots"
                ),
                "seed": seed,
            },
            "daily_half_hour_distribution": {
                "slot_duration_seconds": HALF_HOUR_SECONDS,
                "slot_count": HALF_HOUR_SLOTS_PER_DAY,
                "slot_definition": "floor(episode_time_seconds / 1800) modulo 48",
                "query_count_by_slot": list(query_half_hour_slot_counts),
                "covered_slots": [
                    slot
                    for slot, count in enumerate(query_half_hour_slot_counts)
                    if count > 0
                ],
                "sampled_state_balancing": (
                    None if all_states else "round-robin-across-seeded-slot-order"
                ),
            },
        },
        "cem": {
            **optimizer,
            "elite_count": min(
                candidates, max(2, int(np.ceil(candidates * elite_fraction)))
            ),
        },
        "action_contract": action_contract(),
        "reward": reward_contract(discount_factor),
        "seed": seed,
        "device": str(device),
        "execution": {
            "workers": workers,
            "parallel_unit": "trajectory",
            "trajectory_loaded_once": True,
            "features_built_once_per_trajectory": True,
            "policy_observations_replayed_once_per_trajectory": True,
            "incremental_reward_replay": True,
        },
        "limitations": {
            "planning_horizon_hours": (
                HORIZON_STEPS * runtime.timestep_seconds / 3600.0
            ),
            "terminal_value_approximation": False,
            "candidate_epanet_rollouts": False,
        },
        "out_of_fold": {
            "required": require_out_of_fold,
            "source_scenarios": list(source_split.scenarios),
            "forecast_training_scenarios": list(job.dataset.train.scenarios),
        },
    }


def generate_dataset(
    args: argparse.Namespace,
    *,
    forecast_job: ForecastTrainingSpec,
    source_split: DatasetSplitSpec,
    source_definition: str,
    require_out_of_fold: bool,
    source_phase: str = "train",
    ensemble_bundles: tuple[Path, ...] | None = None,
    manifest_updates: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    _validate_cli_values(args)
    seed_everything(args.seed)
    device = resolve_generation_device(args.device, args.workers)
    forecasting_run = _resolve_project_path(args.forecasting_run)
    trajectory_cache = _resolve_project_path(args.trajectory_cache)
    bundle = forecasting_run / "best"
    runtime = (
        ForecastBundleRuntime(bundle, device) if ensemble_bundles is None
        else EnsembleForecastRuntime(ensemble_bundles, device)
    )
    identity = teacher_configuration(
        forecast_job_hash=str(runtime.manifest.get("job_hash")),
        state_count=args.state_count,
        queries_per_trajectory=args.queries_per_trajectory,
        seed=args.seed,
        discount_factor=args.discount_factor,
        optimizer_kind="cem",
        optimizer=cem_optimizer_settings(
            args.candidates,
            args.iterations,
            args.elite_fraction,
        ),
        source_split=source_split,
        source_network=forecast_job.dataset.network,
    )
    output_dir = (
        PROJECT_ROOT
        / "data/datasets/mpc-teacher-cem"
        / config_hash(identity)
        if args.output_dir is None
        else _resolve_project_path(args.output_dir)
    )
    require_fresh_output(output_dir)
    selections = select_training_queries(
        cache_root=trajectory_cache,
        state_count=args.state_count,
        queries_per_trajectory=args.queries_per_trajectory,
        seed=args.seed,
        bundle_manifest=runtime.manifest,
        forecast_job=forecast_job,
        source_split=source_split,
        require_out_of_fold=require_out_of_fold,
        source_phase=source_phase,
    )
    groups = group_query_selections(selections)
    config = GenerationConfig(
        candidates=args.candidates,
        iterations=args.iterations,
        elite_fraction=args.elite_fraction,
        discount_factor=args.discount_factor,
        seed=args.seed,
    )
    if args.workers > 1 and args.device == "auto":
        print(
            f"workers={args.workers}: using independent CPU forecast runtimes; "
            "accelerator sharing is disabled"
        )
    records = generate_grouped_records(
        runtime,
        groups,
        config,
        workers=args.workers,
    )
    arrays = records_to_arrays(records)
    query_half_hour_slot_counts = np.bincount(
        np.asarray(
            [
                (record.time_seconds // HALF_HOUR_SECONDS)
                % HALF_HOUR_SLOTS_PER_DAY
                for record in records
            ],
            dtype=np.int64,
        ),
        minlength=HALF_HOUR_SLOTS_PER_DAY,
    )
    manifest = build_output_manifest(
        runtime,
        forecasting_run=forecasting_run,
        trajectory_cache=trajectory_cache,
        state_count=len(selections),
        all_states=args.state_count is None,
        trajectory_count=len(
            {selection.reference.identity for selection in selections}
        ),
        queries_per_trajectory=args.queries_per_trajectory,
        candidates=args.candidates,
        iterations=args.iterations,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
        device=device,
        workers=args.workers,
        discount_factor=args.discount_factor,
        query_half_hour_slot_counts=tuple(
            int(value) for value in query_half_hour_slot_counts
        ),
        forecast_job=forecast_job,
        source_split=source_split,
        source_definition=source_definition,
        require_out_of_fold=require_out_of_fold,
    )
    if manifest_updates is not None:
        manifest.update(manifest_updates)
    if ensemble_bundles is not None:
        manifest["selection"]["strategy"] = "per-trajectory-temporal-stratified-validation-states"
        preprocessing = manifest["raw_history_feature_contract"]["forecast_model_preprocessing"]
        preprocessing["statistics"] = "each ensemble member's bundle/normalization.json"
    return write_dataset(output_dir, arrays, manifest)


def _input_spec(raw: dict[str, Any]) -> ForecastInputSpec:
    allowed = {item.name for item in fields(ForecastFeatureSpec)}

    def feature(value: dict[str, Any]) -> ForecastFeatureSpec:
        prepared = {key: item for key, item in value.items() if key in allowed}
        prepared["cycles"] = tuple(prepared.get("cycles", ()))
        prepared["parameters"] = prepared.get("parameters", {})
        prepared["transform_parameters"] = prepared.get("transform_parameters", {})
        return ForecastFeatureSpec(**prepared)

    return ForecastInputSpec(
        history=tuple(feature(item) for item in raw["history"]),
        future_known=tuple(feature(item) for item in raw["future_known"]),
    )


def _resolve_project_path(value: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_cli_values(args: argparse.Namespace) -> None:
    for name in (
        "queries_per_trajectory",
        "candidates",
        "iterations",
        "workers",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.state_count is not None and (
        isinstance(args.state_count, bool)
        or not isinstance(args.state_count, int)
        or args.state_count <= 0
    ):
        raise ValueError("--state-count must be a positive integer when provided")
    if args.candidates < 2:
        raise ValueError("--candidates must be at least two")
    if not 0.0 < args.elite_fraction <= 1.0:
        raise ValueError("--elite-fraction must lie in (0, 1]")
    if not 0.0 <= args.discount_factor <= 1.0:
        raise ValueError("--discount-factor must lie in [0, 1]")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 16-step MPC teacher labels for training states with CEM, "
            "a frozen forecasting bundle, and exact stateful V89 rewards."
        )
    )
    parser.add_argument("--forecasting-run", type=Path, default=DEFAULT_FORECASTING_RUN)
    parser.add_argument(
        "--trajectory-cache", type=Path, default=DEFAULT_TRAJECTORY_CACHE
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "output directory; default: a hash of the complete teacher "
            "configuration under data/datasets/mpc-teacher-cem"
        ),
    )
    parser.add_argument(
        "--state-count",
        type=int,
        default=None,
        help=(
            "optional seeded subset of the temporally stratified training pool; "
            "default: use every selected trajectory query"
        ),
    )
    parser.add_argument(
        "--queries-per-trajectory",
        type=int,
        default=DEFAULT_QUERIES_PER_TRAJECTORY,
        help=(
            "equal episode-time strata and randomly sampled queries per "
            "training trajectory; default: 24"
        ),
    )
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--elite-fraction", type=float, default=DEFAULT_ELITE_FRACTION
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "trajectory worker processes; values above one use independent "
            "CPU model copies"
        ),
    )
    parser.add_argument(
        "--discount-factor", type=float, default=DEFAULT_DISCOUNT_FACTOR
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reference_job = build_experiment().jobs[0]
        dataset_path, manifest_path = generate_dataset(
            args,
            forecast_job=reference_job,
            source_split=forecasting_full.train,
            source_definition=DEFAULT_SOURCE_DEFINITION,
            require_out_of_fold=False,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted: worker processes stopped; no dataset or manifest was written.",
            file=sys.stderr,
        )
        return 130
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"dataset: {dataset_path}")
    print(f"manifest: {manifest_path}")
    print(
        "limitation: the 16-step plan covers 8 hours and has no terminal-value "
        "approximation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
