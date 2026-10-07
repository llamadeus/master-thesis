#!/usr/bin/env python3
"""Behavior-clone the MPC teacher and plot the final RecurrentPPO actor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.nn import functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.serialization import canonical_data, config_hash
from packages.trainer.observations import ControlObservationBuilder
from packages.trainer.simulation import TrajectoryStore
from tools.utils.policy_observation_replay import (
    policy_observation_history_at,
    replay_policy_observations,
)
from training.experiments.control_v89_baseline import \
    build_experiment as build_control_experiment
from tools.mpc.generate_cross_fit_mpc_teacher_dataset_cem import (
    DEFAULT_OUTPUT as MPC_TEACHER_DATASET,
    DEFAULT_TEACHER_CONFIGURATION_HASH,
)
from tools.mpc.generate_validation_mpc_teacher_dataset_cem import (
    DEFAULT_OUTPUT as MPC_VALIDATION_DATASET,
)

HISTORY_STEPS = 48
ACTION_SIZE = 1
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
VEC_NORMALIZE_CLIP_OBS = 10.0
VEC_NORMALIZE_CLIP_REWARD = 10.0
VEC_NORMALIZE_EPSILON = 1e-8
NODE_COUNT = 31
_CONTROL_OBSERVATION = build_control_experiment().jobs[0].observation
OBSERVATION_SIZE = _CONTROL_OBSERVATION.factory(
    **_CONTROL_OBSERVATION.as_dict()
).observation_size(tuple(f"node-{index}" for index in range(NODE_COUNT)))
BC_POLICY_BUNDLE = Path("data/weights/control/mpc-teacher-bc") / config_hash({
    "training_teacher": DEFAULT_TEACHER_CONFIGURATION_HASH,
    "validation_teacher": MPC_VALIDATION_DATASET.name,
    "validation": "separate-validation-dataset",
})


class _PolicySpaceEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)

    def __init__(self, observation_size: int = OBSERVATION_SIZE):
        super().__init__()
        self.observation_size = int(observation_size)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_size, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_size, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


@dataclass(frozen=True)
class LoadedDataset:
    histories: np.ndarray
    actions: np.ndarray
    scenarios: np.ndarray
    trajectory_identities: np.ndarray
    query_indices: np.ndarray
    manifest: dict[str, Any]
    manifest_sha256: str


@dataclass(frozen=True)
class TrajectoryLabels:
    scenario_id: str
    trajectory_identity: str
    row_indices: np.ndarray
    query_indices: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class TrajectorySequence:
    labels: TrajectoryLabels
    observations: np.ndarray


class TrajectorySequenceLoader:
    """Load and replay one complete cached trajectory at a time."""

    def __init__(
        self,
        dataset: LoadedDataset,
        *,
        cache_root: Path,
        network: str,
    ) -> None:
        self.dataset = dataset
        self.cache_root = Path(cache_root)
        self.network = network
        self.store = TrajectoryStore()
        self.expected_node_ids = tuple(
            str(value) for value in dataset.manifest["node_ids"]
        )

    def cache_path(self, labels: TrajectoryLabels) -> Path:
        return (
            self.cache_root
            / self.network
            / labels.scenario_id
            / labels.trajectory_identity
        )

    def load(self, labels: TrajectoryLabels) -> TrajectorySequence:
        cache_path = self.cache_path(labels)
        try:
            trajectory = self.store.load(cache_path)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                "failed to load trajectory cache for "
                f"scenario={labels.scenario_id!r}, "
                f"trajectory_identity={labels.trajectory_identity!r}: {cache_path}: {exc}"
            ) from exc
        if trajectory.network != self.network:
            raise ValueError(
                f"trajectory cache network mismatch at {cache_path}: "
                f"expected {self.network!r}, got {trajectory.network!r}"
            )
        if trajectory.scenario_id != labels.scenario_id:
            raise ValueError(
                f"trajectory cache scenario mismatch at {cache_path}: "
                f"expected {labels.scenario_id!r}, got {trajectory.scenario_id!r}"
            )
        if trajectory.node_ids != self.expected_node_ids:
            raise ValueError(
                f"trajectory cache node order mismatch at {cache_path}"
            )
        row_count = len(trajectory.time_seconds)
        if int(labels.query_indices[-1]) >= row_count:
            raise ValueError(
                f"label query index {int(labels.query_indices[-1])} is outside "
                f"the {row_count}-row trajectory cache at {cache_path}"
            )
        try:
            target = build_control_experiment().jobs[0]
            observations = replay_policy_observations(
                trajectory,
                observation_spec=target.observation,
                maximum_injection=target.max_injection,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"failed to reconstruct policy observations from {cache_path}: {exc}"
            ) from exc
        if not np.all(np.isfinite(observations)):
            raise ValueError(
                f"reconstructed policy observations are non-finite at {cache_path}"
            )
        for row, query_index in zip(
            labels.row_indices, labels.query_indices, strict=True
        ):
            actual = policy_observation_history_at(observations, int(query_index))
            expected = self.dataset.histories[int(row)]
            if not np.array_equal(actual, expected):
                unequal = np.argwhere(actual != expected)
                first = tuple(int(value) for value in unequal[0])
                maximum_error = float(
                    np.max(
                        np.abs(
                            actual.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                )
                raise ValueError(
                    "reconstructed policy_observation_history mismatch for "
                    f"scenario={labels.scenario_id!r}, "
                    f"trajectory_identity={labels.trajectory_identity!r}, "
                    f"query_index={int(query_index)} at {cache_path}; "
                    f"first differing history/feature index={first}, "
                    f"maximum absolute error={maximum_error:.9g}"
                )
        return TrajectorySequence(labels=labels, observations=observations)


class TrajectoryObservationCache:
    """RAM cache of reconstructed observations, keyed by trajectory identity."""

    def __init__(self) -> None:
        self._sequences: dict[str, TrajectorySequence] = {}

    def add(self, sequence: TrajectorySequence) -> None:
        identity = sequence.labels.trajectory_identity
        if identity in self._sequences:
            raise ValueError(
                f"trajectory observation cache already contains {identity!r}"
            )
        self._sequences[identity] = sequence

    def load(self, labels: TrajectoryLabels) -> TrajectorySequence:
        try:
            sequence = self._sequences[labels.trajectory_identity]
        except KeyError as exc:
            raise ValueError(
                "reconstructed trajectory observations are not cached for "
                f"{labels.trajectory_identity!r}"
            ) from exc
        cached = sequence.labels
        if (
            cached.scenario_id != labels.scenario_id
            or not np.array_equal(cached.query_indices, labels.query_indices)
            or not np.array_equal(cached.actions, labels.actions)
        ):
            raise ValueError(
                f"cached trajectory labels disagree for {labels.trajectory_identity!r}"
            )
        return sequence

    @property
    def trajectory_count(self) -> int:
        return len(self._sequences)

    @property
    def observation_count(self) -> int:
        return sum(len(item.observations) for item in self._sequences.values())

    @property
    def observation_bytes(self) -> int:
        return sum(item.observations.nbytes for item in self._sequences.values())


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_dataset(directory: Path) -> LoadedDataset:
    root = _project_path(directory).resolve()
    manifest_path = root / "manifest.json"
    arrays_path = root / "dataset.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise ValueError(
            f"MPC teacher dataset is incomplete: expected {manifest_path} and {arrays_path}"
        )
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid MPC teacher manifest: {manifest_path}") from exc
    _validate_manifest(manifest)
    try:
        with np.load(arrays_path, allow_pickle=False) as stored:
            histories = np.asarray(stored["policy_observation_history"], dtype=np.float32)
            actions = np.asarray(stored["first_action"], dtype=np.float32)
            scenarios = np.asarray(stored["scenario_id"])
            trajectory_identities = np.asarray(stored["trajectory_identity"])
            query_indices = np.asarray(stored["query_index"], dtype=np.int64)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid MPC teacher arrays: {arrays_path}") from exc
    expected_observation_size = int(
        manifest["policy_observation_contract"]["observation_size"]
    )
    if histories.ndim != 3 or histories.shape[1:] != (
        HISTORY_STEPS,
        expected_observation_size,
    ):
        raise ValueError(
            "policy_observation_history does not match the manifest observation width"
        )
    rows = histories.shape[0]
    if (
        actions.shape != (rows,)
        or scenarios.shape != (rows,)
        or trajectory_identities.shape != (rows,)
        or query_indices.shape != (rows,)
    ):
        raise ValueError("MPC teacher label and scenario arrays disagree with history rows")
    if not np.issubdtype(scenarios.dtype, np.str_) or not np.issubdtype(
        trajectory_identities.dtype, np.str_
    ):
        raise ValueError("scenario_id and trajectory_identity must be string arrays")
    if np.any(query_indices < HISTORY_STEPS - 1):
        raise ValueError("query_index cannot precede its complete policy history")
    for start in range(0, rows, 1024):
        if not np.all(np.isfinite(histories[start : start + 1024])):
            raise ValueError("policy observation histories contain non-finite values")
    if not np.all(np.isfinite(actions)) or np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError("first_action must contain finite values in [-1, 1]")
    declared = manifest.get("arrays", {})
    for name, array in (
        ("policy_observation_history", histories),
        ("first_action", actions),
        ("scenario_id", scenarios),
        ("trajectory_identity", trajectory_identities),
        ("query_index", query_indices),
    ):
        shape = declared.get(name, {}).get("shape")
        if shape is not None and tuple(shape) != array.shape:
            raise ValueError(f"manifest shape for {name} disagrees with dataset.npz")
    return LoadedDataset(
        histories,
        actions,
        scenarios,
        trajectory_identities,
        query_indices,
        manifest,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    supported_kinds = {"mpc-teacher-cem", "mpc-teacher-mppi", "mpc-teacher-cem-ensemble"}
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") not in supported_kinds
    ):
        raise ValueError("dataset is not a supported schema-v2 MPC teacher bundle")
    contract = manifest.get("policy_observation_contract", {})
    target = build_control_experiment().jobs[0]
    node_ids = tuple(str(value) for value in manifest.get("node_ids", ()))
    if not node_ids:
        raise ValueError("dataset does not declare its policy observation node order")
    builder = ControlObservationBuilder(
        target.observation,
        node_ids,
        target.max_injection,
    )
    expected_spec = canonical_data(target.observation)
    if (
        contract.get("array") != "policy_observation_history"
        or contract.get("history_steps") != HISTORY_STEPS
        or contract.get("observation_size") != builder.shape[0]
        or contract.get("normalized") is not False
        or contract.get("stored_representation") != "v89-builder-output"
        or contract.get("spec") != expected_spec
    ):
        raise ValueError("dataset policy observations do not match the raw control contract")
    action = manifest.get("action_contract", {})
    expected_action = {
        "raw_minimum": -1.0,
        "raw_maximum": 1.0,
        "max_delta": target.max_delta,
        "minimum_injection": 0.0,
        "maximum_injection": target.max_injection,
        "initial_injection": target.initial_injection,
        "cumulative": True,
    }
    if action != expected_action:
        raise ValueError(
            "dataset action contract does not match the target control job"
        )
    if contract.get("max_injection") != target.max_injection:
        raise ValueError(
            "dataset policy-observation injection denominator does not match "
            "the target control job"
        )
    reward = manifest.get("reward", {})
    if (
        reward.get("id") != target.reward.factory.reward_id
        or reward.get("parameters") != target.reward.as_dict()
    ):
        raise ValueError("dataset reward identity does not match the target control job")
    identity = manifest.get("teacher_configuration")
    identity_hash = manifest.get("teacher_configuration_hash")
    if not isinstance(identity, dict) or identity_hash != config_hash(identity):
        raise ValueError("dataset teacher configuration identity is missing or stale")
    if (
        identity.get("action_contract") != action
        or identity.get("reward") != reward
        or identity.get("policy_observation_contract", {}).get("max_injection")
        != contract.get("max_injection")
    ):
        raise ValueError(
            "dataset teacher identity disagrees with its saved control contracts"
        )
    node_ids = manifest.get("node_ids")
    if not isinstance(node_ids, list) or len(node_ids) != NODE_COUNT:
        raise ValueError("dataset must use the 31-node Hanoi policy observation order")
    source_dataset = manifest.get("source_dataset")
    if not isinstance(source_dataset, dict):
        raise ValueError("dataset manifest is missing its source_dataset contract")
    source_contract = source_dataset.get("contract")
    if not isinstance(source_contract, dict):
        raise ValueError("dataset manifest is missing its source_dataset contract")
    source_network = source_dataset.get("network", source_contract.get("network"))
    if not isinstance(source_network, str) or not source_network:
        raise ValueError("dataset source contract is missing its trajectory network")
    if not isinstance(manifest.get("trajectory_cache_root"), str) or not manifest[
        "trajectory_cache_root"
    ]:
        raise ValueError("dataset manifest is missing trajectory_cache_root")


def validate_dataset_pair(training: LoadedDataset, validation: LoadedDataset) -> None:
    for key in ("node_ids", "action_contract", "policy_observation_contract", "reward"):
        if training.manifest[key] != validation.manifest[key]:
            raise ValueError(f"training and validation dataset contracts differ: {key}")
    if set(training.scenarios) & set(validation.scenarios):
        raise ValueError("training and validation scenarios overlap")
    if set(training.trajectory_identities) & set(validation.trajectory_identities):
        raise ValueError("training and validation trajectory identities overlap")
    train_forecast = training.manifest["teacher_configuration"].get("forecast", {})
    if train_forecast.get("strategy") == "scenario-out-of-fold":
        expected = [fold["job_hash"] for fold in train_forecast["folds"]]
        actual = validation.manifest["teacher_configuration"].get("forecast", {})
        if (actual.get("strategy") != "five-fold-ensemble-mean-return"
                or actual.get("job_hashes") != expected):
            raise ValueError("validation ensemble does not match the training forecasters")
        from training.datasets.hanoi import forecasting_full

        for dataset, split in ((training, forecasting_full.train),
                               (validation, forecasting_full.validation)):
            if (dataset.manifest["source_dataset"]["contract"] != canonical_data(split)
                    or set(dataset.scenarios) != set(split.scenarios)):
                raise ValueError("BC dataset does not cover its complete scenario split")
        training_folds = training.manifest.get("cross_fitting", {}).get("folds", [])
        members = validation.manifest.get("ensemble", {}).get("members", [])
        if len(training_folds) != 5 or len(members) != 5:
            raise ValueError("BC teachers must identify all five forecast checkpoints")
        for fold, member in zip(training_folds, members, strict=True):
            if fold.get("labeling_forecast_bundle_hashes") != member.get("bundle_hashes"):
                raise ValueError("BC teachers use different forecast checkpoint contents")


def group_trajectory_labels(
    dataset: LoadedDataset, indices: np.ndarray | None = None
) -> tuple[TrajectoryLabels, ...]:
    selected = (
        np.arange(len(dataset.actions), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    if selected.ndim != 1 or not len(selected):
        raise ValueError("trajectory grouping requires at least one teacher label")
    if np.any(selected < 0) or np.any(selected >= len(dataset.actions)):
        raise ValueError("trajectory grouping row index is outside the dataset")
    grouped: dict[str, tuple[str, list[int]]] = {}
    for row_value in selected:
        row = int(row_value)
        identity = str(dataset.trajectory_identities[row])
        scenario = str(dataset.scenarios[row])
        if identity not in grouped:
            grouped[identity] = (scenario, [])
        expected_scenario, rows = grouped[identity]
        if scenario != expected_scenario:
            raise ValueError(
                f"trajectory_identity {identity!r} appears in multiple scenarios: "
                f"{expected_scenario!r} and {scenario!r}"
            )
        rows.append(row)
    result: list[TrajectoryLabels] = []
    for identity, (scenario, rows) in grouped.items():
        row_indices = np.asarray(rows, dtype=np.int64)
        order = np.argsort(dataset.query_indices[row_indices], kind="stable")
        row_indices = row_indices[order]
        query_indices = dataset.query_indices[row_indices].copy()
        if np.any(query_indices[1:] == query_indices[:-1]):
            duplicate_offset = int(
                np.flatnonzero(query_indices[1:] == query_indices[:-1])[0]
            )
            duplicate = int(query_indices[duplicate_offset])
            raise ValueError(
                f"trajectory_identity {identity!r} has duplicate query_index {duplicate}"
            )
        result.append(
            TrajectoryLabels(
                scenario_id=scenario,
                trajectory_identity=identity,
                row_indices=row_indices,
                query_indices=query_indices,
                actions=dataset.actions[row_indices].copy(),
            )
        )
    result.sort(key=lambda item: (item.scenario_id, item.trajectory_identity))
    return tuple(result)


def _resolve_trajectory_cache(
    dataset: LoadedDataset, override: Path | None
) -> tuple[Path, str]:
    configured = (
        Path(dataset.manifest["trajectory_cache_root"])
        if override is None
        else Path(override)
    )
    cache_root = _project_path(configured).resolve()
    source_dataset = dataset.manifest["source_dataset"]
    network = source_dataset.get("network")
    if network is None:
        network = source_dataset.get("contract", {}).get("network")
    if not isinstance(network, str) or not network:
        raise ValueError("MPC teacher source dataset does not declare its network")
    return cache_root, network


def trajectory_batches(
    trajectories: Sequence[TrajectoryLabels],
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[TrajectoryLabels, ...]]:
    if batch_size < 1:
        raise ValueError("trajectory batch size must be positive")
    if shuffle:
        if rng is None:
            raise ValueError("shuffled trajectory batches require a random generator")
        order = rng.permutation(len(trajectories))
    else:
        order = np.arange(len(trajectories), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        yield tuple(
            trajectories[int(index)] for index in order[start : start + batch_size]
        )


def fit_vec_normalize(
    loader: TrajectorySequenceLoader,
    training_trajectories: Sequence[TrajectoryLabels],
) -> tuple[VecNormalize, int, TrajectoryObservationCache]:
    if not training_trajectories:
        raise ValueError("VecNormalize fitting requires training trajectories")
    first_sequence = loader.load(training_trajectories[0])
    observation_size = int(first_sequence.observations.shape[1])
    vec = VecNormalize(
        DummyVecEnv([lambda: _PolicySpaceEnv(observation_size)]),
        training=False,
        norm_obs=True,
        norm_reward=True,
        clip_obs=VEC_NORMALIZE_CLIP_OBS,
        clip_reward=VEC_NORMALIZE_CLIP_REWARD,
        epsilon=VEC_NORMALIZE_EPSILON,
    )
    timestep_count = 0
    cache = TrajectoryObservationCache()
    for index, labels in enumerate(tqdm(
        training_trajectories,
        desc="Fit full-trajectory observation normalization",
        unit="trajectory",
        leave=False,
    )):
        sequence = first_sequence if index == 0 else loader.load(labels)
        vec.obs_rms.update(sequence.observations)
        timestep_count += len(sequence.observations)
        cache.add(sequence)
    return vec, timestep_count, cache


def cache_trajectory_observations(
    loader: TrajectorySequenceLoader,
    trajectories: Sequence[TrajectoryLabels],
    cache: TrajectoryObservationCache,
    *,
    description: str,
) -> None:
    for labels in tqdm(
        trajectories,
        desc=description,
        unit="trajectory",
        leave=False,
    ):
        cache.add(loader.load(labels))


def build_model(
    device: torch.device,
    seed: int,
    env=None,
    *,
    observation_size: int = OBSERVATION_SIZE,
) -> RecurrentPPO:
    return RecurrentPPO(
        "MlpLstmPolicy",
        env if env is not None else _PolicySpaceEnv(observation_size),
        n_steps=2,
        batch_size=2,
        seed=seed,
        device=device,
        policy_kwargs={
            "lstm_hidden_size": LSTM_HIDDEN_SIZE,
            "n_lstm_layers": LSTM_LAYERS,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
        verbose=0,
    )


def actor_parameters(policy) -> tuple[torch.nn.Parameter, ...]:
    parameters = (
        *policy.lstm_actor.parameters(),
        *policy.mlp_extractor.policy_net.parameters(),
        *policy.action_net.parameters(),
    )
    return tuple(parameters)


def initial_actor_state(
    policy,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(
        policy.lstm_actor.num_layers,
        batch_size,
        policy.lstm_actor.hidden_size,
        device=device,
        dtype=dtype,
    )
    return zeros, zeros.clone()


def actor_sequence_batch_mean(
    policy,
    observations: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    episode_start: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Evaluate chronological sequences in parallel and carry every actor state."""
    expected_shape = getattr(getattr(policy, "observation_space", None), "shape", None)
    expected_size = (
        int(expected_shape[0])
        if isinstance(expected_shape, tuple) and len(expected_shape) == 1
        else int(observations.shape[2]) if observations.ndim == 3 else -1
    )
    if observations.ndim != 3 or observations.shape[2] != expected_size:
        raise ValueError(
            "batched actor observations must have shape "
            f"(trajectories, timesteps, {expected_size})"
        )
    batch_size, sequence_length, _feature_count = observations.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError("actor sequence batch cannot be empty")
    if state is None:
        state = initial_actor_state(
            policy,
            batch_size=batch_size,
            device=observations.device,
            dtype=observations.dtype,
        )
    episode_starts = torch.zeros(
        (batch_size, sequence_length),
        device=observations.device,
        dtype=observations.dtype,
    )
    if episode_start:
        episode_starts[:, 0] = 1.0
    distribution, next_state = policy.get_distribution(
        observations.reshape(batch_size * sequence_length, -1),
        state,
        episode_starts.reshape(-1),
    )
    means = distribution.distribution.mean.reshape(
        batch_size, sequence_length, -1
    )
    return means, next_state


def actor_sequence_mean(
    policy,
    observations: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    episode_start: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Evaluate one chronological sequence and return its carried actor state."""
    if observations.ndim != 2:
        raise ValueError("single actor observations must have rank two")
    means, next_state = actor_sequence_batch_mean(
        policy,
        observations.unsqueeze(0),
        state,
        episode_start=episode_start,
    )
    return means[0], next_state


@dataclass
class _MetricTotals:
    absolute: float = 0.0
    squared: float = 0.0
    clipped_absolute: float = 0.0
    clipped_squared: float = 0.0
    count: int = 0

    def add(self, other: "_MetricTotals") -> None:
        self.absolute += other.absolute
        self.squared += other.squared
        self.clipped_absolute += other.clipped_absolute
        self.clipped_squared += other.clipped_squared
        self.count += other.count


def _trajectory_batch_pass(
    policy,
    sequences: Sequence[TrajectorySequence],
    vec_normalize: VecNormalize,
    *,
    device: torch.device,
    sequence_chunk_length: int,
    gradient_scale: int | None,
) -> _MetricTotals:
    if not sequences:
        raise ValueError("trajectory pass requires at least one sequence")
    lengths = np.asarray(
        [len(sequence.observations) for sequence in sequences], dtype=np.int64
    )
    totals = _MetricTotals()
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    previous_active = tuple(range(len(sequences)))
    observation_size = int(sequences[0].observations.shape[1])
    if any(sequence.observations.shape[1] != observation_size for sequence in sequences):
        raise ValueError("trajectory observation widths differ")
    for start in range(0, int(lengths.max()), sequence_chunk_length):
        active = tuple(
            index for index, length in enumerate(lengths) if start < int(length)
        )
        if state is not None and active != previous_active:
            keep = torch.as_tensor(
                [
                    position
                    for position, index in enumerate(previous_active)
                    if index in active
                ],
                device=device,
                dtype=torch.long,
            )
            state = tuple(value.index_select(1, keep) for value in state)
        chunk_steps = min(
            sequence_chunk_length,
            max(int(lengths[index]) - start for index in active),
        )
        raw_observations = np.zeros(
            (len(active), chunk_steps, observation_size), dtype=np.float32
        )
        batch_indexes: list[int] = []
        time_indexes: list[int] = []
        target_actions: list[float] = []
        for batch_index, trajectory_index in enumerate(active):
            sequence = sequences[trajectory_index]
            valid_steps = min(
                chunk_steps, len(sequence.observations) - start
            )
            raw_observations[batch_index, :valid_steps] = sequence.observations[
                start : start + valid_steps
            ]
            labels = sequence.labels
            label_start = int(
                np.searchsorted(labels.query_indices, start, side="left")
            )
            label_end = int(
                np.searchsorted(
                    labels.query_indices, start + valid_steps, side="left"
                )
            )
            for label_offset in range(label_start, label_end):
                batch_indexes.append(batch_index)
                time_indexes.append(
                    int(labels.query_indices[label_offset]) - start
                )
                target_actions.append(float(labels.actions[label_offset]))
        has_labels = bool(target_actions)
        normalized = vec_normalize.normalize_obs(raw_observations)
        observations = torch.as_tensor(
            normalized, device=device, dtype=torch.float32
        )
        with torch.set_grad_enabled(gradient_scale is not None and has_labels):
            predictions, state = actor_sequence_batch_mean(
                policy,
                observations,
                state,
                episode_start=start == 0,
            )
            if has_labels:
                selected_predictions = predictions[
                    torch.as_tensor(
                        batch_indexes, device=device, dtype=torch.long
                    ),
                    torch.as_tensor(
                        time_indexes, device=device, dtype=torch.long
                    ),
                ]
                targets = torch.as_tensor(
                    target_actions,
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(1)
                loss_sum = F.mse_loss(
                    selected_predictions,
                    targets,
                    reduction="sum",
                )
                if gradient_scale is not None:
                    (loss_sum / gradient_scale).backward()
                errors = selected_predictions.detach() - targets
                clipped_errors = (
                    selected_predictions.detach().clamp(-1.0, 1.0) - targets
                )
                totals.absolute += float(errors.abs().sum())
                totals.squared += float(errors.square().sum())
                totals.clipped_absolute += float(clipped_errors.abs().sum())
                totals.clipped_squared += float(clipped_errors.square().sum())
                totals.count += targets.numel()
        state = tuple(value.detach() for value in state)
        previous_active = active
    expected_labels = sum(len(sequence.labels.actions) for sequence in sequences)
    if totals.count != expected_labels:
        raise RuntimeError(
            f"trajectory batch supervised {totals.count} of {expected_labels} labels"
        )
    return totals


def _trajectory_pass(
    policy,
    sequence: TrajectorySequence,
    vec_normalize: VecNormalize,
    *,
    device: torch.device,
    sequence_chunk_length: int,
    gradient_scale: int | None,
) -> _MetricTotals:
    """Compatibility wrapper for tests and one-trajectory callers."""
    return _trajectory_batch_pass(
        policy,
        (sequence,),
        vec_normalize,
        device=device,
        sequence_chunk_length=sequence_chunk_length,
        gradient_scale=gradient_scale,
    )


def _epoch(
    model: RecurrentPPO,
    trajectories: Sequence[TrajectoryLabels],
    loader: TrajectoryObservationCache,
    vec_normalize: VecNormalize,
    *,
    device: torch.device,
    trajectory_batch_size: int,
    sequence_chunk_length: int,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
    description: str,
) -> dict[str, float]:
    policy = model.policy
    policy.train(optimizer is not None)
    totals = _MetricTotals()
    batches = trajectory_batches(
        trajectories,
        batch_size=trajectory_batch_size,
        shuffle=optimizer is not None,
        rng=rng,
    )
    batch_count = (
        len(trajectories) + trajectory_batch_size - 1
    ) // trajectory_batch_size
    for batch in tqdm(
        batches,
        total=batch_count,
        desc=description,
        unit="trajectory-batch",
        leave=False,
    ):
        labels_in_batch = sum(len(item.actions) for item in batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        sequences = tuple(loader.load(labels) for labels in batch)
        totals.add(
            _trajectory_batch_pass(
                policy,
                sequences,
                vec_normalize,
                device=device,
                sequence_chunk_length=sequence_chunk_length,
                gradient_scale=labels_in_batch if optimizer is not None else None,
            )
        )
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(actor_parameters(policy), 0.5)
            optimizer.step()
    if totals.count == 0:
        raise ValueError("behavior-cloning split is empty")
    return {
        "mse_loss": totals.squared / totals.count,
        "mae": totals.absolute / totals.count,
        "rmse": float(np.sqrt(totals.squared / totals.count)),
        "clipped_mae": totals.clipped_absolute / totals.count,
        "clipped_rmse": float(np.sqrt(totals.clipped_squared / totals.count)),
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def train(args: argparse.Namespace) -> Path:
    if not np.isfinite(args.initial_action_std) or not 0.0 < args.initial_action_std <= 1.0:
        raise ValueError("--initial-action-std must lie in (0, 1]")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if min(
        args.epochs,
        args.trajectory_batch_size,
        args.sequence_chunk_length,
        args.patience,
    ) < 1:
        raise ValueError(
            "epochs, trajectory batch size, sequence chunk length, and patience "
            "must be positive"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    output = _project_path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"behavior-cloning output is not empty: {output}")
    dataset = _load_dataset(args.dataset)
    dataset_manifest = dataset.manifest
    dataset_manifest_sha256 = dataset.manifest_sha256
    observation_size = int(dataset.histories.shape[2])
    validation_dataset = _load_dataset(args.validation_dataset)
    validate_dataset_pair(dataset, validation_dataset)
    validation_manifest = validation_dataset.manifest
    validation_manifest_sha256 = validation_dataset.manifest_sha256
    train_indices = np.arange(len(dataset.scenarios))
    validation_indices = np.arange(len(validation_dataset.scenarios))
    training_scenarios = tuple(sorted(set(dataset.scenarios)))
    validation_scenarios = tuple(sorted(set(validation_dataset.scenarios)))
    training_trajectories = group_trajectory_labels(dataset, train_indices)
    validation_trajectories = group_trajectory_labels(validation_dataset, validation_indices)
    cache_root, network = _resolve_trajectory_cache(
        dataset, args.trajectory_cache_root
    )
    sequence_loader = TrajectorySequenceLoader(
        dataset,
        cache_root=cache_root,
        network=network,
    )
    vec_normalize, training_timestep_count, observation_cache = fit_vec_normalize(
        sequence_loader, training_trajectories
    )
    validation_cache_root, validation_network = _resolve_trajectory_cache(
        validation_dataset, getattr(args, "validation_trajectory_cache_root", None)
    )
    validation_loader = TrajectorySequenceLoader(
        validation_dataset, cache_root=validation_cache_root, network=validation_network
    )
    cache_trajectory_observations(
        validation_loader,
        validation_trajectories,
        observation_cache,
        description="Cache validation trajectory observations",
    )
    cached_observation_count = observation_cache.observation_count
    cached_observation_bytes = observation_cache.observation_bytes
    del sequence_loader, validation_loader, validation_dataset
    del dataset
    model = build_model(
        device,
        args.seed,
        vec_normalize,
        observation_size=observation_size,
    )
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(args.initial_action_std)))
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    trainable = actor_parameters(model.policy)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    trajectory_rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, 0x42435452])
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_normalization = checkpoint_root / "vec_normalize.pkl"
    vec_normalize.save(str(checkpoint_normalization))
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        training_started = time.perf_counter()
        training = _epoch(
            model,
            training_trajectories,
            observation_cache,
            vec_normalize,
            device=device,
            trajectory_batch_size=args.trajectory_batch_size,
            sequence_chunk_length=args.sequence_chunk_length,
            optimizer=optimizer,
            rng=trajectory_rng,
            description=f"BC epoch {epoch} train",
        )
        training_elapsed = time.perf_counter() - training_started
        validation_started = time.perf_counter()
        validation = _epoch(
            model,
            validation_trajectories,
            observation_cache,
            vec_normalize,
            device=device,
            trajectory_batch_size=args.trajectory_batch_size,
            sequence_chunk_length=args.sequence_chunk_length,
            optimizer=None,
            rng=None,
            description=f"BC epoch {epoch} validation",
        )
        validation_elapsed = time.perf_counter() - validation_started
        print(
            f"epoch {epoch:03d}: train mse={training['mse_loss']:.6f}, "
            f"validation mse={validation['mse_loss']:.6f}, mae={validation['mae']:.6f}"
        )
        if validation["mse_loss"] < best_loss:
            best_loss = validation["mse_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.policy.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "training_elapsed_seconds": training_elapsed,
            "validation_elapsed_seconds": validation_elapsed,
            "elapsed_seconds": training_elapsed + validation_elapsed,
        })
        epoch_output = checkpoint_root / f"epoch-{epoch:06d}"
        epoch_output.mkdir(parents=True, exist_ok=False)
        model.save(str(epoch_output / "model"))
        vec_normalize.save(str(epoch_output / "vec_normalize.pkl"))
        torch.save(
            {
                "schema_version": 1,
                "epoch": epoch,
                "policy": model.policy.state_dict(),
                "actor_optimizer": optimizer.state_dict(),
                "best_policy": best_state,
                "best_validation_mse_loss": best_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "trajectory_rng": trajectory_rng.bit_generator.state,
            },
            epoch_output / "training_state.pt",
        )
        (epoch_output / "metrics.json").write_text(
            json.dumps(history[-1], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (epoch_output / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "recurrent-ppo-behavior-cloning-checkpoint",
                    "epoch": epoch,
                    "seed": args.seed,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "validation_dataset_manifest_sha256": validation_manifest_sha256,
                    "artifacts": {
                        "model": "model.zip",
                        "vec_normalize": "vec_normalize.pkl",
                        "training_state": "training_state.pt",
                        "metrics": "metrics.json",
                    },
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("behavior cloning produced no checkpoint")
    model.policy.load_state_dict(best_state)
    model.save(str(output / "model"))
    vec_normalize.save(str(output / "vec_normalize.pkl"))
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_mse_loss": best_loss,
        "history": history,
        "checkpoint_count": len(history),
        "checkpoint_layout": "checkpoints/epoch-<six-digit-epoch>",
        "runtime": {
            "device": str(device),
            "workers": int(getattr(args, "workers", 1)),
            "training_seconds": sum(
                float(item["training_elapsed_seconds"]) for item in history
            ),
            "validation_seconds": sum(
                float(item["validation_elapsed_seconds"]) for item in history
            ),
        },
    }
    manifest = {
        "schema_version": 2,
        "kind": "recurrent-ppo-behavior-cloning",
        "dataset": str(_project_path(args.dataset).resolve()),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "validation_dataset": str(_project_path(args.validation_dataset).resolve()),
        "validation_dataset_manifest_sha256": validation_manifest_sha256,
        "validation_teacher_configuration_hash": validation_manifest["teacher_configuration_hash"],
        "validation_ensemble": copy.deepcopy(validation_manifest.get("ensemble")),
        "teacher_configuration_hash": dataset_manifest[
            "teacher_configuration_hash"
        ],
        "cross_fitting": copy.deepcopy(dataset_manifest.get("cross_fitting")),
        "action_contract": copy.deepcopy(dataset_manifest["action_contract"]),
        "policy_observation_contract": copy.deepcopy(
            dataset_manifest["policy_observation_contract"]
        ),
        "reward": copy.deepcopy(dataset_manifest["reward"]),
        "observation_preprocessing": {
            "kind": "vec-normalize",
            "artifact": "vec_normalize.pkl",
            "source": "raw-v89-builder-output",
            "fit_split": "behavior-cloning-training-scenarios",
            "fit_contract": "every-timestep-once-from-complete-training-trajectories",
            "full_trajectory": True,
            "training_trajectory_count": len(training_trajectories),
            "training_timestep_observations": training_timestep_count,
            "validation_observations_included": False,
            "reconstructed_observation_cache": {
                "storage": "ram-float32",
                "constructed_once": True,
                "trajectory_count": observation_cache.trajectory_count,
                "observation_count": cached_observation_count,
                "bytes": cached_observation_bytes,
                "reused_across_epochs": True,
            },
            "norm_obs": True,
            "clip_obs": VEC_NORMALIZE_CLIP_OBS,
            "epsilon": VEC_NORMALIZE_EPSILON,
            "continue_updating_during_ppo": True,
        },
        "observation_size": observation_size,
        "history_steps": HISTORY_STEPS,
        "action_size": ACTION_SIZE,
        "policy": {
            "name": "MlpLstmPolicy",
            "lstm_hidden_size": LSTM_HIDDEN_SIZE,
            "n_lstm_layers": LSTM_LAYERS,
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "initial_action_std": args.initial_action_std,
            "initial_log_std": float(np.log(args.initial_action_std)),
            "initial_action_std_source": getattr(args, "initial_action_std_source", "provided"),
        },
        "actor_training": {
            "objective": "mse-sparse-query-deterministic-action-mean",
            "learning_rate": args.learning_rate,
            "trajectory_batch_size": args.trajectory_batch_size,
            "batch_size_unit": "complete-trajectories-per-optimizer-step",
            "sequence_chunk_length": args.sequence_chunk_length,
            "recurrence": "full-trajectory",
            "episode_reset": "zero-hidden-and-cell-only-at-cached-trajectory-start",
            "supervision": "first-action-labels-at-query-index-only",
            "unlabelled_observations_update_recurrent_state": True,
            "chunk_boundary": {
                "hidden_and_cell_carried": True,
                "hidden_and_cell_detached": True,
                "hidden_and_cell_reset": False,
            },
            "optimizer_step": "after-each-trajectory-batch",
            "chunk_execution": "lockstep-across-active-trajectories",
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "seed": args.seed,
        },
        "split": {
            "strategy": "separate-validation-dataset",
            "training_scenarios": list(training_scenarios),
            "validation_scenarios": list(validation_scenarios),
            "training_trajectories": len(training_trajectories),
            "validation_trajectories": len(validation_trajectories),
            "training_labels": len(train_indices),
            "validation_labels": len(validation_indices),
            "trajectory_identity_leakage": False,
        },
        "trajectory_cache": {
            "manifest_root": dataset_manifest["trajectory_cache_root"],
            "validation_manifest_root": validation_manifest["trajectory_cache_root"],
            "validation_effective_root": str(validation_cache_root),
            "validation_network": validation_network,
            "override": (
                None
                if args.trajectory_cache_root is None
                else str(_project_path(args.trajectory_cache_root).resolve())
            ),
            "effective_root": str(cache_root),
            "network": network,
            "layout": "<root>/<network>/<scenario_id>/<trajectory_identity>",
            "loading": "one-raw-trajectory-at-a-time",
        },
        "daily_half_hour_distribution": dataset_manifest.get("selection", {}).get(
            "daily_half_hour_distribution"
        ),
        "best_epoch": best_epoch,
        "best_validation_mse_loss": best_loss,
        "training_history": history,
        "checkpoints": {
            "count": len(history),
            "layout": "checkpoints/epoch-<six-digit-epoch>",
            "self_contained": True,
            "includes_actor_optimizer_and_rng": True,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def generate_final_plots(output: Path, *, workers: int) -> None:
    """Run and archive the standard control validation and test evaluations."""
    from packages.trainer.control.runner import ControlRunner, _run_evaluation
    from packages.trainer.data import resolve_dataset
    from packages.trainer.simulation import prepare_hydraulic_outflow_cache
    from stable_baselines3.common.utils import get_device

    job = build_control_experiment().jobs[0]
    runner = ControlRunner()
    runner.validate(job)
    dataset = resolve_dataset(job.dataset)
    hydraulic_catalog = prepare_hydraulic_outflow_cache(
        dataset, runner.backend, workers=workers, splits=("evaluation", "test"),
        description="behavior-cloning evaluation hydraulics",
    )
    node_ids = dataset.scenarios_for("train")[0].monitored_node_ids
    model_path = output / "model.zip"
    normalization_path = output / "vec_normalize.pkl"
    final_plots = output / "plots" / "final"
    evaluation_root = output / "evaluation" / "final"
    results = {}
    runtime = {}

    def resolve(reference):
        raise RuntimeError(f"unexpected control evaluation dependency: {reference}")

    for split, phase in (("evaluation", "validation"), ("test", "test")):
        started = time.perf_counter()
        results[phase] = _run_evaluation(
            job,
            dataset,
            node_ids,
            runner.backend,
            {},
            resolve,
            None,
            None,
            model_path,
            normalization_path,
            split,
            phase,
            final_plots / phase,
            workers,
            hydraulic_catalog=hydraulic_catalog,
            raw_output_dir=evaluation_root / phase / "raw",
        )
        runtime[phase] = time.perf_counter() - started
        (evaluation_root / phase).mkdir(parents=True, exist_ok=True)
        (evaluation_root / phase / "metrics.json").write_text(
            json.dumps(results[phase], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": 1,
        "validation": results["validation"],
        "test": results["test"],
        "runtime": {
            "validation_seconds": runtime["validation"],
            "test_seconds": runtime["test"],
            "device": str(get_device(job.algorithm.device)),
            "workers": int(workers),
        },
        "raw_timeseries": {
            phase: f"{phase}/raw" for phase in ("validation", "test")
        },
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final_control_evaluation"] = summary
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=MPC_TEACHER_DATASET)
    parser.add_argument("--validation-dataset", type=Path, default=MPC_VALIDATION_DATASET)
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="finish after exporting the BC bundle; skip validation/test rollouts",
    )
    parser.add_argument("--output", type=Path, default=BC_POLICY_BUNDLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--trajectory-batch-size",
        type=_positive_integer,
        default=16,
        help="complete trajectories accumulated before each optimizer step (default: 16)",
    )
    parser.add_argument(
        "--sequence-chunk-length",
        type=_positive_integer,
        default=48,
        help="TBPTT timesteps; recurrent state is carried and detached (default: 48)",
    )
    parser.add_argument(
        "--trajectory-cache-root",
        type=Path,
        default=None,
        help="override a relocated trajectory_cache_root saved in the teacher manifest",
    )
    parser.add_argument("--validation-trajectory-cache-root", type=Path, default=None,
                        help="override the validation teacher's trajectory cache root")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--initial-action-std",
        type=float,
        default=0.05,
        help="Initial PPO exploration standard deviation in normalized action units.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--workers",
        type=_positive_integer,
        default=1,
        help="worker processes for final control evaluation plots (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = train(args)
        if not args.skip_plots:
            generate_final_plots(output, workers=args.workers)
    except KeyboardInterrupt:
        print("behavior cloning interrupted", file=sys.stderr)
        return 130
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"behavior-cloned policy: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
