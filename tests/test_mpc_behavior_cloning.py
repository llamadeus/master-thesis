from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.nn import functional as F

from packages.trainer import RecurrentPPOSpec
from packages.trainer.control.runner import (
    _initialize_pretrained_policy,
    _policy_kwargs,
    _training_vec_normalize,
    _validate_pretrained_policy_contract,
)
from packages.trainer.serialization import canonical_data, config_hash
from packages.trainer.simulation import Trajectory, TrajectoryStore
from tools.control.bc.train_mpc_teacher_recurrent_ppo import (
    HISTORY_STEPS,
    TrajectoryLabels,
    TrajectorySequence,
    TrajectorySequenceLoader,
    _PolicySpaceEnv,
    _epoch,
    _load_dataset,
    _trajectory_batch_pass,
    _trajectory_pass,
    actor_parameters,
    actor_sequence_batch_mean,
    actor_sequence_mean,
    build_model,
    build_parser,
    cache_trajectory_observations,
    fit_vec_normalize,
    generate_final_plots,
    group_trajectory_labels,
    validate_dataset_pair,
    train,
    trajectory_batches,
)
from tools.utils.policy_observation_replay import (
    policy_observation_history_at,
    replay_policy_observations,
)
from tools.mpc.generate_mpc_teacher_dataset_cem import V89_OBSERVATION
from training.datasets.hanoi import forecasting_full
from training.experiments.control_v89_baseline import (
    build_experiment,
)


NODE_COUNT = 31
NODE_IDS = tuple(str(index) for index in range(NODE_COUNT))
OBSERVATION_SIZE = V89_OBSERVATION.factory(
    **V89_OBSERVATION.as_dict()
).observation_size(NODE_IDS)


def _manifest(rows: int, cache_root: Path = Path("/trajectory-cache")) -> dict:
    job = build_experiment().jobs[0]
    observation_spec = V89_OBSERVATION
    observation_size = observation_spec.factory(
        **observation_spec.as_dict()
    ).observation_size(NODE_IDS)
    action_contract = {
        "raw_minimum": -1.0,
        "raw_maximum": 1.0,
        "max_delta": job.max_delta,
        "minimum_injection": 0.0,
        "maximum_injection": job.max_injection,
        "initial_injection": job.initial_injection,
        "cumulative": True,
    }
    reward = {
        "id": job.reward.factory.reward_id,
        "parameters": job.reward.as_dict(),
        "discount_factor": 0.99,
    }
    policy_contract = {
        "array": "policy_observation_history",
        "history_steps": HISTORY_STEPS,
        "observation_size": observation_size,
        "max_injection": job.max_injection,
        "normalized": False,
        "stored_representation": "v89-builder-output",
        "spec": canonical_data(observation_spec),
    }
    identity = {
        "action_contract": action_contract,
        "reward": reward,
        "policy_observation_contract": {
            "max_injection": job.max_injection,
        },
    }
    return {
        "schema_version": 2,
        "kind": "mpc-teacher-cem",
        "trajectory_cache_root": str(cache_root),
        "source_dataset": {
            "network": "hanoi",
            "contract": canonical_data(forecasting_full.train),
        },
        "node_ids": list(NODE_IDS),
        "policy_observation_contract": policy_contract,
        "action_contract": action_contract,
        "reward": reward,
        "teacher_configuration": identity,
        "teacher_configuration_hash": config_hash(identity),
        "arrays": {
            "policy_observation_history": {
                "shape": [rows, HISTORY_STEPS, observation_size]
            },
            "first_action": {"shape": [rows]},
            "scenario_id": {"shape": [rows]},
            "trajectory_identity": {"shape": [rows]},
            "query_index": {"shape": [rows]},
            "fold_index": {"shape": [rows]},
        },
    }


def _trajectory(
    scenario_id: str,
    identity: str,
    *,
    rows: int,
    marker: float,
) -> Trajectory:
    time_seconds = np.arange(rows, dtype=np.int64) * 1_800
    time_axis = np.arange(rows, dtype=np.float32)[:, None]
    chlorine = np.full((rows, NODE_COUNT), 0.28 + marker, dtype=np.float32)
    chlorine += (time_axis % 7) * 0.001
    node_values = np.zeros((rows, NODE_COUNT), dtype=np.float32)
    return Trajectory(
        network="hanoi",
        scenario_id=scenario_id,
        calendar_start_day=12,
        chlorine_profile_hash=f"chlorine-{identity}",
        arsenic_schedule_hash=f"arsenic-{identity}",
        arsenic_schedule_seed=7,
        node_ids=NODE_IDS,
        link_ids=("link",),
        source_ids=("source",),
        time_seconds=time_seconds,
        pressure=node_values.copy(),
        demand=node_values.copy(),
        flow=np.zeros((rows, 1), dtype=np.float32),
        chlorine=chlorine,
        arsenic=node_values.copy(),
        injection=np.linspace(10_000.0, 90_000.0, rows, dtype=np.float32),
        actions=np.zeros(rows, dtype=np.float32),
        event_active=np.zeros(rows, dtype=bool),
        event_strength_by_node=node_values.copy(),
    )


TRAJECTORY_SPECS = (
    ("a", "ta", 54, (47, 51), 0.000),
    ("b", "tb", 57, (47, 49, 54), 0.005),
)


def _write_dataset(root: Path) -> Path:
    cache_root = root / "trajectory-cache"
    records: list[tuple[np.ndarray, float, str, str, int]] = []
    action_value = -0.8
    for scenario, identity, rows, query_indices, marker in TRAJECTORY_SPECS:
        trajectory = _trajectory(scenario, identity, rows=rows, marker=marker)
        TrajectoryStore().save(
            cache_root / "hanoi" / scenario / identity, trajectory
        )
        observations = replay_policy_observations(
            trajectory, observation_spec=V89_OBSERVATION
        )
        for query_index in query_indices:
            records.append(
                (
                    policy_observation_history_at(observations, query_index),
                    action_value,
                    scenario,
                    identity,
                    query_index,
                )
            )
            action_value += 0.3
    # Deliberately interleave trajectories and reverse queries within them.
    order = np.asarray([4, 1, 2, 0, 3], dtype=np.int64)
    records = [records[int(index)] for index in order]
    histories = np.stack([item[0] for item in records]).astype(np.float32)
    actions = np.asarray([item[1] for item in records], dtype=np.float32)
    scenarios = np.asarray([item[2] for item in records], dtype=np.str_)
    identities = np.asarray([item[3] for item in records], dtype=np.str_)
    query_indices = np.asarray([item[4] for item in records], dtype=np.int64)
    np.savez_compressed(
        root / "dataset.npz",
        policy_observation_history=histories,
        first_action=actions,
        scenario_id=scenarios,
        trajectory_identity=identities,
        query_index=query_indices,
        fold_index=np.asarray([index % 5 for index in range(len(records))], dtype=np.int8),
    )
    (root / "manifest.json").write_text(
        json.dumps(_manifest(len(records), cache_root)), encoding="utf-8"
    )
    return cache_root


def _vec_normalize() -> VecNormalize:
    return VecNormalize(
        DummyVecEnv([_PolicySpaceEnv]),
        training=False,
        norm_obs=True,
        norm_reward=True,
    )


def _chunked_predictions(policy, observations: torch.Tensor, chunk_length: int):
    predictions = []
    state = None
    with torch.no_grad():
        for start in range(0, len(observations), chunk_length):
            chunk, state = actor_sequence_mean(
                policy,
                observations[start : start + chunk_length],
                state,
                episode_start=start == 0,
            )
            state = tuple(value.detach() for value in state)
            predictions.append(chunk)
    return torch.cat(predictions), state


class _MemoryLoader:
    def __init__(self, sequences: dict[str, TrajectorySequence]):
        self.sequences = sequences
        self.calls: list[str] = []

    def load(self, labels: TrajectoryLabels) -> TrajectorySequence:
        self.calls.append(labels.trajectory_identity)
        return self.sequences[labels.trajectory_identity]


class MPCBehaviorCloningTests(unittest.TestCase):
    def test_final_evaluations_archive_existing_metrics_and_raw_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/"metrics.json").write_text('{"best_epoch": 1}\n')
            dataset=SimpleNamespace(
                scenarios_for=lambda _split: (
                    SimpleNamespace(monitored_node_ids=("2","3")),
                )
            )
            calls=[]

            def evaluate(*args,**kwargs):
                phase=args[11];raw=kwargs["raw_output_dir"]
                self.assertIs(kwargs["hydraulic_catalog"], hydraulic_catalog)
                raw.mkdir(parents=True)
                (raw/"manifest.json").write_text('{"case_count": 1}\n')
                (raw/"case-00000.npz").write_bytes(b"fixture")
                calls.append((phase,raw))
                return {"mean_reward":1.0 if phase=="validation" else 2.0}

            fake_runner=SimpleNamespace(backend="backend",validate=lambda _job:None)
            hydraulic_catalog=object()
            with mock.patch(
                "packages.trainer.control.runner.ControlRunner",
                return_value=fake_runner,
            ),mock.patch(
                "packages.trainer.control.runner._run_evaluation",
                side_effect=evaluate,
            ),mock.patch(
                "packages.trainer.data.resolve_dataset",
                return_value=dataset,
            ),mock.patch(
                "packages.trainer.simulation.prepare_hydraulic_outflow_cache",
                return_value=hydraulic_catalog,
            ) as prepare_hydraulics:
                generate_final_plots(output,workers=3)

            prepare_hydraulics.assert_called_once_with(
                dataset, "backend", workers=3, splits=("evaluation", "test"),
                description="behavior-cloning evaluation hydraulics",
            )
            self.assertEqual([item[0] for item in calls],["validation","test"])
            summary=json.loads((output/"evaluation/final/metrics.json").read_text())
            self.assertEqual(summary["validation"]["mean_reward"],1.0)
            self.assertEqual(summary["test"]["mean_reward"],2.0)
            self.assertEqual(summary["runtime"]["workers"],3)
            self.assertTrue((output/"evaluation/final/validation/raw/case-00000.npz").is_file())
            self.assertTrue((output/"evaluation/final/test/raw/case-00000.npz").is_file())
            root_metrics=json.loads((output/"metrics.json").read_text())
            self.assertEqual(root_metrics["final_control_evaluation"]["test"]["mean_reward"],2.0)

    def test_cli_uses_explicit_trajectory_batching_and_tbptt_options(self):
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.trajectory_batch_size, 16)
        self.assertEqual(defaults.sequence_chunk_length, 48)
        self.assertIsNone(defaults.trajectory_cache_root)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--batch-size", "2"])

    def test_separate_datasets_reject_scenario_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root)
            dataset = _load_dataset(root)
            with self.assertRaisesRegex(ValueError, "scenarios overlap"):
                validate_dataset_pair(dataset, dataset)

    def test_dataset_loader_preserves_raw_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root)
            with np.load(root / "dataset.npz", allow_pickle=False) as stored:
                expected = np.asarray(
                    stored["policy_observation_history"], dtype=np.float32
                )
            for kind in ("mpc-teacher-cem", "mpc-teacher-mppi"):
                manifest = json.loads((root / "manifest.json").read_text())
                manifest["kind"] = kind
                (root / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                loaded = _load_dataset(root)
                np.testing.assert_array_equal(loaded.histories, expected)

    def test_grouping_and_batch_shuffle_keep_trajectories_chronological(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root)
            dataset = _load_dataset(root)
            groups = group_trajectory_labels(dataset)
            self.assertEqual([len(item.actions) for item in groups], [2, 3])
            for group in groups:
                self.assertTrue(np.all(np.diff(group.query_indices) > 0))
            repeated = tuple(
                TrajectoryLabels(
                    scenario_id=f"s{index}",
                    trajectory_identity=f"t{index}",
                    row_indices=np.asarray([index]),
                    query_indices=np.asarray([47, 48]) + index,
                    actions=np.asarray([0.0, 0.1], dtype=np.float32),
                )
                for index in range(6)
            )
            batches = tuple(
                trajectory_batches(
                    repeated,
                    batch_size=2,
                    shuffle=True,
                    rng=np.random.default_rng(4),
                )
            )
            shuffled = [item for batch in batches for item in batch]
            self.assertNotEqual(
                [item.trajectory_identity for item in shuffled],
                [item.trajectory_identity for item in repeated],
            )
            for item in shuffled:
                self.assertTrue(np.all(np.diff(item.query_indices) > 0))
            memory_loader = _MemoryLoader(
                {
                    item.trajectory_identity: TrajectorySequence(
                        item,
                        np.repeat(
                            np.arange(7, dtype=np.float32)[:, None],
                            OBSERVATION_SIZE,
                            axis=1,
                        ),
                    )
                    for item in repeated
                }
            )
            for item in shuffled:
                sequence = memory_loader.load(item)
                np.testing.assert_array_equal(
                    sequence.observations[:, 0], np.arange(7, dtype=np.float32)
                )

    def test_reconstructed_histories_match_and_cache_failures_are_contextual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = _write_dataset(root)
            dataset = _load_dataset(root)
            labels = group_trajectory_labels(dataset)[0]
            loader = TrajectorySequenceLoader(
                dataset, cache_root=cache_root, network="hanoi"
            )
            sequence = loader.load(labels)
            self.assertEqual(len(sequence.observations), 54)
            with self.assertRaisesRegex(ValueError, "failed to load trajectory cache"):
                TrajectorySequenceLoader(
                    dataset, cache_root=root / "missing", network="hanoi"
                ).load(labels)

            dataset.histories[int(labels.row_indices[0]), 0, 0] += 1.0
            with self.assertRaisesRegex(
                ValueError, "policy_observation_history mismatch.*query_index"
            ):
                loader.load(labels)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = _write_dataset(root)
            dataset = _load_dataset(root)
            labels = group_trajectory_labels(dataset)[0]
            loader = TrajectorySequenceLoader(
                dataset, cache_root=cache_root, network="hanoi"
            )
            metadata_path = loader.cache_path(labels) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["network"] = "not-hanoi"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "network mismatch"):
                loader.load(labels)

    def test_normalization_uses_every_training_timestep_once_and_no_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = _write_dataset(root)
            dataset = _load_dataset(root)
            training_indices = np.flatnonzero(dataset.scenarios == "a")
            training = group_trajectory_labels(dataset, training_indices)
            validation = group_trajectory_labels(
                dataset, np.flatnonzero(dataset.scenarios == "b")
            )
            inner = TrajectorySequenceLoader(
                dataset, cache_root=cache_root, network="hanoi"
            )
            training_sequence = inner.load(training[0])
            validation_sequence = inner.load(validation[0])
            expected = training_sequence.observations
            recording = _MemoryLoader(
                {
                    training[0].trajectory_identity: training_sequence,
                    validation[0].trajectory_identity: validation_sequence,
                }
            )
            vec, timestep_count, observation_cache = fit_vec_normalize(
                recording, training
            )
            try:
                self.assertEqual(recording.calls, [training[0].trajectory_identity])
                self.assertNotIn(validation[0].trajectory_identity, recording.calls)
                self.assertEqual(timestep_count, len(expected))
                self.assertGreater(timestep_count, int(training[0].query_indices[-1]))
                self.assertAlmostEqual(
                    float(vec.obs_rms.count), len(expected) + 1e-4, places=6
                )
                np.testing.assert_allclose(
                    vec.obs_rms.mean,
                    expected.mean(axis=0),
                    rtol=1e-4,
                    atol=1e-5,
                )
                cache_trajectory_observations(
                    recording,
                    validation,
                    observation_cache,
                    description="test validation cache",
                )
                self.assertEqual(
                    recording.calls,
                    [
                        training[0].trajectory_identity,
                        validation[0].trajectory_identity,
                    ],
                )
                self.assertEqual(observation_cache.trajectory_count, 2)
                self.assertEqual(
                    observation_cache.observation_count,
                    len(training_sequence.observations)
                    + len(validation_sequence.observations),
                )
                calls_before_reuse = tuple(recording.calls)
                observation_cache.load(training[0])
                observation_cache.load(validation[0])
                observation_cache.load(training[0])
                self.assertEqual(tuple(recording.calls), calls_before_reuse)
            finally:
                vec.close()

    def test_full_and_chunked_inference_match_with_carried_state(self):
        model = build_model(torch.device("cpu"), 11)
        policy = model.policy
        policy.eval()
        observations = torch.randn(73, OBSERVATION_SIZE)
        with torch.no_grad():
            full, full_state = actor_sequence_mean(
                policy, observations, episode_start=True
            )
        chunked, chunked_state = _chunked_predictions(policy, observations, 17)
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked_state[0], full_state[0])
        torch.testing.assert_close(chunked_state[1], full_state[1])

        batch = torch.stack((observations, observations * 0.5))
        with torch.no_grad():
            batched, _state = actor_sequence_batch_mean(
                policy, batch, episode_start=True
            )
            first, _state = actor_sequence_mean(
                policy, batch[0], episode_start=True
            )
            second, _state = actor_sequence_mean(
                policy, batch[1], episode_start=True
            )
        torch.testing.assert_close(batched[0], first)
        torch.testing.assert_close(batched[1], second)

    def test_trajectory_batch_processes_each_chunk_in_one_recurrent_call(self):
        model = build_model(torch.device("cpu"), 17)
        policy = model.policy
        policy.eval()
        rng = np.random.default_rng(17)
        sequences = []
        for index, length in enumerate((12, 10, 7, 5)):
            labels = TrajectoryLabels(
                scenario_id=f"s{index}",
                trajectory_identity=f"t{index}",
                row_indices=np.asarray([index * 2, index * 2 + 1]),
                query_indices=np.asarray([3, length - 1]),
                actions=np.asarray([-0.25, 0.5], dtype=np.float32),
            )
            sequences.append(
                TrajectorySequence(
                    labels,
                    rng.normal(size=(length, OBSERVATION_SIZE)).astype(
                        np.float32
                    ),
                )
            )
        vec = _vec_normalize()
        recurrent_shapes = []
        original = policy.get_distribution

        def record_distribution(observations, state, episode_starts):
            recurrent_shapes.append(
                (tuple(observations.shape), tuple(state[0].shape))
            )
            return original(observations, state, episode_starts)

        try:
            with mock.patch.object(
                policy, "get_distribution", side_effect=record_distribution
            ):
                batched_totals = _trajectory_batch_pass(
                    policy,
                    sequences,
                    vec,
                    device=torch.device("cpu"),
                    sequence_chunk_length=4,
                    gradient_scale=None,
                )
            self.assertEqual(
                recurrent_shapes,
                [
                    ((16, OBSERVATION_SIZE), (1, 4, 128)),
                    ((16, OBSERVATION_SIZE), (1, 4, 128)),
                    ((8, OBSERVATION_SIZE), (1, 2, 128)),
                ],
            )
            individual_totals = [
                _trajectory_pass(
                    policy,
                    sequence,
                    vec,
                    device=torch.device("cpu"),
                    sequence_chunk_length=4,
                    gradient_scale=None,
                )
                for sequence in sequences
            ]
            self.assertEqual(batched_totals.count, 8)
            self.assertAlmostEqual(
                batched_totals.squared,
                sum(item.squared for item in individual_totals),
                places=6,
            )
            self.assertAlmostEqual(
                batched_totals.absolute,
                sum(item.absolute for item in individual_totals),
                places=6,
            )
        finally:
            vec.close()

    def test_hidden_state_resets_at_episode_start_not_chunk_boundaries(self):
        model = build_model(torch.device("cpu"), 12)
        policy = model.policy
        policy.eval()
        observations = torch.randn(28, OBSERVATION_SIZE)
        with torch.no_grad():
            first, carried = actor_sequence_mean(
                policy, observations[:13], episode_start=True
            )
            continued, _ = actor_sequence_mean(
                policy, observations[13:], carried, episode_start=False
            )
            reset_at_boundary, _ = actor_sequence_mean(
                policy, observations[13:], episode_start=True
            )
            random_state = tuple(torch.randn_like(value) for value in carried)
            reset_from_random, _ = actor_sequence_mean(
                policy,
                observations[:13],
                random_state,
                episode_start=True,
            )
        full, _ = _chunked_predictions(policy, observations, len(observations))
        torch.testing.assert_close(torch.cat((first, continued)), full)
        torch.testing.assert_close(reset_from_random, first)
        self.assertFalse(torch.allclose(continued, reset_at_boundary))

    def test_sparse_loss_uses_only_labels_and_unlabelled_steps_influence_later(self):
        model = build_model(torch.device("cpu"), 13)
        policy = model.policy
        policy.eval()
        observations = np.zeros((12, OBSERVATION_SIZE), dtype=np.float32)
        labels = TrajectoryLabels(
            scenario_id="s",
            trajectory_identity="t",
            row_indices=np.asarray([0, 1]),
            query_indices=np.asarray([3, 9]),
            actions=np.asarray([-0.25, 0.5], dtype=np.float32),
        )
        sequence = TrajectorySequence(labels, observations)
        vec = _vec_normalize()
        try:
            totals = _trajectory_pass(
                policy,
                sequence,
                vec,
                device=torch.device("cpu"),
                sequence_chunk_length=4,
                gradient_scale=None,
            )
            normalized = torch.as_tensor(vec.normalize_obs(observations))
            predictions, _ = _chunked_predictions(policy, normalized, 4)
            targets = torch.as_tensor(labels.actions[:, None])
            expected = F.mse_loss(
                predictions[labels.query_indices],
                targets,
                reduction="sum",
            )
            self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
            self.assertEqual(float(vec.obs_rms.count), 0.0001)
            np.testing.assert_array_equal(vec.obs_rms.mean, 0.0)
            self.assertEqual(totals.count, 2)
            self.assertAlmostEqual(totals.squared, float(expected), places=6)

            changed = normalized.clone()
            changed[4:9] = 5.0
            original_predictions, _ = _chunked_predictions(policy, normalized, 20)
            changed_predictions, _ = _chunked_predictions(policy, changed, 20)
            torch.testing.assert_close(
                original_predictions[3], changed_predictions[3]
            )
            self.assertFalse(
                torch.allclose(original_predictions[9], changed_predictions[9])
            )
        finally:
            vec.close()

    def test_full_trajectory_update_changes_only_actor_mean_path(self):
        model = build_model(torch.device("cpu"), 3)
        policy = model.policy
        with torch.no_grad():
            policy.log_std.fill_(float(np.log(0.05)))
        frozen_before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
            if "lstm_critic" in name or "value_net" in name or name == "log_std"
        }
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        trainable = actor_parameters(policy)
        for parameter in trainable:
            parameter.requires_grad_(True)
        actor_before = [parameter.detach().clone() for parameter in trainable]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)
        labels = TrajectoryLabels(
            scenario_id="s",
            trajectory_identity="t",
            row_indices=np.asarray([0, 1]),
            query_indices=np.asarray([3, 9]),
            actions=np.asarray([-1.0, 1.0], dtype=np.float32),
        )
        sequence = TrajectorySequence(
            labels,
            np.random.default_rng(2).normal(
                size=(12, OBSERVATION_SIZE)
            ).astype(np.float32),
        )
        loader = _MemoryLoader({"t": sequence})
        vec = _vec_normalize()
        try:
            metrics = _epoch(
                model,
                (labels,),
                loader,
                vec,
                device=torch.device("cpu"),
                trajectory_batch_size=1,
                sequence_chunk_length=4,
                optimizer=optimizer,
                rng=np.random.default_rng(5),
                description="test trajectory update",
            )
            self.assertGreaterEqual(metrics["mse_loss"], 0.0)
            self.assertAlmostEqual(
                metrics["mse_loss"], metrics["rmse"] ** 2, places=6
            )
        finally:
            vec.close()
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(actor_before, trainable, strict=True)
            )
        )
        for name, expected in frozen_before.items():
            torch.testing.assert_close(policy.state_dict()[name], expected)
        self.assertIsNone(policy.log_std.grad)

    def test_bc_bundle_vecnormalize_initializes_and_updates_with_ppo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            cache_root = _write_dataset(dataset_root)
            dataset_manifest = json.loads(
                (dataset_root / "manifest.json").read_text()
            )
            dataset_manifest["trajectory_cache_root"] = str(root / "moved-cache")
            (dataset_root / "manifest.json").write_text(
                json.dumps(dataset_manifest), encoding="utf-8"
            )
            validation_root = root / "validation"
            validation_root.mkdir()
            with mock.patch(f"{__name__}.TRAJECTORY_SPECS", tuple(
                (f"val-{scenario}", f"val-{identity}", rows, queries, marker + 1.0)
                for scenario, identity, rows, queries, marker in TRAJECTORY_SPECS
            )):
                _write_dataset(validation_root)
            bundle = root / "bundle"
            train(
                argparse.Namespace(
                    dataset=dataset_root,
                    output=bundle,
                    trajectory_cache_root=cache_root,
                    seed=42,
                    validation_dataset=validation_root,
                    epochs=1,
                    trajectory_batch_size=1,
                    sequence_chunk_length=11,
                    learning_rate=1e-3,
                    initial_action_std=0.05,
                    patience=1,
                    device="cpu",
                )
            )
            manifest = json.loads((bundle / "manifest.json").read_text())
            metrics = json.loads((bundle / "metrics.json").read_text())
            from training import run_principal_v89_pipeline as principal
            configuration = dict(
                teacher_dataset=str(dataset_root), validation_teacher_dataset=str(validation_root),
                seed=42, epochs=1, trajectory_batch_size=1, sequence_chunk_length=11,
                learning_rate=1e-3, initial_action_std=0.05, patience=1,
            )
            self.assertTrue(principal._bc_is_complete(bundle, configuration))
            validation_manifest_path = validation_root / "manifest.json"
            original_manifest = validation_manifest_path.read_text()
            validation_manifest_path.write_text(original_manifest + "\n")
            self.assertFalse(principal._bc_is_complete(bundle, configuration))
            validation_manifest_path.write_text(original_manifest)
            original_strategy = manifest["split"]["strategy"]
            manifest["split"]["strategy"] = "whole-scenario"
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(principal._bc_is_complete(bundle, configuration))
            manifest["split"]["strategy"] = original_strategy
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["actor_training"]["objective"],
                "mse-sparse-query-deterministic-action-mean",
            )
            self.assertIn("best_validation_mse_loss", manifest)
            self.assertIn("best_validation_mse_loss", metrics)
            self.assertIn("mse_loss", manifest["training_history"][0]["training"])
            self.assertIn("mse_loss", manifest["training_history"][0]["validation"])
            self.assertEqual(metrics["runtime"]["device"], "cpu")
            self.assertEqual(metrics["runtime"]["workers"], 1)
            self.assertIn("training_elapsed_seconds", metrics["history"][0])
            self.assertIn("validation_elapsed_seconds", metrics["history"][0])
            self.assertEqual(manifest["split"]["training_scenarios"], ["a", "b"])
            self.assertEqual(manifest["split"]["validation_scenarios"], ["val-a", "val-b"])
            expected_rows = sum(item[2] for item in TRAJECTORY_SPECS)
            expected_labels = sum(len(item[3]) for item in TRAJECTORY_SPECS)
            self.assertEqual(
                manifest["observation_preprocessing"][
                    "training_timestep_observations"
                ],
                expected_rows,
            )
            self.assertFalse(
                manifest["observation_preprocessing"][
                    "validation_observations_included"
                ]
            )
            self.assertEqual(manifest["split"]["training_trajectories"], 2)
            self.assertEqual(manifest["split"]["training_labels"], expected_labels)
            self.assertEqual(
                manifest["actor_training"]["recurrence"], "full-trajectory"
            )
            self.assertEqual(
                manifest["actor_training"]["batch_size_unit"],
                "complete-trajectories-per-optimizer-step",
            )
            self.assertEqual(
                manifest["actor_training"]["sequence_chunk_length"], 11
            )
            self.assertEqual(
                manifest["actor_training"]["chunk_execution"],
                "lockstep-across-active-trajectories",
            )
            observation_cache = manifest["observation_preprocessing"][
                "reconstructed_observation_cache"
            ]
            self.assertTrue(observation_cache["constructed_once"])
            self.assertTrue(observation_cache["reused_across_epochs"])
            self.assertEqual(observation_cache["trajectory_count"], 4)
            self.assertEqual(observation_cache["observation_count"], 2 * (54 + 57))
            self.assertEqual(
                observation_cache["bytes"],
                2 * (54 + 57) * OBSERVATION_SIZE * np.dtype(np.float32).itemsize,
            )
            self.assertEqual(
                manifest["trajectory_cache"]["effective_root"],
                str(cache_root.resolve()),
            )
            self.assertTrue((bundle / "vec_normalize.pkl").is_file())
            checkpoint = bundle / "checkpoints" / "epoch-000001"
            self.assertTrue((checkpoint / "model.zip").is_file())
            self.assertTrue((checkpoint / "vec_normalize.pkl").is_file())
            self.assertTrue((checkpoint / "training_state.pt").is_file())
            self.assertTrue((checkpoint / "metrics.json").is_file())
            self.assertTrue((checkpoint / "manifest.json").is_file())
            state = torch.load(
                checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(state["epoch"], 1)
            self.assertIn("actor_optimizer", state)
            self.assertIn("trajectory_rng", state)
            self.assertEqual(manifest["checkpoints"]["count"], 1)
            self.assertEqual(
                manifest["action_contract"], _manifest(5)["action_contract"]
            )
            self.assertEqual(manifest["reward"], _manifest(5)["reward"])

            source = RecurrentPPO.load(str(bundle / "model.zip"), device="cpu")
            torch.testing.assert_close(
                source.policy.log_std,
                torch.full_like(source.policy.log_std, float(np.log(0.05))),
            )
            with torch.no_grad():
                source.policy.action_net.bias.fill_(0.75)
            source.save(str(bundle / "model"))

            algorithm = RecurrentPPOSpec(
                n_steps=2,
                batch_size=2,
                normalize_observations=True,
                normalize_rewards=True,
                pretrained_policy=bundle,
                device="cpu",
            )
            raw = DummyVecEnv([_PolicySpaceEnv])
            vec = _training_vec_normalize(
                raw,
                algorithm,
                V89_OBSERVATION,
                pretrained_bundle=bundle,
            )
            target = build_model(torch.device("cpu"), 6, vec)
            critic_before = {
                name: value.detach().clone()
                for name, value in target.policy.state_dict().items()
                if "lstm_critic" in name or "value_net" in name
            }
            count_before = float(vec.obs_rms.count)
            initialization = _initialize_pretrained_policy(
                target,
                algorithm,
                bundle=bundle,
                job=build_experiment().jobs[0],
            )
            self.assertEqual(initialization["best_epoch"], 1)
            torch.testing.assert_close(
                target.policy.action_net.bias, source.policy.action_net.bias
            )
            for name, expected in critic_before.items():
                torch.testing.assert_close(target.policy.state_dict()[name], expected)
            self.assertEqual(target.policy.optimizer.state, {})
            target.learn(total_timesteps=2, progress_bar=False)
            self.assertGreater(float(vec.obs_rms.count), count_before)
            vec.close()

    def test_experiment_uses_vecnormalize_without_fixed_feature_transform(self):
        job = build_experiment().jobs[0]
        kwargs = _policy_kwargs(job.algorithm, job.observation, OBSERVATION_SIZE)
        self.assertNotIn("features_extractor_class", kwargs)
        self.assertTrue(job.algorithm.normalize_observations)
        self.assertTrue(job.algorithm.normalize_rewards)
        self.assertEqual(job.reward.factory.__name__, "V89Reward")
        self.assertEqual(job.max_injection, 500_000.0)
        self.assertEqual(job.max_delta, 50_000.0)
        self.assertEqual(job.reward.as_dict(), job.reward.factory().p)

    def test_dataset_rejects_stale_v89_contracts(self):
        manifest = _manifest(5)
        mutations = (
            (
                "maximum injection",
                lambda value: value["action_contract"].update(
                    maximum_injection=499_999.0
                ),
            ),
            (
                "delta",
                lambda value: value["action_contract"].update(max_delta=74_999.0),
            ),
            (
                "observation denominator",
                lambda value: value["policy_observation_contract"].update(
                    max_injection=499_999.0
                ),
            ),
            ("reward", lambda value: value["reward"].update(id="other")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                candidate = json.loads(json.dumps(manifest))
                mutate(candidate)
                with self.assertRaisesRegex(ValueError, "contract|reward|denominator"):
                    from tools.control.bc.train_mpc_teacher_recurrent_ppo import (
                        _validate_manifest,
                    )

                    _validate_manifest(candidate)

    def test_dataset_accepts_v89_reward_and_observation_contract(self):
        from tools.control.bc.train_mpc_teacher_recurrent_ppo import _validate_manifest

        manifest = _manifest(5)
        self.assertEqual(manifest["policy_observation_contract"]["observation_size"], 43)
        _validate_manifest(manifest)
        manifest["reward"]["parameters"]["blend_grace_steps"] = 5.0
        with self.assertRaisesRegex(ValueError, "reward identity"):
            _validate_manifest(manifest)

    def test_v89_dataset_trains_transferable_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            training.mkdir()
            cache_root = _write_dataset(training)
            validation = root / "validation"
            validation.mkdir()
            with mock.patch(f"{__name__}.TRAJECTORY_SPECS", tuple(
                (f"val-{scenario}", f"val-{identity}", rows, queries, marker + 1.0)
                for scenario, identity, rows, queries, marker in TRAJECTORY_SPECS
            )):
                _write_dataset(validation)
            output = root / "bundle"
            train(argparse.Namespace(
                dataset=training,
                output=output,
                trajectory_cache_root=cache_root,
                seed=42,
                validation_dataset=validation,
                epochs=1,
                trajectory_batch_size=1,
                sequence_chunk_length=11,
                learning_rate=1e-3,
                initial_action_std=0.05,
                patience=1,
                device="cpu",
            ))
            manifest = json.loads((output / "manifest.json").read_text())
            model = RecurrentPPO.load(str(output / "model.zip"), device="cpu")
            self.assertEqual(manifest["reward"]["id"], "v89")
            self.assertEqual(manifest["observation_size"], 43)
            self.assertEqual(model.observation_space.shape, (43,))
            self.assertEqual(
                manifest["observation_preprocessing"]["source"],
                "raw-v89-builder-output",
            )
            job = build_experiment().jobs[0]
            algorithm = replace(
                job.algorithm,
                pretrained_policy=output,
                n_steps=2,
                batch_size=2,
                device="cpu",
            )
            vec = _training_vec_normalize(
                DummyVecEnv([lambda: _PolicySpaceEnv(43)]),
                algorithm,
                V89_OBSERVATION,
                pretrained_bundle=output,
            )
            try:
                target = build_model(torch.device("cpu"), 6, vec, observation_size=43)
                initialization = _initialize_pretrained_policy(
                    target, algorithm, bundle=output, job=job
                )
                self.assertEqual(initialization["best_epoch"], 1)
                torch.testing.assert_close(
                    target.policy.action_net.bias, model.policy.action_net.bias
                )
            finally:
                vec.close()

    def test_ppo_rejects_stale_bc_contracts(self):
        job = build_experiment().jobs[0]
        dataset_manifest = _manifest(5)
        bc_manifest = {
            "action_contract": dataset_manifest["action_contract"],
            "policy_observation_contract": dataset_manifest[
                "policy_observation_contract"
            ],
            "reward": dataset_manifest["reward"],
        }
        for field, stale in (
            ("maximum_injection", 499_999.0),
            ("max_delta", 74_999.0),
        ):
            with self.subTest(field=field):
                candidate = json.loads(json.dumps(bc_manifest))
                candidate["action_contract"][field] = stale
                with self.assertRaisesRegex(ValueError, "action contract"):
                    _validate_pretrained_policy_contract(candidate, job)
        candidate = json.loads(json.dumps(bc_manifest))
        candidate["policy_observation_contract"]["max_injection"] = 499_999.0
        with self.assertRaisesRegex(ValueError, "denominator"):
            _validate_pretrained_policy_contract(candidate, job)
        candidate = json.loads(json.dumps(bc_manifest))
        candidate["reward"]["parameters"]["recovery_injection_weight"] = 2.0
        with self.assertRaisesRegex(ValueError, "reward identity"):
            _validate_pretrained_policy_contract(candidate, job)


if __name__ == "__main__":
    unittest.main()
