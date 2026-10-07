from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import tools.mpc.generate_mpc_teacher_dataset_cem as mpc_tool
from packages.trainer.forecasting.materialization import TrajectoryReference
from packages.trainer.observations import ControlObservationBuilder
from packages.trainer.simulation import Trajectory
from training.datasets.hanoi import forecasting_full
from training.rewards.v89 import DEFAULT_PARAMETERS as V89_PARAMETERS, V89Reward
from training.experiments.control_v89_baseline import (
    build_experiment as build_v89_experiment,
)
from training.experiments.forecasting_encoder_decoder_lstm_cross_fit import (
    FOLDS,
    build_experiment as build_cross_fit_experiment,
)
from tools.mpc.generate_mpc_teacher_dataset_cem import (
    ACTION_CONTROL_POINT_OFFSETS,
    ACTION_SEQUENCE_LENGTH,
    CEMBatchEvaluation,
    DEFAULT_OUTPUT,
    ForecastBundleRuntime,
    HORIZON_STEPS,
    MAX_INJECTION,
    MIN_CEM_STD,
    POLICY_OBSERVATION_SIZE,
    PreparedForecastQuery,
    PreparedForecastTrajectory,
    REQUIRED_DATASET_ARRAYS,
    actions_to_injections,
    build_output_manifest,
    build_parser,
    expand_action_control_point_std,
    expand_action_control_points,
    group_query_selections,
    half_hour_slot_for_query_index,
    incremental_reward_states,
    optimize_cem,
    optimize_cem_action_sequence,
    policy_observation_history_at,
    replay_policy_observations,
    require_completed_best_bundle,
    reconstruct_reward_state,
    resolve_generation_device,
    select_training_queries,
    simulation_sample_from_trajectory,
    stratified_query_sample,
    temporally_stratified_query_pool,
    write_dataset,
)


class _RecordingForecast(torch.nn.Module):
    def __init__(self, node_count: int):
        super().__init__()
        self.node_count = node_count
        self.history_shape: tuple[int, ...] | None = None
        self.future: np.ndarray | None = None

    def forward(self, history: torch.Tensor, future: torch.Tensor):
        self.history_shape = tuple(history.shape)
        self.future = future.detach().cpu().numpy().copy()
        chlorine = future[..., :1].expand(-1, -1, self.node_count)
        return {"chlorine": chlorine}


class CEMTeacherDatasetTests(unittest.TestCase):
    @staticmethod
    def _trajectory(rows: int = 120, node_count: int = 2) -> Trajectory:
        time = np.arange(rows, dtype=np.int64) * 1_800
        chlorine = np.full((rows, node_count), 0.3, dtype=np.float32)
        chlorine[40:55, 0] = 0.19
        node_values = np.zeros((rows, node_count), dtype=np.float32)
        link_values = np.zeros((rows, 1), dtype=np.float32)
        return Trajectory(
            network="synthetic",
            scenario_id="scenario",
            calendar_start_day=0,
            chlorine_profile_hash="chlorine",
            arsenic_schedule_hash="arsenic",
            arsenic_schedule_seed=1,
            node_ids=tuple(str(index + 1) for index in range(node_count)),
            link_ids=("L",),
            source_ids=("1",),
            time_seconds=time,
            pressure=node_values,
            demand=node_values,
            flow=link_values,
            chlorine=chlorine,
            arsenic=node_values,
            injection=np.linspace(0.0, 250_000.0, rows, dtype=np.float32),
            actions=np.sin(np.arange(rows)).astype(np.float32),
            event_active=np.zeros(rows, dtype=bool),
            event_strength_by_node=node_values,
        )

    def test_cem_is_deterministic_includes_hold_and_preserves_global_best(self) -> None:
        populations: list[np.ndarray] = []

        def evaluate(actions: np.ndarray) -> CEMBatchEvaluation:
            populations.append(actions.copy())
            target = np.linspace(-0.5, 0.5, actions.shape[1], dtype=np.float32)
            returns = -np.sum((actions - target) ** 2, axis=1)
            if len(populations) > 1:
                returns = returns - 100.0
            return CEMBatchEvaluation(returns, actions[..., None])

        first = optimize_cem(
            evaluate,
            horizon=4,
            candidates=12,
            iterations=3,
            elite_fraction=0.25,
            seed=71,
        )
        first_populations = [item.copy() for item in populations]
        populations.clear()
        second = optimize_cem(
            evaluate,
            horizon=4,
            candidates=12,
            iterations=3,
            elite_fraction=0.25,
            seed=71,
        )

        np.testing.assert_array_equal(first.best_actions, second.best_actions)
        np.testing.assert_array_equal(
            first.diagnostics.action_mean, second.diagnostics.action_mean
        )
        self.assertEqual(first.best_return, second.best_return)
        self.assertTrue(
            all(np.array_equal(population[0], np.zeros(4)) for population in populations)
        )
        self.assertTrue(
            all(
                current >= previous
                for previous, current in zip(
                    first.diagnostics.best_so_far,
                    first.diagnostics.best_so_far[1:],
                )
            )
        )
        initial_returns = -np.sum(
            (
                first_populations[0]
                - np.linspace(-0.5, 0.5, 4, dtype=np.float32)
            )
            ** 2,
            axis=1,
        )
        np.testing.assert_array_equal(
            first.best_actions,
            first_populations[0][int(np.argmax(initial_returns))],
        )

    def test_action_control_points_expand_piecewise_linearly(self) -> None:
        points = np.asarray(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.5, -1.0],
            dtype=np.float32,
        )
        expected = np.asarray(
            [
                0.0,
                0.2,
                0.4,
                0.6,
                0.7,
                0.8,
                0.8 + 0.2 / 3.0,
                0.8 + 0.4 / 3.0,
                1.0,
                0.5,
                0.0,
                -0.5,
                -0.625,
                -0.75,
                -0.875,
                -1.0,
            ],
            dtype=np.float32,
        )

        actual = expand_action_control_points(points)

        self.assertEqual(ACTION_SEQUENCE_LENGTH, 8)
        self.assertEqual(
            ACTION_CONTROL_POINT_OFFSETS, (0, 1, 2, 3, 5, 8, 11, 15)
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(
            actual[np.asarray(ACTION_CONTROL_POINT_OFFSETS)], points
        )

    def test_action_control_point_expansion_is_batched_bounded_and_holds(self) -> None:
        points = np.asarray(
            [[0.0] * ACTION_SEQUENCE_LENGTH, [2.0, -2.0] * 4],
            dtype=np.float32,
        )

        expanded = expand_action_control_points(points)

        self.assertEqual(expanded.shape, (2, HORIZON_STEPS))
        np.testing.assert_array_equal(expanded[0], np.zeros(HORIZON_STEPS))
        self.assertTrue(np.all(expanded >= -1.0))
        self.assertTrue(np.all(expanded <= 1.0))
        np.testing.assert_array_equal(
            expanded[1, np.asarray(ACTION_CONTROL_POINT_OFFSETS)],
            np.asarray([1.0, -1.0] * 4, dtype=np.float32),
        )

    def test_action_control_point_std_projects_independent_marginals(self) -> None:
        standard_deviations = np.arange(1, 9, dtype=np.float32)

        expanded = expand_action_control_point_std(standard_deviations)

        np.testing.assert_array_equal(
            expanded[np.asarray(ACTION_CONTROL_POINT_OFFSETS)], standard_deviations
        )
        expected_step_four = np.sqrt(0.5**2 * 4.0**2 + 0.5**2 * 5.0**2)
        self.assertAlmostEqual(float(expanded[4]), expected_step_four, places=6)

    def test_action_sequence_cem_optimizes_eight_points_and_returns_16_steps(self) -> None:
        populations: list[np.ndarray] = []

        def evaluate(actions: np.ndarray) -> CEMBatchEvaluation:
            populations.append(actions.copy())
            returns = -np.sum(np.square(actions - 0.25), axis=1)
            return CEMBatchEvaluation(returns, actions[..., None])

        result = optimize_cem_action_sequence(
            evaluate,
            candidates=12,
            iterations=3,
            elite_fraction=0.25,
            seed=71,
        )
        repeated = optimize_cem_action_sequence(
            evaluate,
            candidates=12,
            iterations=3,
            elite_fraction=0.25,
            seed=71,
        )

        self.assertEqual(result.best_actions.shape, (HORIZON_STEPS,))
        self.assertEqual(result.best_predicted_chlorine.shape, (HORIZON_STEPS, 1))
        self.assertEqual(result.diagnostics.action_mean.shape, (3, HORIZON_STEPS))
        self.assertEqual(result.diagnostics.action_std.shape, (3, HORIZON_STEPS))
        np.testing.assert_array_equal(result.best_actions, repeated.best_actions)
        np.testing.assert_array_equal(
            result.diagnostics.action_mean, repeated.diagnostics.action_mean
        )
        for population in populations:
            self.assertEqual(population.shape, (12, HORIZON_STEPS))
            np.testing.assert_array_equal(population[0], np.zeros(HORIZON_STEPS))
            np.testing.assert_allclose(
                population,
                expand_action_control_points(
                    population[:, np.asarray(ACTION_CONTROL_POINT_OFFSETS)]
                ),
                rtol=0.0,
                atol=1e-7,
            )
        self.assertEqual(
            float(result.best_actions[0]),
            float(result.best_actions[ACTION_CONTROL_POINT_OFFSETS[0]]),
        )

    def test_cem_settings_record_only_the_action_parameterization_change(self) -> None:
        settings = mpc_tool.cem_optimizer_settings(64, 4, 0.1)

        self.assertEqual(settings["candidates"], 64)
        self.assertEqual(settings["iterations"], 4)
        self.assertEqual(settings["elite_fraction"], 0.1)
        self.assertEqual(settings["initial_action_std"], 1.0)
        self.assertEqual(settings["minimum_action_std"], MIN_CEM_STD)
        self.assertEqual(
            settings["action_parameterization"]["control_point_offsets"],
            list(ACTION_CONTROL_POINT_OFFSETS),
        )
        self.assertEqual(
            settings["action_parameterization"]["decision_steps"],
            ACTION_SEQUENCE_LENGTH,
        )
        self.assertEqual(
            settings["action_parameterization"]["expanded_horizon_steps"],
            HORIZON_STEPS,
        )

    def test_action_conversion_is_cumulative_and_bounded(self) -> None:
        raw = np.asarray(
            [
                [2.0, 1.0, -2.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        actual = actions_to_injections(raw, 240_000.0)
        expected = np.asarray(
            [
                [290_000.0, 340_000.0, 290_000.0, 240_000.0, 190_000.0],
                [190_000.0, 140_000.0, 190_000.0, 240_000.0, 290_000.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual >= 0.0))
        self.assertTrue(np.all(actual <= 500_000.0))
        self.assertTrue(np.all(np.abs(np.diff(actual, axis=1)) <= mpc_tool.MAX_DELTA))

    def test_action_conversion_clips_at_500k_and_keeps_fine_adjustments(self) -> None:
        actual = actions_to_injections(
            np.asarray([1.0, 1.0, -1.0, -1.0, 1.0 / 15.0], dtype=np.float32),
            450_000.0,
        )
        np.testing.assert_allclose(
            actual,
            np.asarray(
                [500_000.0, 500_000.0, 450_000.0, 400_000.0, 403_333.34375],
                dtype=np.float32,
            ),
        )

    def test_candidate_forecast_is_batched_with_distinct_injections(self) -> None:
        runtime = object.__new__(ForecastBundleRuntime)
        runtime.device = torch.device("cpu")
        runtime.node_ids = ("1", "2")
        runtime.model = _RecordingForecast(node_count=2)
        runtime.future_mean = np.asarray([10.0, 0.25, -0.25], dtype=np.float32)
        runtime.future_std = np.asarray([2.0, 0.5, 0.25], dtype=np.float32)
        runtime.target_mean = np.zeros(2, dtype=np.float32)
        runtime.target_std = np.ones(2, dtype=np.float32)
        runtime.head = {
            "name": "chlorine",
            "prediction": "level",
            "transform": "identity",
            "scale": None,
        }
        runtime.manifest = {
            "future_known_feature_contract": {
                "transforms": [
                    {
                        "feature": "chlorine-injection:1",
                        "transform": "identity",
                        "scale": None,
                    },
                    {"feature": "day_sin", "transform": "identity", "scale": None},
                    {"feature": "day_cos", "transform": "identity", "scale": None},
                ]
            }
        }
        times = np.stack(
            (
                np.linspace(-1.0, 1.0, HORIZON_STEPS),
                np.linspace(1.0, -1.0, HORIZON_STEPS),
            ),
            axis=1,
        ).astype(np.float32)
        prepared = PreparedForecastQuery(
            raw_history=np.zeros((48, 4), dtype=np.float32),
            normalized_history=np.zeros((48, 4), dtype=np.float32),
            base_future_known=np.concatenate(
                (np.zeros((HORIZON_STEPS, 1), dtype=np.float32), times), axis=1
            ),
            future_injection_index=0,
        )
        injections = np.stack(
            (
                np.zeros(HORIZON_STEPS),
                np.arange(HORIZON_STEPS),
                np.arange(HORIZON_STEPS) + 100,
            )
        ).astype(np.float32)
        predicted = runtime.predict_candidates(
            prepared, injections, np.zeros(2, dtype=np.float32)
        )

        self.assertEqual(predicted.shape, (3, HORIZON_STEPS, 2))
        self.assertEqual(runtime.model.history_shape, (3, 48, 4))
        assert runtime.model.future is not None
        np.testing.assert_allclose(
            runtime.model.future[..., 0], (injections - 10.0) / 2.0
        )
        standardized_times = (
            times - runtime.future_mean[1:]
        ) / runtime.future_std[1:]
        np.testing.assert_allclose(runtime.model.future[0, :, 1:], standardized_times)
        np.testing.assert_allclose(runtime.model.future[1, :, 1:], standardized_times)

    def test_prepared_query_uses_frozen_forecast_history_statistics(self) -> None:
        runtime = object.__new__(ForecastBundleRuntime)
        runtime.history_mean = np.asarray([2.0, 10.0], dtype=np.float32)
        runtime.history_std = np.asarray([0.5, 4.0], dtype=np.float32)
        rows = 70
        transformed_history = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        raw_history = transformed_history + 1_000.0
        future_known = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        prepared = PreparedForecastTrajectory(
            raw_history=raw_history,
            transformed_history=transformed_history,
            transformed_future_known=future_known,
            future_injection_index=0,
        )

        query = runtime.query_from_prepared(prepared, query_index=47)

        np.testing.assert_array_equal(query.raw_history, raw_history[:48])
        np.testing.assert_allclose(
            query.normalized_history,
            (transformed_history[:48] - runtime.history_mean) / runtime.history_std,
        )
        np.testing.assert_array_equal(query.base_future_known, future_known[48:64])

    def test_forecast_experiment_uses_raw_features_before_standardization(self) -> None:
        job = mpc_tool.build_experiment().jobs[0]

        features = (*job.observation.history, *job.observation.future_known)
        self.assertTrue(all(feature.transform == "identity" for feature in features))

    def test_writer_emits_compressed_pickle_free_schema(self) -> None:
        count, iterations, history_width, node_count = 2, 3, 5, 2
        arrays: dict[str, np.ndarray] = {
            "scenario_id": np.asarray(["s0", "s1"], dtype=np.str_),
            "trajectory_identity": np.asarray(["a", "b"], dtype=np.str_),
            "chlorine_profile_hash": np.asarray(["c", "d"], dtype=np.str_),
            "arsenic_schedule_hash": np.asarray(["e", "f"], dtype=np.str_),
            "calendar_start_day": np.zeros(count, dtype=np.int16),
            "query_index": np.arange(count, dtype=np.int64),
            "time_seconds": np.arange(count, dtype=np.int64) * 1_800,
            "raw_causal_history": np.zeros(
                (count, 48, history_width), dtype=np.float32
            ),
            "policy_observation_history": np.zeros(
                (count, 48, POLICY_OBSERVATION_SIZE), dtype=np.float32
            ),
            "current_injection": np.zeros(count, dtype=np.float32),
            "best_raw_actions": np.zeros((count, 16), dtype=np.float32),
            "best_injections": np.zeros((count, 16), dtype=np.float32),
            "first_action": np.zeros(count, dtype=np.float32),
            "predicted_chlorine": np.zeros(
                (count, 16, node_count), dtype=np.float32
            ),
            "predicted_return": np.zeros(count, dtype=np.float32),
            "reward_state_zlib_base64": np.asarray(
                [b"eJyrBgEEAQI=", b"eJyrBgEEAQI="], dtype=np.bytes_
            ),
            "cem_seed": np.arange(count, dtype=np.uint32),
            "cem_iteration_population_max": np.zeros(
                (count, iterations), dtype=np.float32
            ),
            "cem_iteration_population_mean": np.zeros(
                (count, iterations), dtype=np.float32
            ),
            "cem_iteration_elite_mean": np.zeros(
                (count, iterations), dtype=np.float32
            ),
            "cem_iteration_best_so_far": np.zeros(
                (count, iterations), dtype=np.float32
            ),
            "cem_iteration_action_mean": np.zeros(
                (count, iterations, 16), dtype=np.float32
            ),
            "cem_iteration_action_std": np.ones(
                (count, iterations, 16), dtype=np.float32
            ),
        }
        self.assertEqual(set(arrays), set(REQUIRED_DATASET_ARRAYS))
        manifest = {
            "schema_version": 2,
            "kind": "mpc-teacher-cem",
            "source_split": "train",
            "limitations": {
                "planning_horizon_hours": 8.0,
                "terminal_value_approximation": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset_path, manifest_path = write_dataset(
                Path(directory), arrays, manifest
            )
            with np.load(dataset_path, allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), set(REQUIRED_DATASET_ARRAYS))
                self.assertEqual(stored["predicted_chlorine"].shape, (2, 16, 2))
                self.assertEqual(
                    stored["policy_observation_history"].shape,
                    (2, 48, POLICY_OBSERVATION_SIZE),
                )
                self.assertEqual(stored["scenario_id"].tolist(), ["s0", "s1"])
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_manifest["schema_version"], 2)
            self.assertEqual(stored_manifest["source_split"], "train")
            self.assertFalse(
                stored_manifest["limitations"]["terminal_value_approximation"]
            )
            self.assertEqual(
                stored_manifest["arrays"]["best_raw_actions"]["shape"], [2, 16]
            )
            with self.assertRaises(FileExistsError):
                write_dataset(Path(directory), arrays, manifest)

        arrays["predicted_chlorine"] = np.stack(
            [arrays["predicted_chlorine"] + index for index in range(5)], axis=1
        )
        manifest["kind"] = "mpc-teacher-cem-ensemble"
        manifest["source_split"] = "validation"
        with tempfile.TemporaryDirectory() as directory:
            dataset_path, _ = write_dataset(Path(directory), arrays, manifest)
            with np.load(dataset_path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["predicted_chlorine"], arrays["predicted_chlorine"])
                self.assertEqual(stored["predicted_chlorine"].shape, (2, 5, 16, 2))

    def test_manifest_describes_policy_replay_and_action_alignment(self) -> None:
        runtime = object.__new__(ForecastBundleRuntime)
        runtime.raw_history_names = ("chlorine:1", "chlorine-injection:1")
        runtime.node_ids = tuple(str(index + 1) for index in range(31))
        runtime.source_ids = ("1",)
        runtime.timestep_seconds = 1_800
        runtime.bundle = Path("bundle")
        runtime.manifest = {
            "job_hash": "job",
            "history_feature_contract": {"names": list(runtime.raw_history_names)},
            "future_known_feature_contract": {
                "names": ["chlorine-injection:1"]
            },
        }
        manifest = build_output_manifest(
            runtime,
            forecasting_run=Path("run"),
            trajectory_cache=Path("cache"),
            state_count=2,
            all_states=False,
            trajectory_count=1,
            queries_per_trajectory=24,
            candidates=8,
            iterations=2,
            elite_fraction=0.25,
            seed=102,
            device=torch.device("cpu"),
            workers=1,
            discount_factor=0.99,
            query_half_hour_slot_counts=(1, 1) + (0,) * 46,
        )

        contract = manifest["policy_observation_contract"]
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(contract["builder"]["shape"], [43])
        self.assertEqual(contract["history_steps"], 48)
        self.assertEqual(contract["observation_size"], 43)
        self.assertFalse(contract["normalized"])
        self.assertEqual(contract["stored_representation"], "v89-builder-output")
        self.assertEqual(
            manifest["raw_history_feature_contract"]["forecast_model_preprocessing"],
            {
                "method": "forecast-training-split-mean-std",
                "stored": "raw-physical",
                "applied_at": "forecast-model-input-boundary",
                "statistics": "forecasting_bundle/normalization.json",
            },
        )
        self.assertNotIn("normalization", contract)
        self.assertIn("query-state observation", contract["alignment"])
        self.assertTrue(
            manifest["execution"][
                "policy_observations_replayed_once_per_trajectory"
            ]
        )
        distribution = manifest["selection"]["daily_half_hour_distribution"]
        self.assertEqual(distribution["slot_count"], 48)
        self.assertEqual(distribution["query_count_by_slot"][:2], [1, 1])
        sampling = manifest["selection"]["per_trajectory_sampling"]
        self.assertEqual(sampling["temporal_strata"], 24)
        self.assertFalse(sampling["fixed_spacing"])
        self.assertEqual(
            manifest["source_dataset"]["definition"],
            "training.datasets.hanoi.forecasting_full.train",
        )
        self.assertEqual(
            manifest["source_dataset"]["contract"]["arsenic"]["seed"],
            6_103,
        )
        self.assertEqual(
            manifest["action_contract"],
            {
                "raw_minimum": -1.0,
                "raw_maximum": 1.0,
                "max_delta": mpc_tool.MAX_DELTA,
                "minimum_injection": 0.0,
                "maximum_injection": 500_000.0,
                "initial_injection": 0.0,
                "cumulative": True,
            },
        )
        self.assertEqual(
            manifest["reward"]["parameters"],
            mpc_tool.V89_PARAMETERS,
        )
        self.assertEqual(
            manifest["teacher_configuration_hash"],
            mpc_tool.config_hash(manifest["teacher_configuration"]),
        )
        self.assertEqual(
            manifest["teacher_configuration"]["optimizer"]["settings"],
            {
                key: value
                for key, value in manifest["cem"].items()
                if key != "elite_count"
            },
        )
        self.assertEqual(
            manifest["cem"]["action_parameterization"],
            {
                "kind": "fixed-piecewise-linear",
                "parameter_space": "raw-action-delta",
                "control_point_offsets": [0, 1, 2, 3, 5, 8, 11, 15],
                "decision_steps": 8,
                "expanded_horizon_steps": 16,
                "interpolation": "linear",
                "stored_plan_representation": "expanded-per-step",
                "diagnostic_representation": "expanded-marginal-mean-and-std",
            },
        )

    def test_query_selection_resolves_only_the_supplied_split(self) -> None:
        bundle_manifest = {
            "job": {
                "dataset": mpc_tool.canonical_data(
                    mpc_tool.build_experiment().jobs[0].dataset
                )
            }
        }
        with patch.object(
            mpc_tool,
            "resolve_dataset_split",
            side_effect=RuntimeError("stop-after-resolve"),
        ) as resolver:
            with self.assertRaisesRegex(RuntimeError, "stop-after-resolve"):
                select_training_queries(
                    cache_root=Path("cache"),
                    state_count=None,
                    queries_per_trajectory=24,
                    seed=102,
                    bundle_manifest=bundle_manifest,
                )
        resolver.assert_called_once_with(
            forecasting_full.network,
            forecasting_full.train,
            scenarios_dir=forecasting_full.scenarios_dir,
            uncertainty_seed=forecasting_full.uncertainty_seed,
            split="train",
        )

    def test_query_selection_rejects_held_out_scenarios_seen_by_forecaster(self) -> None:
        job = build_cross_fit_experiment().jobs[0]
        overlapping = forecasting_full.train
        bundle_manifest = {"job": {"dataset": mpc_tool.canonical_data(job.dataset)}}
        with self.assertRaisesRegex(ValueError, "occur in forecast training"):
            select_training_queries(
                cache_root=Path("cache"),
                state_count=None,
                queries_per_trajectory=24,
                seed=102,
                bundle_manifest=bundle_manifest,
                forecast_job=job,
                source_split=overlapping,
                require_out_of_fold=True,
            )

    def test_query_selection_accepts_matching_held_out_schedule_contract(self) -> None:
        job = build_cross_fit_experiment().jobs[0]
        job = replace(job, dataset=replace(job.dataset, scenarios_dir="custom-fold", uncertainty_seed=777))
        bundle_manifest = {"job": {"dataset": mpc_tool.canonical_data(job.dataset)}}
        with patch.object(
            mpc_tool,
            "resolve_dataset_split",
            side_effect=RuntimeError("stop-after-resolve"),
        ) as resolver:
            with self.assertRaisesRegex(RuntimeError, "stop-after-resolve"):
                select_training_queries(
                    cache_root=Path("cache"),
                    state_count=None,
                    queries_per_trajectory=24,
                    seed=102,
                    bundle_manifest=bundle_manifest,
                    forecast_job=job,
                    source_split=FOLDS[0].held_out,
                    require_out_of_fold=True,
                )
        resolved_spec = resolver.call_args.args[1]
        self.assertEqual(resolver.call_args.kwargs["scenarios_dir"], "custom-fold")
        self.assertEqual(resolver.call_args.kwargs["uncertainty_seed"], 777)
        self.assertIs(resolved_spec.arsenic, forecasting_full.train.arsenic)
        self.assertIs(resolved_spec.chlorine, forecasting_full.train.chlorine)

    def test_no_argument_cli_selects_full_stratified_pool(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.state_count)
        self.assertEqual(args.queries_per_trajectory, 24)
        self.assertFalse(hasattr(args, "state_stride"))
        self.assertEqual(args.workers, 1)
        self.assertIsNone(args.output_dir)
        self.assertEqual(DEFAULT_OUTPUT.name, mpc_tool.DEFAULT_TEACHER_CONFIGURATION_HASH)
        self.assertAlmostEqual(MIN_CEM_STD, 1.0 / 60.0)

    def test_query_grouping_is_deterministic_per_trajectory(self) -> None:
        first = TrajectoryReference("a", Path("a"), "s0", "arsenic", "chlorine")
        second = TrajectoryReference("b", Path("b"), "s1", "arsenic", "chlorine")
        selections = (
            mpc_tool.QuerySelection(first, 95),
            mpc_tool.QuerySelection(first, 47),
            mpc_tool.QuerySelection(second, 143),
        )
        groups = group_query_selections(selections)
        self.assertEqual([item.reference.identity for item in groups], ["a", "b"])
        self.assertEqual(groups[0].query_indexes, (47, 95))
        self.assertEqual(groups[1].query_indexes, (143,))

    @staticmethod
    def _query_availability(trajectory_count: int = 96):
        return [
            (
                TrajectoryReference(
                    f"identity-{index}",
                    Path(f"cache-{index}"),
                    f"scenario-{index}",
                    "arsenic",
                    "chlorine",
                ),
                1_200,
            )
            for index in range(trajectory_count)
        ]

    def test_temporal_strata_are_random_deterministic_and_clock_balanced(self) -> None:
        available = self._query_availability()
        first = temporally_stratified_query_pool(available, 24, seed=102)
        second = temporally_stratified_query_pool(available, 24, seed=102)
        changed = temporally_stratified_query_pool(available, 24, seed=103)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        grouped: dict[str, list[int]] = {}
        for selection in first:
            grouped.setdefault(selection.reference.identity, []).append(
                selection.query_index
            )
        self.assertEqual(len(grouped), 96)
        valid_strata = np.array_split(np.arange(47, 1_184), 24)
        for indexes in grouped.values():
            self.assertEqual(len(indexes), 24)
            self.assertEqual(len(set(indexes)), 24)
            for index, stratum in zip(indexes, valid_strata, strict=True):
                self.assertIn(index, stratum)
            self.assertFalse(np.all(np.diff(indexes) == 48))

        slots = np.bincount(
            [half_hour_slot_for_query_index(item.query_index) for item in first],
            minlength=48,
        )
        self.assertEqual(np.count_nonzero(slots), 48)
        self.assertLessEqual(int(slots.max() - slots.min()), 1)

    def test_state_subsampling_is_balanced_across_half_hour_slots(self) -> None:
        eligible = temporally_stratified_query_pool(
            self._query_availability(), 24, seed=102
        )

        selected = stratified_query_sample(eligible, 96, seed=102)
        repeated = stratified_query_sample(eligible, 96, seed=102)
        slots = np.bincount(
            [half_hour_slot_for_query_index(item.query_index) for item in selected],
            minlength=48,
        )
        self.assertEqual(selected, repeated)
        np.testing.assert_array_equal(slots, np.full(48, 2))
        self.assertEqual(len(set(selected)), 96)

        small = stratified_query_sample(eligible, 8, seed=102)
        self.assertEqual(
            len(
                {
                    half_hour_slot_for_query_index(item.query_index)
                    for item in small
                }
            ),
            8,
        )

    def test_incremental_reward_replay_matches_independent_reconstruction(self) -> None:
        trajectory = self._trajectory()
        actual = dict(incremental_reward_states(trajectory, (47, 95)))
        self.assertEqual(actual[47], reconstruct_reward_state(trajectory, 47))
        self.assertEqual(actual[95], reconstruct_reward_state(trajectory, 95))

    def test_teacher_contract_matches_v89_training_and_rejects_other_rewards(self) -> None:
        configuration = mpc_tool.DEFAULT_TEACHER_CONFIGURATION
        for job in build_v89_experiment().jobs:
            self.assertEqual(configuration["reward"]["id"], job.reward.factory.reward_id)
            self.assertEqual(configuration["reward"]["parameters"], job.reward.as_dict())
            self.assertEqual(mpc_tool.V89_OBSERVATION, job.observation)
            self.assertEqual(configuration["reward"]["discount_factor"], job.algorithm.gamma)
        self.assertIsInstance(mpc_tool.reward_from_manifest(configuration), V89Reward)
        for contract in (
            {**configuration["reward"], "id": "other"},
            {**configuration["reward"], "parameters": {}},
        ):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                mpc_tool.reward_from_manifest({"reward": contract})

    def test_candidate_returns_match_continuous_v89_replay(self) -> None:
        trajectory = self._trajectory()
        query_index = 43
        predictions = np.stack([
            trajectory.chlorine[44:60],
            np.full((HORIZON_STEPS, 2), 0.35, dtype=np.float32),
        ])
        actions = np.stack([
            np.linspace(-0.5, 0.5, HORIZON_STEPS, dtype=np.float32),
            np.zeros(HORIZON_STEPS, dtype=np.float32),
        ])
        injections = actions_to_injections(actions, float(trajectory.injection[query_index]))
        state = reconstruct_reward_state(trajectory, query_index)
        self.assertEqual(
            state,
            dict(incremental_reward_states(trajectory, (query_index,)))[query_index],
        )
        actual = mpc_tool.score_predicted_candidates(
            predictions, actions, injections, reward_state=state,
            trajectory=trajectory, query_index=query_index,
            timestep_seconds=1_800, discount_factor=0.99,
        )
        expected = []
        for candidate in range(2):
            reward = V89Reward(V89_PARAMETERS)
            reward.load_state_dict(state)
            total = 0.0
            for step in range(HORIZON_STEPS):
                sample = replace(
                    mpc_tool.reward_input_from_trajectory(trajectory, query_index + step + 1),
                    nodes_quality=predictions[candidate, step],
                    raw_action=float(actions[candidate, step]),
                    current_injection=float(injections[candidate, step]),
                )
                value, _ = reward.compute(sample)
                total += 0.99 ** step * value
            expected.append(total)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertEqual(
            state, reconstruct_reward_state(trajectory, query_index)
        )

    def test_generator_replays_v89_observations_and_reward_state_before_labeling(self) -> None:
        trajectory = self._trajectory(node_count=31)
        reference = SimpleNamespace(load=lambda: trajectory)
        group = mpc_tool.TrajectoryQueryGroup(reference, (43, 47))
        runtime = SimpleNamespace(prepare_trajectory=lambda _: None)
        with patch.object(
            mpc_tool, "_validate_selected_trajectory"
        ), patch.object(mpc_tool, "_generate_prepared_record") as label:
            mpc_tool.generate_trajectory_records(
                runtime, group, mpc_tool.GenerationConfig(64, 4, 0.1, 0.99, 17)
            )
            self.assertEqual(label.call_count, 2)
            for call in label.call_args_list:
                query_index = call.args[1].query_index
                observations, state = call.args[4:6]
                self.assertEqual(observations.shape, (48, 43))
                self.assertIn("attack_alpha", state)
                self.assertIn("release_step", state)
                self.assertEqual(state["episode_step"], query_index)
                self.assertEqual(
                    observations[query_index, 31], trajectory.injection[query_index]
                )

    def test_policy_observation_replay_matches_stateful_builder(self) -> None:
        trajectory = self._trajectory(node_count=31)
        actual = replay_policy_observations(
            trajectory, through_index=95, observation_spec=mpc_tool.V89_OBSERVATION
        )

        builder = ControlObservationBuilder(
            mpc_tool.V89_OBSERVATION,
            trajectory.node_ids,
            MAX_INJECTION,
        )
        builder.reset()
        expected = np.stack(
            [
                builder.build(simulation_sample_from_trajectory(trajectory, index))
                for index in range(96)
            ]
        )

        self.assertEqual(actual.shape, (96, POLICY_OBSERVATION_SIZE))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape[1], 43)
        self.assertEqual(actual[0, 40], 0.0)
        self.assertEqual(actual[0, 42], 0.0)
        self.assertEqual(actual[95, 42], 1.0)
        self.assertGreater(actual[40, 40], 0.0)

    def test_policy_history_ends_at_query_before_first_action(self) -> None:
        trajectory = self._trajectory(node_count=31)
        query_index = 95
        replayed = replay_policy_observations(
            trajectory, through_index=query_index,
            observation_spec=mpc_tool.V89_OBSERVATION,
        )
        history = policy_observation_history_at(replayed, query_index)

        self.assertEqual(history.shape, (48, 43))
        np.testing.assert_array_equal(history[0], replayed[48])
        np.testing.assert_array_equal(history[-1], replayed[query_index])
        self.assertAlmostEqual(
            float(history[-1, 31]),
            float(trajectory.injection[query_index]),
            places=6,
        )
        first_action = 0.75
        first_injection = actions_to_injections(
            np.asarray([[first_action]], dtype=np.float32),
            float(trajectory.injection[query_index]),
        )[0, 0]
        self.assertNotAlmostEqual(
            float(history[-1, 33]), float(first_injection / MAX_INJECTION)
        )

    def test_parallel_workers_use_cpu_and_reject_explicit_accelerator(self) -> None:
        self.assertEqual(resolve_generation_device("auto", 4), torch.device("cpu"))
        self.assertEqual(resolve_generation_device("cpu", 4), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "requires --device cpu"):
            resolve_generation_device("mps", 2)

    def test_keyboard_interrupt_returns_130_without_publishing(self) -> None:
        with patch.object(mpc_tool, "generate_dataset", side_effect=KeyboardInterrupt):
            self.assertEqual(mpc_tool.main([]), 130)

    def test_incomplete_best_bundle_does_not_fall_back_to_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint-only")
            with self.assertRaisesRegex(
                FileNotFoundError,
                "completed forecasting best/ bundle is required.*cannot be reconstructed",
            ):
                require_completed_best_bundle(run / "best")


if __name__ == "__main__":
    unittest.main()
