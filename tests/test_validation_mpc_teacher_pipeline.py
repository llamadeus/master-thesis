from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from packages.trainer.serialization import canonical_data, config_hash
from tools.mpc import generate_mpc_teacher_dataset_cem as cem
from tools.mpc import generate_cross_fit_mpc_teacher_dataset_cem as cross_fit
from tools.mpc import generate_validation_mpc_teacher_dataset_cem as validation
from training import run_principal_v89_pipeline as principal
from training.datasets.hanoi import forecasting_full
from tests import test_generate_mpc_teacher_dataset_cem as cem_tests


class ValidationTeacherTests(unittest.TestCase):
    def test_every_member_scores_every_candidate_before_return_mean(self):
        trajectory = cem_tests.CEMTeacherDatasetTests._trajectory()
        query = 47
        actions = np.array([[0.8] * 16, [-0.2] * 16], dtype=np.float32)
        injections = cem.actions_to_injections(actions, trajectory.injection[query])
        predictions = np.empty((2, 5, 16, 2), dtype=np.float32)
        predictions[0] = np.array([0.05, 0.1, 0.3, 0.5, 0.55])[:, None, None]
        predictions[1] = 0.28
        context = dict(reward_state=cem.reconstruct_reward_state(trajectory, query),
                       trajectory=trajectory, query_index=query,
                       timestep_seconds=1800, discount_factor=0.99)
        original_state = copy.deepcopy(context["reward_state"])
        expected = np.mean([cem.score_predicted_candidates(predictions[:, k], actions,
                            injections, **context) for k in range(5)], axis=0)
        with patch.object(cem, "score_predicted_candidates", wraps=cem.score_predicted_candidates) as score:
            actual = cem.score_forecast_predictions(predictions, actions, injections, **context)
        self.assertEqual(score.call_count, 5)
        for k, call in enumerate(score.call_args_list):
            np.testing.assert_array_equal(call.args[0], predictions[:, k])
            np.testing.assert_array_equal(call.args[1], actions)
        np.testing.assert_allclose(actual, expected)
        averaged_trajectory_score = cem.score_predicted_candidates(predictions.mean(axis=1), actions,
                                                                  injections, **context)
        self.assertFalse(np.allclose(actual, averaged_trajectory_score))
        self.assertEqual(context["reward_state"], original_state)

    def test_record_labels_first_action_of_plan_with_best_mean_return(self):
        trajectory = cem_tests.CEMTeacherDatasetTests._trajectory(node_count=31)
        query = 47
        actions = np.array([[0.8] * 16, [-0.2] * 16], dtype=np.float32)
        predictions = np.empty((2, 5, 16, 31), dtype=np.float32)
        predictions[0] = np.array([0.05, 0.1, 0.3, 0.5, 0.55])[:, None, None]
        predictions[1] = 0.28
        runtime = SimpleNamespace(
            timestep_seconds=1800,
            query_from_prepared=lambda *_: SimpleNamespace(raw_history=np.zeros((48, 38))),
            predict_candidates=lambda *_: predictions,
        )
        selection = cem.QuerySelection(SimpleNamespace(identity="validation-trajectory"), query)
        observations = cem.replay_policy_observations(trajectory, observation_spec=cem.V89_OBSERVATION)
        state = cem.reconstruct_reward_state(trajectory, query)
        evaluations = []

        def optimize(evaluate, **kwargs):
            result = evaluate(actions)
            evaluations.append(result)
            best = int(np.argmax(result.returns))
            return cem.CEMResult(actions[best], result.returns[best],
                                 result.predicted_chlorine[best], None)

        with patch.object(cem, "optimize_cem_action_sequence", side_effect=optimize):
            record = cem._generate_prepared_record(runtime, selection, trajectory, None,
                observations, state, cem.GenerationConfig(64, 4, 0.1, 0.99, 17))
        best = int(np.argmax(evaluations[0].returns))
        self.assertEqual(record.first_action, float(actions[best, 0]))
        np.testing.assert_array_equal(record.best_raw_actions, actions[best])
        np.testing.assert_array_equal(record.predicted_chlorine, predictions[best])
        self.assertEqual(record.predicted_chlorine.shape, (5, 16, 31))

    def test_runtime_keeps_member_preprocessing_and_predictions_separate(self):
        members = []
        for k in range(5):
            member = SimpleNamespace(bundle=Path(str(k)), manifest={}, node_ids=("n",),
                source_ids=("s",), timestep_seconds=1800, raw_history_names=("c",),
                prepare_trajectory=lambda trajectory, k=k: (trajectory, k),
                query_from_prepared=lambda prepared, query: SimpleNamespace(raw_history=prepared, query=query),
                predict_candidates=lambda prepared, injections, current, k=k: np.full((2, 16, 1), k))
            members.append(member)
        bundles = tuple(Path(str(k)) for k in range(5))
        with patch.object(cem, "ForecastBundleRuntime", side_effect=members):
            runtime = cem.EnsembleForecastRuntime(bundles, torch.device("cpu"))
        prepared = runtime.prepare_trajectory("trajectory")
        self.assertEqual(prepared, tuple(("trajectory", k) for k in range(5)))
        result = runtime.predict_candidates(runtime.query_from_prepared(prepared, 47), None, None)
        for k in range(5):
            np.testing.assert_array_equal(result[:, k], k)

    def test_seed_identity_and_generation_use_only_five_matching_folds_and_validation(self):
        seed = 17
        experiment = principal.build_forecasting_experiment(seed)
        identity = validation.validation_teacher_configuration(experiment, seed=seed)
        runs = tuple(SimpleNamespace(job=job, bundle=Path(f"bundle-{k}"), run_dir=Path(f"run-{k}"))
                     for k, job in enumerate(experiment.jobs))
        args = argparse.Namespace(seed=seed, workers=3, device="cpu", output_dir=Path("output"),
                                  trajectory_cache=Path("cache"))
        with patch.object(cross_fit, "preflight_forecast", side_effect=runs) as preflight, \
             patch.object(cross_fit, "_forecast_bundle_hashes", return_value={"model.pt": "hash"}), \
             patch.object(cem, "generate_dataset", return_value=(Path("dataset"), Path("manifest"))) as generate:
            validation.execute_pipeline(args, experiment=experiment)
        self.assertEqual(preflight.call_count, 5)
        call = generate.call_args
        self.assertEqual(call.kwargs["ensemble_bundles"], tuple(run.bundle for run in runs))
        self.assertEqual(call.kwargs["source_split"], forecasting_full.validation)
        self.assertEqual(call.kwargs["source_phase"], "evaluation")
        self.assertEqual(call.kwargs["manifest_updates"]["teacher_configuration"], identity)
        for run in runs:
            self.assertFalse(set(run.job.dataset.train.scenarios) & set(forecasting_full.validation.scenarios))
        other = validation.validation_teacher_configuration(
            principal.build_forecasting_experiment(42), seed=42
        )
        self.assertNotEqual(config_hash(identity), config_hash(other))
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            validation.validation_teacher_configuration(experiment, seed=42)

    def test_query_selection_uses_validation_schedules_and_read_only_cache(self):
        job = principal.build_forecasting_experiment(17).jobs[0]
        manifest = {"job": {"dataset": canonical_data(job.dataset)}}
        with patch.object(cem, "resolve_dataset_split") as resolve, \
             patch.object(cem, "ForecastTrajectoryResolver") as resolver:
            resolver.return_value._prepared.side_effect = RuntimeError("stop-before-labels")
            with self.assertRaisesRegex(RuntimeError, "stop-before-labels"):
                cem.select_training_queries(cache_root=Path("validation-cache"), state_count=None,
                    queries_per_trajectory=24, seed=17, bundle_manifest=manifest, forecast_job=job,
                    source_split=forecasting_full.validation, source_phase="evaluation")
        resolve.assert_called_once_with(
            "hanoi", forecasting_full.validation,
            scenarios_dir="hanoi", uncertainty_seed=42, split="evaluation",
        )
        self.assertFalse(resolver.call_args.args[1].materialize_missing)
        self.assertEqual(resolver.call_args.args[1].cache_root, Path("validation-cache"))
        resolver.return_value._prepared.assert_called_once_with("evaluation")

    def test_resume_rejects_changed_ensemble_checkpoint(self):
        identity = validation.validation_teacher_configuration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for k, job_hash in enumerate(identity["forecast"]["job_hashes"]):
                bundle = root / str(k)
                bundle.mkdir()
                for name in ("model.pt", "manifest.json", "normalization.json"):
                    (bundle / name).write_text("original")
                members.append(dict(bundle=str(bundle), job_hash=job_hash,
                                    bundle_hashes=cross_fit._forecast_bundle_hashes(bundle)))
            manifest = dict(kind="mpc-teacher-cem-ensemble", teacher_configuration=identity,
                teacher_configuration_hash=config_hash(identity), ensemble={"members": members},
                arrays={"first_action": {"shape": [1]}})
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "dataset.npz").write_bytes(b"present")
            self.assertTrue(principal._teacher_is_complete(root, identity))
            (root / "3/model.pt").write_text("changed")
            self.assertFalse(principal._teacher_is_complete(root, identity))

    def test_small_synthetic_generation_writes_loadable_ensemble_dataset(self):
        from tools.control.bc.train_mpc_teacher_recurrent_ppo import _load_dataset

        seed = 17
        experiment = principal.build_forecasting_experiment(seed)
        job = experiment.jobs[0]
        identity = validation.validation_teacher_configuration(experiment, seed=seed)
        trajectory = replace(cem_tests.CEMTeacherDatasetTests._trajectory(node_count=31),
                             scenario_id="scenario-80")
        reference = SimpleNamespace(identity="validation-trajectory", load=lambda: trajectory,
            scenario_id=trajectory.scenario_id, chlorine_schedule_hash=trajectory.chlorine_profile_hash,
            arsenic_schedule_hash=trajectory.arsenic_schedule_hash)
        runtime = SimpleNamespace(
            bundle=Path("bundle"), node_ids=trajectory.node_ids, source_ids=trajectory.source_ids,
            raw_history_names=("chlorine:1",), timestep_seconds=1800,
            manifest={"job_hash": config_hash(job), "job": {"dataset": canonical_data(job.dataset)},
                      "history_feature_contract": {"names": ["chlorine:1"]},
                      "future_known_feature_contract": {"names": ["injection"]}},
            prepare_trajectory=lambda trajectory: None,
            query_from_prepared=lambda *args: SimpleNamespace(raw_history=np.zeros((48, 1))),
            predict_candidates=lambda prepared, injections, current: np.broadcast_to(
                np.array([0.2, 0.25, 0.3, 0.35, 0.4])[None, :, None, None],
                (len(injections), 5, 16, 31)).astype(np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(forecasting_run=root / "run", trajectory_cache=root / "cache",
                output_dir=root / "teacher", state_count=1, queries_per_trajectory=24,
                candidates=2, iterations=1, elite_fraction=0.5, seed=seed, device="cpu",
                workers=1, discount_factor=0.99)
            with patch.object(cem, "EnsembleForecastRuntime", return_value=runtime), \
                 patch.object(cem, "select_training_queries", return_value=(cem.QuerySelection(reference, 47),)):
                dataset_path, manifest_path = cem.generate_dataset(
                    args, forecast_job=job, source_split=forecasting_full.validation,
                    source_definition="validation", require_out_of_fold=False,
                    source_phase="evaluation", ensemble_bundles=tuple(Path(str(k)) for k in range(5)),
                    manifest_updates={"kind": "mpc-teacher-cem-ensemble", "source_split": "validation",
                        "teacher_configuration": identity, "teacher_configuration_hash": config_hash(identity)},
                )
            loaded = _load_dataset(dataset_path.parent)
            self.assertEqual(loaded.scenarios.tolist(), ["scenario-80"])
            self.assertEqual(loaded.query_indices.tolist(), [47])
            self.assertEqual(loaded.histories.shape[-1], 43)
            with np.load(dataset_path, allow_pickle=False) as stored:
                self.assertEqual(stored["predicted_chlorine"].shape, (1, 5, 16, 31))
                self.assertEqual(stored["first_action"][0], stored["best_raw_actions"][0, 0])
            self.assertEqual(json.loads(manifest_path.read_text())["source_split"], "validation")

    def test_five_fold_preflight_returns_exactly_the_fold_models(self):
        experiment = principal.build_forecasting_experiment(17)
        with patch.object(cross_fit, "preflight_forecast", side_effect=list(range(5))):
            folds = cross_fit.preflight_all_forecasts(experiment)
        self.assertEqual(folds, tuple(range(5)))


if __name__ == "__main__":
    unittest.main()
