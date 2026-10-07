from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from packages.trainer.serialization import canonical_data, config_hash
from tools.mpc import generate_cross_fit_mpc_teacher_dataset_cem as pipeline
from tools.mpc import generate_mpc_teacher_dataset_cem as cem
from training.datasets.hanoi import forecasting_full
from training.experiments.forecasting_encoder_decoder_lstm_cross_fit import (
    FOLDS,
    build_experiment,
)


class CrossFitMpcTeacherPipelineTests(unittest.TestCase):
    def test_aggregate_configuration_uses_only_labeling_folds(self) -> None:
        configuration = pipeline.cross_fit_teacher_configuration()

        self.assertEqual(
            configuration["source_dataset"]["split"],
            canonical_data(forecasting_full.train),
        )
        self.assertEqual(len(configuration["forecast"]["folds"]), 5)

    def test_preflight_requires_completed_matching_forecast_artifacts(self) -> None:
        job = build_experiment().jobs[0]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            bundle = run_dir / "best"
            bundle.mkdir()
            (bundle / "model.pt").write_bytes(b"model")
            (bundle / "normalization.json").write_text("{}", encoding="utf-8")
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "job_hash": config_hash(job),
                        "job": {"dataset": canonical_data(job.dataset)},
                    }
                ),
                encoding="utf-8",
            )
            metrics = run_dir / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "final": {
                            "evaluation": {"metrics": {"rmse": 0.1}},
                            "test": {"metrics": {"rmse": 0.2}},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (run_dir / "artifacts.json").write_text(
                json.dumps(
                    {"artifacts": {"best": str(bundle), "metrics": str(metrics)}}
                ),
                encoding="utf-8",
            )

            with patch.object(
                pipeline, "expected_run_directory", return_value=run_dir
            ):
                completed = pipeline.preflight_forecast(job, "experiment")
            self.assertEqual(completed.bundle, bundle.resolve())

            (run_dir / "status.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            with patch.object(
                pipeline, "expected_run_directory", return_value=run_dir
            ), self.assertRaisesRegex(ValueError, "not completed"):
                pipeline.preflight_forecast(job, "experiment")

    def test_preflight_accepts_only_the_five_declared_fold_jobs(self) -> None:
        experiment = build_experiment()
        extra_job = replace(experiment.jobs[0], name="extra-forecast-job")
        with self.assertRaisesRegex(ValueError, "exactly 5 fold jobs"):
            pipeline.preflight_all_forecasts(
                replace(experiment, jobs=experiment.jobs + (extra_job,))
            )

        wrong_job = replace(experiment.jobs[0], name="unexpected-forecast-job")
        with self.assertRaisesRegex(ValueError, "does not match its fold"):
            pipeline.preflight_all_forecasts(
                replace(experiment, jobs=(wrong_job,) + experiment.jobs[1:])
            )

    def test_fold_teacher_validation_checks_exact_oof_coverage(self) -> None:
        fold = FOLDS[0]
        job = build_experiment().jobs[0]
        scenarios = list(fold.held_out.scenarios)
        identities = [
            identity
            for scenario in scenarios
            for identity in (f"{scenario}-trajectory",) * 2
        ]
        scenario_rows = [scenario for scenario in scenarios for _ in range(2)]
        queries = [query for _scenario in scenarios for query in (47, 95)]
        arrays = {
            name: np.zeros(len(scenario_rows), dtype=np.float32)
            for name in cem.REQUIRED_DATASET_ARRAYS
        }
        arrays["scenario_id"] = np.asarray(scenario_rows, dtype=np.str_)
        arrays["trajectory_identity"] = np.asarray(identities, dtype=np.str_)
        arrays["query_index"] = np.asarray(queries, dtype=np.int64)
        arrays["policy_observation_history"] = np.zeros(
            (len(scenario_rows), cem.HISTORY_STEPS, cem.POLICY_OBSERVATION_SIZE),
            dtype=np.float32,
        )
        arrays["best_raw_actions"] = np.zeros(
            (len(scenario_rows), cem.HORIZON_STEPS), dtype=np.float32
        )
        arrays["predicted_chlorine"] = np.zeros(
            (len(scenario_rows), cem.HORIZON_STEPS, 31), dtype=np.float32
        )
        identity = {}
        manifest = {
            "source_dataset": {
                "network": forecasting_full.network,
                "contract": canonical_data(fold.held_out),
            },
            "forecast_job_hash": config_hash(job),
            "forecast_training_dataset_contract": canonical_data(job.dataset),
            "forecasting_bundle": str(Path("bundle").resolve()),
            "teacher_configuration": identity,
            "teacher_configuration_hash": config_hash(identity),
            "action_contract": cem.action_contract(),
            "reward": cem.reward_contract(cem.DEFAULT_DISCOUNT_FACTOR),
            "policy_observation_contract": {
                "spec": canonical_data(cem.V89_OBSERVATION),
                "history_steps": cem.HISTORY_STEPS,
                "observation_size": cem.POLICY_OBSERVATION_SIZE,
                "max_injection": cem.MAX_INJECTION,
            },
            "node_ids": [f"node-{index}" for index in range(31)],
            "out_of_fold": {
                "required": True,
                "source_scenarios": list(fold.held_out.scenarios),
                "forecast_training_scenarios": list(job.dataset.train.scenarios),
            },
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        }
        forecast = pipeline.CompletedForecast(
            job,
            Path("run"),
            Path("bundle"),
            Path("metrics"),
            {},
            {},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.npz"
            manifest_path = root / "manifest.json"
            np.savez_compressed(dataset_path, **arrays)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                patch.object(pipeline, "_expected_trajectory_count", return_value=16),
                patch.object(cem, "DEFAULT_QUERIES_PER_TRAJECTORY", 2),
            ):
                loaded, _manifest = pipeline.validate_fold_teacher(
                    0, dataset_path, manifest_path, forecast
                )
        self.assertEqual(len(loaded["scenario_id"]), 32)

        self.assertEqual(loaded["policy_observation_history"].shape[-1], 43)

    def _fold_artifacts(self, root: Path) -> tuple[pipeline.FoldArtifact, ...]:
        artifacts = []
        for fold in FOLDS:
            fold_root = root / f"fold-{fold.index}"
            validation = fold_root / "validation"
            validation.mkdir(parents=True)
            bundle = fold_root / "forecast-bundle"
            bundle.mkdir()
            for name in ("manifest.json", "model.pt", "normalization.json"):
                (bundle / name).write_bytes(
                    f"fold-{fold.index}-{name}".encode("utf-8")
                )
            dataset_path = fold_root / "dataset.npz"
            manifest_path = fold_root / "manifest.json"
            scenario = fold.held_out.scenarios[0]
            arrays = {
                "scenario_id": np.asarray([scenario], dtype=np.str_),
                "trajectory_identity": np.asarray(
                    [f"trajectory-{fold.index}"], dtype=np.str_
                ),
                "query_index": np.asarray([47], dtype=np.int64),
                "first_action": np.asarray([fold.index / 10], dtype=np.float32),
            }
            np.savez_compressed(dataset_path, **arrays)
            manifest = {
                "schema_version": 2,
                "kind": "mpc-teacher-cem",
                "action_contract": {"maximum_injection": 500_000.0},
                "forecast_job_hash": f"fold-job-{fold.index}",
                "forecasting_bundle": str(bundle),
                "teacher_configuration_hash": f"teacher-{fold.index}",
                "source_dataset": {"contract": canonical_data(fold.held_out)},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            summary = {"valid_count": 32}
            (validation / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            np.savez_compressed(
                validation / "validation_cases.npz",
                scenario_id=np.asarray([scenario], dtype=np.str_),
            )
            artifacts.append(
                pipeline.FoldArtifact(
                    fold.index,
                    dataset_path,
                    manifest_path,
                    validation,
                    manifest,
                    arrays,
                    summary,
                )
            )
        return tuple(artifacts)

    def test_merge_is_deterministic_and_preserves_fold_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = self._fold_artifacts(root)
            arrays, manifest = pipeline.merge_fold_artifacts(folds)

            self.assertEqual(arrays["fold_index"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(
                arrays["scenario_id"].tolist(),
                [f"scenario-{index}" for index in range(5)],
            )
            self.assertEqual(len(manifest["cross_fitting"]["folds"]), 5)
            self.assertNotIn("forecasting_bundle", manifest)
            self.assertIn("aggregate_teacher_hash", manifest)
            self.assertIn(
                "validation_cases_sha256",
                manifest["cross_fitting"]["folds"][0],
            )
            self.assertIn(
                "model.pt",
                manifest["cross_fitting"]["folds"][0][
                    "labeling_forecast_bundle_hashes"
                ],
            )
            dataset_path, manifest_path = pipeline.write_merged_dataset(
                root / "merged", arrays, manifest
            )
            self.assertTrue(dataset_path.is_file())
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_manifest["arrays"]["fold_index"]["shape"], [5])

    def test_merge_rejects_duplicate_trajectory_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = list(self._fold_artifacts(root))
            folds[1].arrays["trajectory_identity"][0] = folds[0].arrays[
                "trajectory_identity"
            ][0]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                pipeline.merge_fold_artifacts(tuple(folds))

    def test_validation_gate_requires_two_valid_cases_per_scenario(self) -> None:
        fold = FOLDS[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "selected_count": 32,
                "valid_count": 32,
                "failed_count": 0,
                "cancelled_count": 0,
                "interrupted": False,
            }
            (root / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            scenarios = [scenario for scenario in fold.held_out.scenarios for _ in range(2)]
            np.savez_compressed(
                root / "validation_cases.npz",
                scenario_id=np.asarray(scenarios, dtype=np.str_),
            )
            self.assertEqual(
                pipeline.validate_fold_replay(0, root)["valid_count"], 32
            )


if __name__ == "__main__":
    unittest.main()
