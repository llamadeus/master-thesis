from __future__ import annotations

import io
import json
import tempfile
from unittest.mock import patch
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np

from packages.trainer.serialization import config_hash
from training import run_principal_v89_pipeline as pipeline
from training.datasets.hanoi import control_full, forecasting_full
from training.experiments.control_v89_baseline import build_experiment as build_v89_baseline
from training.rewards.v89 import DEFAULT_PARAMETERS, V89Observation, V89Reward


class PrincipalV89PipelineTests(unittest.TestCase):
    def test_bc_initial_action_std_uses_final_cem_first_action_median(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            deviations = np.full((3, 4, 16), 0.7, dtype=np.float32)
            deviations[:, -1, 0] = (0.02, 0.08, 0.2)
            np.savez(
                dataset / "dataset.npz",
                cem_iteration_action_std=deviations,
                first_action=np.zeros(3, dtype=np.float32),
            )
            configuration = {
                "teacher_dataset": str(dataset),
                "initial_action_std_source": pipeline.CEM_ACTION_STD_SOURCE,
                "seed": 17,
                "validation_teacher_dataset": str(dataset),
                "epochs": 1,
                "trajectory_batch_size": 1,
                "sequence_chunk_length": 1,
                "learning_rate": 1e-3,
                "patience": 1,
            }
            self.assertAlmostEqual(pipeline._initial_action_std(configuration), 0.08)
            arguments = pipeline._bc_arguments(
                configuration, dataset / "bundle", workers=1, device="cpu"
            )
            self.assertAlmostEqual(arguments.initial_action_std, 0.08)
            self.assertEqual(arguments.initial_action_std_source, pipeline.CEM_ACTION_STD_SOURCE)

    def test_paper_seeds_are_the_default_replicates(self) -> None:
        self.assertEqual(pipeline.SEEDS, (17, 42, 123, 1337, 71026))

    def test_pipeline_records_stage_timings_and_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, "PIPELINE_STATE_ROOT", Path(directory)):
            with pipeline._measure_stage(
                17, "forecasting", workers=2, concurrent_jobs=1,
                device_source="forecast-job-config",
            ):
                pass

            root = Path(directory) / "seed-17"
            timings = json.loads((root / "stage-timings.json").read_text())
            provenance_exists = (root / "provenance.json").exists()

        self.assertFalse(provenance_exists)
        self.assertEqual(timings["records"][0]["stage"], "forecasting")
        self.assertEqual(timings["records"][0]["status"], "completed")
        self.assertEqual(timings["records"][0]["settings"]["workers"], 2)
        self.assertEqual(timings["records"][0]["settings"]["concurrent_jobs"], 1)
        self.assertGreaterEqual(timings["records"][0]["elapsed_seconds"], 0.0)

    def test_seed_is_propagated_through_forecasting_teacher_bc_and_control(self) -> None:
        seed = 1337
        forecast = pipeline.build_forecasting_experiment(seed)
        teacher = pipeline.teacher_configuration(forecast, seed)
        teacher_output = pipeline.teacher_output_directory(teacher)
        bc = pipeline.behavior_cloning_configuration(seed, teacher_output)
        bc_output = pipeline.behavior_cloning_output_directory(bc)
        control = pipeline.build_control_experiment(seed, bc_output)

        self.assertEqual({job.training_seed for job in forecast.jobs}, {seed})
        self.assertEqual(len(forecast.jobs), 5)
        self.assertEqual(teacher["selection"]["seed"], seed)
        self.assertEqual(bc["seed"], seed)
        self.assertEqual(bc["reward_version"], "v89")
        self.assertEqual(bc["loss"], "mse")
        self.assertEqual(bc["initial_action_std_source"], pipeline.CEM_ACTION_STD_SOURCE)
        self.assertNotIn("train_fraction", bc)
        validation = pipeline.validation_teacher_configuration(forecast, seed)
        self.assertEqual(validation["source_dataset"]["split"]["scenarios"],
                         list(forecasting_full.validation.scenarios))
        self.assertEqual(validation["forecast"]["job_hashes"],
                         [config_hash(job) for job in forecast.jobs])
        self.assertNotEqual(bc["teacher_dataset"], bc["validation_teacher_dataset"])
        self.assertEqual(len(teacher["source_dataset"]["split"]["scenarios"]), 80)
        self.assertEqual({job.training_seed for job in control.jobs}, {seed})
        self.assertEqual(
            control.job("mpc-teacher-bc").algorithm.pretrained_policy,
            bc_output,
        )
        self.assertEqual(
            [fold["job_hash"] for fold in teacher["forecast"]["folds"]],
            [config_hash(job) for job in forecast.jobs[:5]],
        )

    def test_mpc_generation_records_fold_training_and_validation_wall_times(self) -> None:
        from tools.mpc import generate_cross_fit_mpc_teacher_dataset_cem as cross_fit
        from tools.mpc import generate_validation_mpc_teacher_dataset_cem as ensemble

        forecast = pipeline.build_forecasting_experiment(17)
        runs = tuple(
            cross_fit.CompletedForecast(job, Path("run"), Path("bundle"), Path("metrics"), {}, {})
            for job in forecast.jobs
        )
        clock = [0.0]

        def generate(args, **kwargs):
            clock[0] += 10.0
            return args.output_dir / "dataset.npz", args.output_dir / "manifest.json"

        def replay(*args):
            clock[0] += 2.0
            return {}

        def merge_output(output, *args):
            clock[0] += 3.0
            return output / "dataset.npz", output / "manifest.json"

        def validation(*args, **kwargs):
            clock[0] += 20.0

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, "PIPELINE_STATE_ROOT", Path(directory)), \
             patch.object(pipeline.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(pipeline, "_require_forecasting_complete"), \
             patch.object(pipeline, "_teacher_is_complete", return_value=False), \
             patch.object(pipeline, "_archive_incomplete", return_value=None), \
             patch.object(cross_fit, "preflight_all_forecasts", return_value=runs), \
             patch.object(cross_fit.cem, "generate_dataset", side_effect=generate), \
             patch.object(cross_fit, "validate_fold_teacher", return_value=({}, {})), \
             patch.object(cross_fit.validator, "ensure_fresh_output"), \
             patch.object(cross_fit.validator, "run_validation", return_value=([], {}, False)), \
             patch.object(cross_fit.validator, "cases_to_arrays", return_value={}), \
             patch.object(cross_fit.validator, "_atomic_npz"), \
             patch.object(cross_fit.validator, "_atomic_json"), \
             patch.object(cross_fit.validator, "write_plots"), \
             patch.object(cross_fit, "validate_fold_replay", side_effect=replay), \
             patch.object(cross_fit, "merge_fold_artifacts", return_value=({}, {})), \
             patch.object(cross_fit, "write_merged_dataset", side_effect=merge_output), \
             patch.object(cross_fit, "forecast_metrics_report", return_value={}), \
             patch.object(cross_fit, "_atomic_json"), \
             patch.object(ensemble, "execute_pipeline", side_effect=validation):
            pipeline.run_seed(17, workers=2, validation_workers=4, device="cpu",
                              generate_bc_plots=False, stages=("mpc",))
            records = json.loads(
                (Path(directory) / "seed-17/stage-timings.json").read_text()
            )["records"]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "mpc")
        self.assertEqual(records[0]["elapsed_seconds"], 83.0)
        training, validation = records[0]["components"]
        self.assertEqual(training["stage"], "mpc-teacher-training")
        self.assertEqual(training["elapsed_seconds"], 63.0)
        self.assertEqual(validation["stage"], "mpc-teacher-validation")
        self.assertEqual(validation["elapsed_seconds"], 20.0)
        folds = training["components"]
        self.assertEqual([fold["settings"]["fold_index"] for fold in folds], list(range(5)))
        self.assertTrue(all(fold["elapsed_seconds"] == 12.0 for fold in folds))
        self.assertTrue(all(fold["status"] == "completed" for fold in folds))
        self.assertTrue(all(fold["settings"]["validation_workers"] == 4 for fold in folds))

    def test_nested_timings_preserve_failed_and_interrupted_attempts(self) -> None:
        for exception, status in ((RuntimeError("generation failed"), "failed"),
                                  (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, \
                 patch.object(pipeline, "PIPELINE_STATE_ROOT", Path(directory)):
                with self.assertRaises(type(exception)):
                    with pipeline._measure_stage(17, "mpc") as components:
                        with pipeline._measure_stage(
                            17, "mpc-teacher-training", records=components,
                        ) as folds:
                            with pipeline._measure_stage(
                                17, "mpc-teacher-fold", records=folds, fold_index=0,
                            ):
                                pass
                            with pipeline._measure_stage(
                                17, "mpc-teacher-fold", records=folds, fold_index=1,
                            ):
                                raise exception
                record = json.loads(
                    (Path(directory) / "seed-17/stage-timings.json").read_text()
                )["records"][0]
                self.assertEqual(record["status"], status)
                training = record["components"][0]
                self.assertEqual(training["status"], status)
                self.assertEqual([fold["status"] for fold in training["components"]],
                                 ["completed", status])

    def test_baseline_and_fine_tuning_are_matched_except_initialization(self) -> None:
        control = pipeline.build_control_experiment(
            42,
            pipeline.PROJECT_ROOT / "hypothetical-bc-bundle",
        )
        baseline = control.job("baseline")
        fine_tuning = control.job("mpc-teacher-bc")

        self.assertEqual(baseline.dataset, fine_tuning.dataset)
        self.assertEqual(baseline.dataset, control_full)
        self.assertEqual(baseline.observation, fine_tuning.observation)
        self.assertEqual(baseline.reward, fine_tuning.reward)
        self.assertIs(baseline.reward.factory, V89Reward)
        self.assertIs(baseline.observation.factory, V89Observation)
        self.assertEqual(baseline.reward.as_dict(), DEFAULT_PARAMETERS)
        self.assertEqual(baseline.observation.as_dict(), DEFAULT_PARAMETERS)
        self.assertEqual(baseline.algorithm, build_v89_baseline().jobs[0].algorithm)
        self.assertEqual(
            {key: getattr(baseline.algorithm, key) for key in (
                "n_envs", "n_steps", "learning_rate", "batch_size", "ent_coef",
                "n_epochs", "gamma", "gae_lambda", "clip_range", "target_kl",
                "vf_coef", "max_grad_norm", "lstm_hidden_size", "n_lstm_layers",
            )},
            {
                "n_envs": 4, "n_steps": 2048, "learning_rate": 1e-4,
                "batch_size": 512, "ent_coef": 0.0, "n_epochs": 10,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                "target_kl": 0.02, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
            },
        )
        self.assertEqual(DEFAULT_PARAMETERS, {
            "target": 0.30,
            "tracking_reference": 0.01,
            "alpha_one_threshold": 0.15,
            "alpha_zero_threshold": 0.20,
            "blend_grace_steps": 4.0,
            "blend_decay_steps": 12.0,
            "warmup_steps": 12.0,
            "blend_steps": 12.0,
            "upper_penalty_min_factor": 0.0001,
            "wasted_movement_reference": 0.01,
            "wasted_movement_weight": 0.1,
            "wasted_movement_gate_full_threshold": 0.35,
            "wasted_movement_gate_zero_threshold": 0.40,
            "action_jerk_reference": 0.05,
            "action_jerk_weight": 0.1,
        })
        self.assertEqual(baseline.total_timesteps, fine_tuning.total_timesteps)
        self.assertEqual(baseline.total_timesteps, 1_000_000)
        self.assertEqual(baseline.algorithm.n_envs * baseline.algorithm.n_steps, 8_192)
        self.assertEqual(baseline.max_injection, fine_tuning.max_injection)
        self.assertEqual(baseline.max_delta, fine_tuning.max_delta)
        self.assertEqual(baseline.initial_injection, fine_tuning.initial_injection)
        self.assertEqual(
            baseline.algorithm,
            replace(fine_tuning.algorithm, pretrained_policy=None),
        )

    def test_resume_reuses_training_labels_and_adds_validation_stage(self):
        from tools.mpc import generate_validation_mpc_teacher_dataset_cem as ensemble
        from tools.control.bc import train_mpc_teacher_recurrent_ppo as bc_trainer

        events = []
        with patch.object(pipeline, "_write_json") as write_json, patch.object(pipeline, "_set_status"), \
             patch.object(pipeline, "_read_json", return_value=None), \
             patch.object(pipeline, "_require_forecasting_complete"), \
             patch.object(pipeline, "_require_teacher_datasets_complete"), \
             patch.object(pipeline, "_teacher_is_complete", side_effect=[True, False]), \
             patch.object(pipeline, "_archive_incomplete", return_value=None) as archive, \
             patch.object(pipeline, "_run_teacher") as training_teacher, \
             patch.object(pipeline, "_bc_is_complete", return_value=False), \
             patch.object(ensemble, "execute_pipeline", side_effect=lambda *a, **kw: events.append("validation")) as generate, \
             patch.object(bc_trainer, "train", side_effect=lambda *a: events.append("bc")) as train, \
             redirect_stdout(io.StringIO()):
            pipeline.run_seed(
                17,
                workers=1,
                validation_workers=1,
                device="cpu",
                generate_bc_plots=False,
                stages=("mpc", "behavior-cloning"),
            )
        training_teacher.assert_not_called()
        mpc = next(
            record
            for call in write_json.call_args_list
            if call.args[0].name == "stage-timings.json"
            for record in call.args[1]["records"]
            if record["stage"] == "mpc"
        )
        self.assertEqual([component["stage"] for component in mpc["components"]],
                         ["mpc-teacher-validation"])
        self.assertEqual(events, ["validation", "bc"])
        self.assertEqual(generate.call_args.args[0].seed, 17)
        self.assertEqual({job.training_seed for job in generate.call_args.kwargs["experiment"].jobs}, {17})
        args = train.call_args.args[0]
        self.assertNotEqual(args.dataset, args.validation_dataset)
        self.assertTrue(any("mpc-teacher-bc" in str(call.args[0]) for call in archive.call_args_list))

    def test_dry_run_prints_plan_without_starting_execution(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = pipeline.main(["run", "--seed", "17", "--dry-run"])

        self.assertEqual(result, 0)
        self.assertIn('"seed": 17', stream.getvalue())
        self.assertIn('"training_seed": 17', stream.getvalue())
        self.assertIn('"reward_version": "v89"', stream.getvalue())
        self.assertIn('"pipeline": "hanoi-principal-v89"', stream.getvalue())

    def test_bc_plots_require_explicit_opt_in(self) -> None:
        parser = pipeline.build_parser()
        self.assertFalse(parser.parse_args(["run"]).generate_bc_plots)
        self.assertTrue(
            parser.parse_args(["run", "--generate-bc-plots"]).generate_bc_plots
        )

    def test_training_registry_disables_plots_without_changing_job_hashes(self) -> None:
        control = pipeline.build_control_experiment(17, Path("hypothetical-bc-bundle"))
        forecast = pipeline.build_forecasting_experiment(17)
        hashes_before = [config_hash(job) for job in (*control.jobs, *forecast.jobs)]

        registry = pipeline._training_registry()

        self.assertFalse(registry.runner_for("control").render_plots)
        self.assertTrue(registry.runner_for("control").defer_evaluation)
        self.assertTrue(registry.runner_for("control").evaluate_best_model)
        self.assertFalse(registry.runner_for("forecast").render_plots)
        self.assertEqual(
            hashes_before,
            [config_hash(job) for job in (*control.jobs, *forecast.jobs)],
        )

    def test_control_training_and_deferred_evaluation_have_separate_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []

            def execute(*args, **kwargs):
                job = next(iter(kwargs["selected"]))
                runner = kwargs["registry"].runner_for("control")
                self.assertTrue(runner.defer_evaluation)
                self.assertFalse(runner.render_plots)
                runner.on_training_complete()
                records = json.loads(
                    (root / "seed-17" / "stage-timings.json").read_text()
                )["records"]
                self.assertEqual(records[-1]["stage"],
                                 "baseline" if job == "baseline" else "fine-tuning")
                events.append(job)

            with patch.object(pipeline, "PIPELINE_STATE_ROOT", root), \
                 patch.object(pipeline, "_require_matched_teacher_contract"), \
                 patch.object(pipeline, "_require_behavior_cloning_complete"), \
                 patch.object(pipeline, "execute_experiment", side_effect=execute), \
                 redirect_stdout(io.StringIO()):
                pipeline.run_seed(17, workers=1, validation_workers=1, device="cpu",
                                  generate_bc_plots=False, stages=("baseline", "fine-tuning"))

            records = json.loads((root / "seed-17" / "stage-timings.json").read_text())["records"]
            self.assertEqual(events, ["baseline", "mpc-teacher-bc"])
            self.assertEqual([record["stage"] for record in records],
                             ["baseline", "baseline-evaluation", "fine-tuning",
                              "fine-tuning-evaluation"])

    def test_deferred_evaluation_failure_keeps_training_time_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def execute(*args, **kwargs):
                kwargs["registry"].runner_for("control").on_training_complete()
                raise RuntimeError("evaluation failed")

            with patch.object(pipeline, "PIPELINE_STATE_ROOT", root), \
                 patch.object(pipeline, "execute_experiment", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    pipeline.run_seed(17, workers=1, validation_workers=1, device="cpu",
                                      generate_bc_plots=False, stages=("baseline",))

            records = json.loads((root / "seed-17" / "stage-timings.json").read_text())["records"]
            self.assertEqual([(record["stage"], record["status"]) for record in records],
                             [("baseline", "completed"), ("baseline-evaluation", "failed")])
            status = json.loads((root / "seed-17" / "status.json").read_text())
            self.assertEqual((status["phase"], status["status"]),
                             ("baseline-evaluation", "failed"))

    def test_requested_bc_plots_run_after_training_stage_timings(self) -> None:
        from tools.control.bc import train_mpc_teacher_recurrent_ppo as bc_trainer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []

            def generate_plots(*args, **kwargs):
                records = json.loads(
                    (root / "state" / "seed-17" / "stage-timings.json").read_text()
                )["records"]
                self.assertEqual([record["stage"] for record in records],
                                 ["behavior-cloning", "baseline"])
                events.append("plots")

            with patch.object(pipeline, "PIPELINE_STATE_ROOT", root / "state"), \
                 patch.object(pipeline, "behavior_cloning_output_directory", return_value=root / "bc"), \
                 patch.object(pipeline, "_require_matched_teacher_contract"), \
                 patch.object(pipeline, "_require_teacher_datasets_complete"), \
                 patch.object(pipeline, "_bc_is_complete", return_value=True), \
                 patch.object(pipeline, "execute_experiment", side_effect=lambda *a, **kw: events.append("baseline")), \
                 patch.object(bc_trainer, "generate_final_plots", side_effect=generate_plots), \
                 redirect_stdout(io.StringIO()):
                pipeline.run_seed(17, workers=1, validation_workers=1, device="cpu",
                                  generate_bc_plots=True, stages=("behavior-cloning", "baseline"))

            timings = json.loads((root / "state" / "seed-17" / "stage-timings.json").read_text())
            self.assertEqual(events, ["baseline", "plots"])
            self.assertEqual([record["stage"] for record in timings["records"]],
                             ["behavior-cloning", "baseline", "behavior-cloning-plots"])

    def test_teacher_contract_matches_v89_control(self) -> None:
        forecast = pipeline.build_forecasting_experiment(17)
        teacher = pipeline.teacher_configuration(forecast, 17)
        validation = pipeline.validation_teacher_configuration(forecast, 17)
        control = pipeline.build_control_experiment(17, Path("hypothetical-bc-bundle"))
        pipeline._require_matched_teacher_contract(teacher, validation, control)
        for configuration in (teacher, validation):
            self.assertEqual(configuration["reward"]["id"], "v89")
            self.assertEqual(configuration["reward"]["parameters"], DEFAULT_PARAMETERS)
            self.assertEqual(configuration["policy_observation_contract"]["observation_size"], 43)

    def test_stage_selection_is_canonical_and_repeatable(self) -> None:
        args = pipeline.build_parser().parse_args(
            [
                "run",
                "--seed",
                "17",
                "--stage",
                "fine-tuning",
                "--stage",
                "forecasting",
            ]
        )

        self.assertEqual(
            pipeline._selected_stages(args.stage),
            ("forecasting", "fine-tuning"),
        )

    def test_forecasting_and_baseline_can_run_independently(self) -> None:
        for stage, selected in (
            ("forecasting", None),
            ("baseline", {"baseline"}),
        ):
            with self.subTest(stage=stage), \
                 patch.object(pipeline, "_write_json"), \
                 patch.object(pipeline, "_set_status"), \
                 patch.object(pipeline, "execute_experiment") as execute:
                pipeline.run_seed(
                    17,
                    workers=1,
                    validation_workers=1,
                    device="cpu",
                    generate_bc_plots=False,
                    stages=(stage,),
                )

            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs.get("selected"), selected)

    def test_mpc_stage_fails_when_forecasting_is_incomplete(self) -> None:
        message = "stage 'mpc' requires completed forecasting artifacts"
        stderr = io.StringIO()
        with patch.object(pipeline, "_write_json"), \
             patch.object(pipeline, "_set_status"), \
             patch.object(
                 pipeline,
                 "_require_forecasting_complete",
                 side_effect=RuntimeError(message),
             ), \
             patch.object(pipeline, "_run_teacher") as teacher, \
             redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            result = pipeline.main(
                ["run", "--seed", "17", "--stage", "mpc"]
            )

        self.assertEqual(result, 2)
        self.assertIn(message, stderr.getvalue())
        teacher.assert_not_called()

    def test_behavior_cloning_requires_both_teacher_datasets(self) -> None:
        message = "requires completed MPC artifacts"
        stderr = io.StringIO()
        with patch.object(pipeline, "_write_json"), \
             patch.object(pipeline, "_set_status"), \
             patch.object(
                 pipeline,
                 "_require_teacher_datasets_complete",
                 side_effect=RuntimeError(message),
             ), \
             redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            result = pipeline.main(
                ["run", "--seed", "17", "--stage", "behavior-cloning"]
            )

        self.assertEqual(result, 2)
        self.assertIn(message, stderr.getvalue())

    def test_fine_tuning_requires_completed_behavior_cloning(self) -> None:
        message = "requires a completed behavior-cloning bundle"
        stderr = io.StringIO()
        with patch.object(pipeline, "_write_json"), \
             patch.object(pipeline, "_set_status"), \
             patch.object(
                 pipeline,
                 "_require_behavior_cloning_complete",
                 side_effect=RuntimeError(message),
             ), \
             patch.object(pipeline, "execute_experiment") as execute, \
             redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            result = pipeline.main(
                ["run", "--seed", "17", "--stage", "fine-tuning"]
            )

        self.assertEqual(result, 2)
        self.assertIn(message, stderr.getvalue())
        execute.assert_not_called()

    def test_dependency_checks_reject_incomplete_artifacts(self) -> None:
        forecast = pipeline.build_forecasting_experiment(17)
        with patch(
            "tools.mpc.generate_cross_fit_mpc_teacher_dataset_cem."
            "preflight_all_forecasts",
            side_effect=FileNotFoundError("missing status"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "run stage 'forecasting' first",
            ):
                pipeline._require_forecasting_complete(forecast, 17)

        with patch.object(
            pipeline,
            "_teacher_is_complete",
            side_effect=(False, True),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "training teacher dataset.*run stage 'mpc' first",
            ):
                pipeline._require_teacher_datasets_complete(
                    pipeline.PROJECT_ROOT / "training-teacher",
                    {},
                    pipeline.PROJECT_ROOT / "validation-teacher",
                    {},
                    17,
                )

        with patch.object(pipeline, "_bc_is_complete", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError,
                "run stage 'behavior-cloning' first",
            ):
                pipeline._require_behavior_cloning_complete(
                    pipeline.PROJECT_ROOT / "bc",
                    {},
                    17,
                )


if __name__ == "__main__":
    unittest.main()
