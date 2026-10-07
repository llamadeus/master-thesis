from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from paper.tools.results_projection import SEEDS, scenario_metrics
from tools.analysis import export_v89_ablation_paper_results as exporter


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_run(root: Path, *, budget: int = 1_000_000, continuation_cutoff: int | None = None) -> Path:
    """Small artifact fixture exercising provenance without real model inference."""
    directory = root / "012345abcdef"
    policy = "baseline-4m" if budget > 1_000_000 else "without-actor-weights"
    test_cases = [dict(scenario_id=f"scenario-{90 + index % 10}", schedule_hash=f"schedule-{index // 10}",
                       index=index, file=f"case-{index:05d}.npz") for index in range(100)]
    validation_cases = [dict(scenario_id=f"scenario-{80 + index % 10}", schedule_hash=f"schedule-{index // 10}")
                        for index in range(50)]
    configured_budget = 3_400_000 if continuation_cutoff is not None else budget
    job = dict(training_seed=17, total_timesteps=configured_budget, kind="control", evaluate_test=True,
               reward=dict(factory="training.rewards.v89:V89Reward"),
               observation=dict(factory="training.rewards.v89:V89Observation"),
               algorithm=dict(n_envs=4, n_steps=2048, pretrained_policy="bundle" if budget == 1_000_000 else None,
                              pretrained_components=list(exporter.VARIANTS[policy])),
               dataset=dict(test=dict(scenarios=[f"scenario-{i}" for i in range(90, 100)],
                                      arsenic=dict(schedule_count=10, seed=8130))))
    save(directory / "config.resolved.json", dict(job=job, job_hash=directory.name, schema_version=2))
    save(directory / "status.json", dict(status="completed"))
    history = []
    for nominal in range(0, budget + 1, 50_000):
        actual = ((nominal + 8191) // 8192) * 8192
        checkpoint = dict(timesteps=actual, nominal_thresholds=[nominal],
                          evaluation=dict(evaluation_return=-abs(nominal - 100_000), status="complete",
                                          protocol_version=3 if continuation_cutoff is not None and actual > continuation_cutoff else 2,
                                          cases=[dict(case, values=dict(evaluation_return=-abs(nominal - 100_000)))
                                                 for case in validation_cases]))
        history.append(checkpoint)
        snapshot = directory / "checkpoints/history" / f"{actual:012d}"
        save(snapshot / "evaluation.json", checkpoint)
        save(snapshot / "manifest.json", dict(job_hash=directory.name, timesteps=actual, nominal_thresholds=[nominal],
                                              schema_version=1, kind="recurrent-ppo-training-checkpoint"))
        save(snapshot / "trainer_state.json", dict(completed_timesteps=actual))
        (snapshot / "model.zip").write_bytes(f"model-{actual}".encode())
        (snapshot / "vec_normalize.pkl").write_bytes(f"normalization-{actual}".encode())
    recorded_history = [row for row in history if continuation_cutoff is None or row["timesteps"] > continuation_cutoff]
    best = max(recorded_history, key=lambda checkpoint: checkpoint["evaluation"]["evaluation_return"])
    save(directory / "metrics.json", dict(status="complete", protocol_version=3 if continuation_cutoff is not None else 2, history=recorded_history,
         final_timesteps=history[-1]["timesteps"], final_validation=None, best_mean_validation_reward=best["evaluation"]["evaluation_return"],
         runtime={}, test=dict(status="complete", cases=test_cases, evaluation_return=-12)))
    if continuation_cutoff is not None:
        save(directory / "checkpoints/latest.json", dict(job_hash=directory.name,
             target_timesteps=budget, completed_timesteps=history[-1]["timesteps"],
             evaluation_protocol_version=3, history=recorded_history,
             legacy_evaluation_cutoff_timesteps=continuation_cutoff))
    save(directory / "best/manifest.json", dict(job=job, job_hash=directory.name, schema_version=1,
         kind="recurrent-ppo", node_ids=[str(i) for i in range(2, 33)], evaluation=best["evaluation"]))
    for name in ("model.zip", "vec_normalize.pkl"):
        source = directory / "checkpoints/history" / f"{best['timesteps']:012d}" / name
        (directory / "best" / name).write_bytes(source.read_bytes())
    trace_dir = directory / "evaluation-data/final/test"
    save(trace_dir / "manifest.json", dict(kind="control-evaluation-timeseries", schema_version=1,
         split="test", phase="test", duration_seconds=25 * 86400, hydraulic_timestep_seconds=1800,
         node_ids=[str(i) for i in range(2, 33)], case_count=100, cases=test_cases))
    for case in test_cases:
        (trace_dir / case["file"]).write_bytes(b"not loaded by provenance validation")
    return directory


class AblationAggregationTests(unittest.TestCase):
    def test_equal_scenario_weighting_and_explicit_nonapplicability(self) -> None:
        raw = []
        for seed in SEEDS:
            for scenario in range(10):
                values = [0, 10, None] if scenario == 0 else [None, 2]
                raw.extend(dict(approach="variant", seed=str(seed), scenario=str(scenario),
                                metric="arsenic_burden", component="total", value=value) for value in values)
        projected = scenario_metrics(raw)
        result = exporter.summarize(projected, ("variant",), (("arsenic_burden", "total"),))[0]
        self.assertEqual(result["seed_values"], [2.3] * 5)
        self.assertEqual(projected[0]["applicable_count"], 2)
        self.assertEqual(projected[0]["not_applicable_count"], 1)

    def test_paired_ci_uses_five_seed_differences_and_direction(self) -> None:
        base = dict(approach="base", metric="arsenic_burden", component="total", unit="u", seed_values=[10] * 5)
        variant = dict(base, approach="variant", seed_values=[11, 12, 13, 14, 15])
        result = exporter.paired_summary(variant, base)
        self.assertEqual(result["paired_difference"]["seed_values"], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(result["paired_difference"]["mean"], 3)
        self.assertAlmostEqual(result["ci95"][0], 1.036756838522439)
        variant["seed_values"][0] = None
        result = exporter.paired_summary(variant, base)
        self.assertIsNone(result["ci95"])
        self.assertEqual(result["paired_difference"]["applicable_seeds"], 4)

    def test_clearance_support_is_specific_to_pair_and_does_not_fill_none(self) -> None:
        case = dict(training_seed=17, scenario_id="scenario-90", clearance_events=[
            dict(reference_clearance_h=20, policy_clearance_h=dict(a=10, b=12, third=None)),
            dict(reference_clearance_h=30, policy_clearance_h=dict(a=None, b=25, third=20)),
        ])
        support = defaultdict(int)
        rows = exporter.clearance_pair_rows(case, "a", "b", support)
        self.assertEqual([row["value"] for row in rows], [10, 8])
        self.assertEqual(support["eligible_events"], 1)
        self.assertEqual(support["paired_exclusions"], 1)
        self.assertEqual(support["variant_no_exceedance"], 1)
        other = exporter.clearance_pair_rows(case, "a", "third", defaultdict(int))
        self.assertTrue(all(row["value"] is None for row in other))

    def test_missing_seed_or_scenario_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "five entries"):
            exporter.seed_statistics([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "ten scenario"):
            exporter.summarize([], ("variant",), (("arsenic_burden", "total"),))
        with self.assertRaisesRegex(ValueError, "Invalid seed value"):
            exporter.seed_statistics([1, 2, float("nan"), 4, 5])
        self.assertIsNone(exporter.seed_statistics([None] * 5)["mean"])


class AblationArtifactValidationTests(unittest.TestCase):
    def test_paper_validation_provenance_is_relative_to_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "external-results"
            policies = ("baseline", "pretrained", *exporter.PAPER_VARIANTS)
            runs = {(policy, seed): root / "weights" / policy / str(seed)
                    for policy in policies for seed in SEEDS}

            def validate(directory, policy, seed, *, validation_only):
                self.assertTrue(validation_only)
                return None, [dict(approach=policy, training_seed=seed, evaluation_return=-12)], dict(
                    run_directory=str(directory), nominal_budget=(
                        exporter.LONG_BASELINE_BUDGET if policy == "baseline-4m" else 1_000_000),
                    validation_inventory=["same-fixed-cases"],
                    runtime=dict(training_seconds=120, checkpoint_evaluation_seconds=30))

            with patch.object(exporter, "discover_runs", return_value=runs), \
                    patch.object(exporter, "validate_run", side_effect=validate), \
                    patch.object(exporter, "read_json", return_value=dict(job={})), \
                    patch.object(exporter, "validate_matched_jobs"):
                data = exporter.build_paper_validation(root)

            self.assertEqual(len(data["source_runs"]), 20)
            self.assertEqual(data["source_runs"][0]["run_directory"], "weights/baseline/17")
            self.assertTrue(all(not Path(row["run_directory"]).is_absolute()
                                for row in data["source_runs"]))
            self.assertNotIn(str(root), json.dumps(data))
            self.assertTrue(all(row["evaluation_return"] == -12 for row in data["rows"]))
            self.assertTrue(all(row["training_seconds"] == 120 for row in data["runtime_rows"]))

    def test_none_final_validation_requires_verified_selected_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = fake_run(Path(temp))
            _, curves, provenance = exporter.validate_run(directory, "without-actor-weights", 17)
            self.assertEqual(len(curves), 21)
            self.assertEqual(provenance["selected_interactions"], 106496)
            self.assertFalse(provenance["final_validation_present"])
            (directory / "best/model.zip").write_bytes(b"different policy")
            with self.assertRaisesRegex(ValueError, "differs from history snapshot"):
                exporter.validate_run(directory, "without-actor-weights", 17)

    def test_long_baseline_uses_all_69_checkpoints_and_actual_rollout_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = fake_run(Path(temp), budget=3_400_000)
            _, curves, provenance = exporter.validate_run(directory, "baseline-4m", 17)
            self.assertEqual(len(curves), 69)
            self.assertEqual(provenance["actual_interactions"], 3_407_872)
            self.assertEqual(curves[-1]["nominal_interactions"], 3_400_000)

    def test_four_million_continuation_recovers_earlier_best_from_all_81_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = fake_run(Path(temp), budget=4_000_000, continuation_cutoff=3_407_872)
            manifest, curves, source = exporter.validate_run(directory, "baseline-4m", 17, validation_only=True)
            self.assertIsNone(manifest)
            self.assertEqual(len(curves), 81)
            self.assertEqual(source["nominal_budget"], 4_000_000)
            self.assertEqual(source["configured_budget"], 3_400_000)
            self.assertEqual(source["actual_interactions"], 4_005_888)
            self.assertEqual(source["selected_interactions"], 106_496)
            self.assertEqual(source["published_selected_interactions"], 3_457_024)
            with self.assertRaisesRegex(ValueError, "not of the full-budget selected policy"):
                exporter.validate_run(directory, "baseline-4m", 17)

    def test_continuation_rejects_missing_old_checkpoint_and_changed_validation_cases(self) -> None:
        for tamper, message in (("missing", "Incomplete learning-curve"),
                                ("inventory", "Validation cases change"),
                                ("protocol", "Incomplete checkpoint evaluation"),
                                ("state", "Incompatible baseline continuation")):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                directory = fake_run(Path(temp), budget=4_000_000, continuation_cutoff=3_407_872)
                path = directory / "checkpoints/history/000000000000/evaluation.json"
                if tamper == "missing":
                    snapshot = path.parent
                    for file in snapshot.iterdir():
                        file.unlink()
                    snapshot.rmdir()
                elif tamper == "state":
                    path = directory / "checkpoints/latest.json"
                    value = exporter.read_json(path)
                    value["target_timesteps"] = 3_700_000
                    save(path, value)
                else:
                    value = exporter.read_json(path)
                    if tamper == "inventory":
                        value["evaluation"]["cases"][0]["schedule_hash"] = "different"
                    else:
                        value["evaluation"]["protocol_version"] = 4
                    save(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    exporter.validate_run(directory, "baseline-4m", 17, validation_only=True)

    def test_tied_validation_return_keeps_earliest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = fake_run(Path(temp))
            metrics = exporter.read_json(directory / "metrics.json")
            later = metrics["history"][3]
            later["evaluation"] = metrics["history"][2]["evaluation"]
            save(directory / "metrics.json", metrics)
            save(directory / "checkpoints/history" / f"{later['timesteps']:012d}" / "evaluation.json", later)
            _, _, provenance = exporter.validate_run(directory, "without-actor-weights", 17)
            self.assertEqual(provenance["selected_interactions"], 106496)

    def test_run_discovery_rejects_duplicate_or_missing_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for policy in exporter.VARIANTS:
                for seed in SEEDS:
                    path = root / "weights/control/ablations/v89" / policy / "hanoi-control-v89-ablations" / policy / str(seed)
                    save(path / "config.resolved.json", dict(experiment="hanoi-control-v89-ablations",
                                                             job=dict(name=policy, training_seed=seed)))
            for seed in SEEDS:
                save(root / "pipelines/hanoi-principal-v89" / f"seed-{seed}" / "plan.json",
                     dict(seed=seed, reward_version="v89", control=dict(
                         baseline=dict(job_hash="baseline"), mpc_teacher_bc=dict(job_hash="pretrained"))))
            self.assertEqual(len(exporter.discover_runs(root)), 30)
            path = root / "weights/control/ablations/v89/without-actor-weights/hanoi-control-v89-ablations/without-actor-weights"
            (path / "17/config.resolved.json").unlink()
            (path / "17").rmdir()
            with self.assertRaisesRegex(ValueError, "exactly the five"):
                exporter.discover_runs(root)
            save(path / "duplicate/config.resolved.json", dict(experiment="hanoi-control-v89-ablations",
                                                                job=dict(name="without-actor-weights", training_seed=42)))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                exporter.discover_runs(root)

    def test_transfer_ablations_require_same_bc_bundle_and_ppo_settings(self) -> None:
        jobs = {}
        for seed in SEEDS:
            for policy in exporter.POLICIES:
                jobs[policy, seed] = dict(training_seed=seed, algorithm=dict(
                    learning_rate=.0001, pretrained_policy="same-bundle" if policy not in ("baseline", "baseline-4m") else None))
        exporter.validate_matched_jobs(jobs)
        different = deepcopy(jobs)
        different["without-actor-weights", 17]["algorithm"]["pretrained_policy"] = "different-bundle"
        with self.assertRaisesRegex(ValueError, "Source BC bundle differs"):
            exporter.validate_matched_jobs(different)
        different = deepcopy(jobs)
        different["baseline-4m", 17]["algorithm"]["learning_rate"] = .01
        with self.assertRaisesRegex(ValueError, "Non-ablation job settings differ"):
            exporter.validate_matched_jobs(different)

    def test_history_selection_and_raw_inventory_tampering_is_rejected(self) -> None:
        for tamper, message in (("best", "not validation-selected"), ("evaluation", "history mismatch"),
                                ("index", "Incomplete raw case indexes"), ("case", "Duplicate evaluation case")):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                directory = fake_run(Path(temp))
                if tamper == "best":
                    path = directory / "best/manifest.json"
                    value = exporter.read_json(path)
                    value["evaluation"]["evaluation_return"] = 123
                elif tamper == "evaluation":
                    path = directory / "checkpoints/history/000000106496/evaluation.json"
                    value = exporter.read_json(path)
                    value["evaluation"]["evaluation_return"] = 123
                else:
                    path = directory / "evaluation-data/final/test/manifest.json"
                    value = exporter.read_json(path)
                    if tamper == "index":
                        value["cases"][0]["index"] = 1
                    else:
                        value["cases"][0] = value["cases"][1]
                save(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    exporter.validate_run(directory, "without-actor-weights", 17)

    def test_export_cannot_replace_principal_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not replace Principal"):
            exporter.write_package({}, exporter.ROOT / "paper/results")


if __name__ == "__main__":
    unittest.main()
