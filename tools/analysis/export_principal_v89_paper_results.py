#!/usr/bin/env python3
"""Project the completed V89 Principal runs into the five live paper inputs.

The exporter reads the copied Windows artifacts by their hashes in each seed's
plan, so paths embedded in those plans need not exist on the current machine.
Learning curves come from the training histories; physical metrics come from
the final test evaluation of each validation-selected checkpoint. The measured
offline materialization report supplies the shared preparation cost. Every
input is validated before replacing publication files. No simulation or model
inference is performed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))

from paper.tools.results_projection import (  # noqa: E402
    CONTROL_COMPONENTS, FORMAT_VERSION, PRIMARY_COMPONENTS, SEEDS,
    forecast_projection, scenario_metrics,
)
from tools.analysis.export_evaluation_results import (  # noqa: E402
    control_rows, forecast_rows, forecast_summary,
)

PAPER_FILES = (
    "package_metadata.csv", "computation_costs.csv",
    "learning-plot-data.json", "control-data.json", "forecast-plot-data.json",
)
COST_COLUMNS = (
    "cost_id", "stage", "approach", "cost_scope", "training_seed", "fold_id",
    "hardware_id", "execution_device", "worker_count", "parallelism_description",
    "wall_time_seconds", "simulator_steps", "online_interactions", "notes",
)
METADATA_COLUMNS = (
    "dataset_id", "schema_version", "data_kind", "created_at", "producer",
    "code_revision", "experiment_id", "notes",
)
COMPONENT_KEYS = {
    ("chlorine_shortfall_fraction", "main_normal"): "chlorine_shortfall_fraction",
    ("normal_chlorine_excess_fraction", "main_normal"): "normal_chlorine_excess_fraction",
    ("arsenic_burden", "main"): "arsenic_burden_main",
    ("arsenic_burden", "tail"): "arsenic_burden_tail",
    ("arsenic_burden", "total"): "arsenic_burden",
    ("chlorine_ceiling_violation_fraction", "total"): "chlorine_ceiling_violation_fraction",
    ("chlorine_use", "main_normal"): "chlorine_use_main_normal",
    ("chlorine_use", "main_event"): "chlorine_use_main_event",
    ("chlorine_use", "tail_event"): "chlorine_use_tail_event",
    ("chlorine_use", "total"): "chlorine_use",
}
assert set(COMPONENT_KEYS) == set(CONTROL_COMPONENTS) - {("arsenic_clearance_lead", "total")}

from tools.analysis.paper_arsenic_clearance import clearance_projection, METRIC  # noqa: E402


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(directory: Path, *names: str) -> None:
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty artifact: {path}")


def finite_seconds(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid runtime for {label}: {value}")
    return number


def mpc_generation_costs(plan: dict, record: dict, seed: int, hardware_id: str) -> list[dict]:
    """Export direct generation timers; fold rows are included in training time."""
    components = record.get("components", [])
    if [part["stage"] for part in components] != [
        "mpc-teacher-training", "mpc-teacher-validation",
    ]:
        raise ValueError(f"Missing direct MPC generation timings for seed {seed}")
    training, validation = components
    folds = training.get("components", [])
    if (len(folds) != 5 or any(fold["stage"] != "mpc-teacher-fold" for fold in folds)
            or [fold["settings"].get("fold_index") for fold in folds] != list(range(5))):
        raise ValueError(f"Missing or duplicate MPC fold timings for seed {seed}")

    def validate(part: dict, parent: dict | None = None) -> None:
        seconds = finite_seconds(part["elapsed_seconds"], f"{seed}/{part['stage']}")
        start = datetime.fromisoformat(part["started_at"])
        finish = datetime.fromisoformat(part["finished_at"])
        if part["status"] != "completed" or seconds <= 0 or start >= finish:
            raise ValueError(f"Incomplete MPC timing for seed {seed}: {part['stage']}")
        if parent is not None and not (
            datetime.fromisoformat(parent["started_at"]) <= start < finish
            <= datetime.fromisoformat(parent["finished_at"])
        ):
            raise ValueError(f"MPC timing outside parent stage for seed {seed}")
        children = part.get("components", [])
        if sum(child["elapsed_seconds"] for child in children) > seconds + 1e-6:
            raise ValueError(f"MPC component times exceed parent runtime for seed {seed}")
        if any(datetime.fromisoformat(left["finished_at"])
               > datetime.fromisoformat(right["started_at"])
               for left, right in zip(children, children[1:])):
            raise ValueError(f"Overlapping MPC component timings for seed {seed}")

    validate(record)
    rows = []
    for label, key, part in (("training", "teacher", training),
                             ("validation", "validation_teacher", validation)):
        validate(part, record)
        output_hash = Path(part["settings"]["output"].replace("\\", "/")).name
        if output_hash != plan[key]["configuration_hash"]:
            raise ValueError(f"MPC {label} timing dataset mismatch for seed {seed}")
        rows.append(cost_row(
            f"mpc_teacher_{label}_generation", "pretrained", "per_training_seed",
            f"mpc-teachers-{label}-{seed}", part["elapsed_seconds"],
            "auto (resolved device not recorded)", part["settings"]["workers"],
            f"BC {label} demonstration generation", hardware_id, seed=seed,
            notes="Direct wall-clock measurement of teacher dataset generation. "
                  + ("Includes fold replay checks and dataset merging. " if label == "training" else "")
                  + f"Started at {part['started_at']}.",
        ))
    for fold in folds:
        validate(fold, training)
        index = fold["settings"]["fold_index"]
        rows.append(cost_row(
            "mpc_teacher_fold_generation", "pretrained", "per_fold",
            f"mpc-teachers-{seed}-fold-{index}", fold["elapsed_seconds"],
            "auto (resolved device not recorded)", fold["settings"]["workers"],
            "Fold teacher generation and replay checks", hardware_id, seed=seed, fold=index,
            notes="Direct wall-clock measurement; included in mpc_teacher_training_generation, "
                  "excluded from additive totals.",
        ))
    return rows


def run_paths(root: Path, plan: dict, approach: str) -> Path:
    key = "baseline" if approach == "baseline" else "mpc_teacher_bc"
    kind = "baseline" if approach == "baseline" else "mpc-teacher-bc"
    job_hash = plan["control"][key]["job_hash"]
    return root / "weights/control/principal/v89" / kind / "hanoi-principal-v89" / kind / job_hash


def forecast_path(root: Path, job: dict) -> Path:
    return (root / "weights/forecasting/encoder-decoder-lstm-cross-fit"
            / "forecasting-encoder-decoder-lstm-cross-fit" / job["name"] / job["job_hash"])


def selected_case_burdens(root: Path, case_key: tuple[str, str], reference: float) -> dict:
    """Project one matched case across seeds and recover its fixed schedule index."""
    scenario_id, schedule_hash = case_key
    schedule_indexes = set()
    burdens = []
    for seed in SEEDS:
        plan = read_json(root / "pipelines/hanoi-principal-v89" / f"seed-{seed}" / "plan.json")
        values = {}
        for approach in ("baseline", "pretrained"):
            directory = run_paths(root, plan, approach)
            metrics = read_json(directory / "metrics.json")
            cases = [case for case in metrics["test"]["cases"]
                     if (case["scenario_id"], case["schedule_hash"]) == case_key]
            if len(cases) != 1 or not math.isclose(
                cases[0]["zero_injection_arsenic_burden"], reference,
                rel_tol=1e-9, abs_tol=1e-9,
            ):
                raise ValueError(f"Missing or mismatched selected case for {approach}, seed {seed}")
            values[approach] = cases[0]["values"]["arsenic_burden"]
            manifest = read_json(directory / "evaluation-data/final/test/manifest.json")
            raw_cases = [case for case in manifest["cases"]
                         if (case["scenario_id"], case["schedule_hash"]) == case_key]
            if len(raw_cases) != 1:
                raise ValueError(f"Missing selected-case trajectory identity: {directory}")
            scenarios = read_json(directory / "best/manifest.json")["job"]["dataset"]["test"]["scenarios"]
            schedule_index, scenario_index = divmod(raw_cases[0]["index"], len(scenarios))
            if scenarios[scenario_index] != scenario_id:
                raise ValueError(f"Selected-case enumeration mismatch: {directory}")
            schedule_indexes.add(schedule_index)
        burdens.append(dict(seed=str(seed), reference=reference, **values))
    if len(schedule_indexes) != 1:
        raise ValueError("Selected case has different schedule indexes across runs")
    schedule_index = schedule_indexes.pop()
    return dict(scenario_id=scenario_id, schedule_index=schedule_index,
                schedule_id=f"arsenic-{schedule_index}", burden_by_seed=burdens)


def control_figure_data(root: Path) -> dict:
    """Read matched final test cases, averaging each burden across five seeds."""
    grouped = {}
    selected_key = None
    selected_rows = []
    for seed in SEEDS:
        plan = read_json(root / "pipelines/hanoi-principal-v89" / f"seed-{seed}" / "plan.json")
        for approach in ("baseline", "pretrained"):
            directory = run_paths(root, plan, approach)
            metrics = read_json(directory / "metrics.json")
            published = read_json(directory / "best/manifest.json")
            selected = max(sorted(metrics["history"], key=lambda row: row["timesteps"]),
                           key=lambda row: row["evaluation"]["evaluation_return"])
            key = "baseline" if approach == "baseline" else "mpc_teacher_bc"
            if (read_json(directory / "status.json")["status"] != "completed"
                    or published["job_hash"] != plan["control"][key]["job_hash"]
                    or published["job"]["training_seed"] != seed
                    or metrics["final_timesteps"] < published["job"]["total_timesteps"]
                    or metrics["final_validation"] != selected["evaluation"]
                    or published["evaluation"] != selected["evaluation"]):
                raise ValueError(f"Invalid validation-selected test policy: {directory}")
            manifest = read_json(directory / "evaluation-data/final/test/manifest.json")
            scenarios = published["job"]["dataset"]["test"]["scenarios"]
            identities = {}
            for case in manifest["cases"]:
                schedule, scenario = divmod(case["index"], len(scenarios))
                identity = (case["scenario_id"], case["schedule_hash"])
                if scenarios[scenario] != case["scenario_id"] or identity in identities:
                    raise ValueError(f"Invalid test-case identity: {directory}")
                identities[identity] = schedule
                if case["scenario_id"] == "scenario-90" and schedule == 8:
                    if selected_key is not None and identity != selected_key:
                        raise ValueError("Selected case differs across policies")
                    selected_key = identity
            cases = metrics["test"]["cases"]
            if (len(cases) != 100 or len(identities) != 100
                    or {(c["scenario_id"], c["schedule_hash"]) for c in cases} != set(identities)):
                raise ValueError(f"Incomplete final test cases: {directory}")
            for case in cases:
                identity = (case["scenario_id"], case["schedule_hash"])
                record = grouped.setdefault(identity, dict(
                    scenario_id=identity[0], schedule_id=f"arsenic-{identities[identity]}",
                    reference=case["zero_injection_arsenic_burden"],
                    baseline={}, pretrained={},
                ))
                value = case["values"]["arsenic_burden"]
                if (seed in record[approach] or not math.isfinite(value) or value < 0
                        or record["schedule_id"] != f"arsenic-{identities[identity]}"
                        or not math.isclose(record["reference"], case["zero_injection_arsenic_burden"],
                                            rel_tol=1e-9, abs_tol=1e-9)):
                    raise ValueError(f"Mismatched test case: {identity}")
                record[approach][seed] = value
                if seed == SEEDS[0] and identity == selected_key:
                    selected_rows.extend(dict(approach=approach, metric=metric,
                                              component=component, value=case["values"][metric])
                                         for metric, component in PRIMARY_COMPONENTS if metric != METRIC)
    if len(grouped) != 100 or selected_key is None or len(selected_rows) != 10:
        raise ValueError("Incomplete matched-case projection")
    points = []
    for identity, record in sorted(grouped.items()):
        if any(set(record[approach]) != set(SEEDS) for approach in ("baseline", "pretrained")):
            raise ValueError(f"Missing seed in matched case: {identity}")
        points.append(dict(scenario_id=record["scenario_id"], schedule_id=record["schedule_id"],
                           **{approach: sum(record[approach][s] for s in SEEDS) / len(SEEDS)
                              for approach in ("baseline", "pretrained")}))
    reference = grouped[selected_key]["reference"]
    selected_rows.append(dict(approach="zero_injection", metric="arsenic_burden",
                              component="total", value=reference))
    return dict(case_burdens=points, selected=dict(
        case_ids=[f"{selected_key[0]}:{selected_key[1]}"],
        reason="Selected illustration of a larger pretrained arsenic-burden reduction; four source-injection events with two overlapping in time; seed 17",
        rows=selected_rows, **selected_case_burdens(root, selected_key, reference),
    ))


def cost_row(stage: str, approach: str, scope: str, identity: str, seconds: object,
             device: str, workers: object, description: str, hardware_id: str,
             *, seed: int | None = None, fold: int | None = None,
             interactions: int | None = None, simulator_steps: int | None = None,
             notes: str = "") -> dict:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError(f"Invalid worker count for {identity}: {workers}")
    return dict(zip(COST_COLUMNS, (
        identity, stage, approach, scope, seed, fold, hardware_id, device,
        worker_count, description, finite_seconds(seconds, identity), simulator_steps,
        interactions, notes,
    )))


def validate_control_run(directory: Path, metrics: dict, job_hash: str) -> int:
    require_files(directory, "metrics.json", "status.json", "best/model.zip",
                  "best/vec_normalize.pkl", "best/manifest.json")
    status = read_json(directory / "status.json")
    if status.get("status") != "completed":
        raise ValueError(f"Control run did not complete: {directory}")
    selected = max(sorted(metrics["history"], key=lambda row: row["timesteps"]),
                   key=lambda row: row["evaluation"]["evaluation_return"])
    score = selected["evaluation"]["evaluation_return"]
    published = read_json(directory / "best/manifest.json")
    if (published.get("job_hash") != job_hash
            or published.get("evaluation") != selected["evaluation"]
            or metrics.get("final_validation") != selected["evaluation"]
            or not math.isclose(metrics["best_mean_validation_reward"], score,
                                rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError(f"Final policy is not the validation-selected checkpoint: {directory}")
    history = metrics["history"]
    for item in history:
        snapshot = directory / "checkpoints/history" / f"{item['timesteps']:012d}"
        require_files(snapshot, "model.zip", "vec_normalize.pkl", "manifest.json",
                      "evaluation.json", "trainer_state.json")
        manifest = read_json(snapshot / "manifest.json")
        if (manifest.get("job_hash") != job_hash or
                manifest.get("timesteps") != item["timesteps"]) or sorted(
            manifest.get("nominal_thresholds", [])
        ) != sorted(item["nominal_thresholds"]):
            raise ValueError(f"Checkpoint identity mismatch: {snapshot}")
    final = directory / "evaluation-data/final"
    for split, count in (("validation", 50), ("test", 100)):
        path = final / split
        manifest = read_json(path / "manifest.json")
        if manifest.get("case_count") != count or len(manifest.get("cases", [])) != count:
            raise ValueError(f"Incomplete final raw evaluation: {path}")
        for case in manifest["cases"]:
            require_files(path, case["file"])
    return selected["timesteps"]


def timed_stages(root: Path, seed: int) -> dict[str, list[dict]]:
    directory = root / "pipelines/hanoi-principal-v89" / f"seed-{seed}"
    status = read_json(directory / "status.json")
    if status.get("status") != "completed" or status.get("seed") != seed:
        raise ValueError(f"Incomplete pipeline for seed {seed}")
    records = read_json(directory / "stage-timings.json")["records"]
    stages = {}
    for record in records:
        stages.setdefault(record["stage"], []).append(record)
    # Old aggregate timers, interrupted attempts, and resume-only checks remain
    # in the log. Use the latest completed generation with direct components.
    measured_mpc = [record for record in stages.get("mpc", [])
                    if record["status"] == "completed" and record.get("components")]
    if not measured_mpc:
        raise ValueError(f"Missing direct MPC generation timings for seed {seed}")
    stages["mpc"] = [max(measured_mpc, key=lambda record: record["started_at"])]
    selected_records = [record for rows in stages.values() for record in rows]
    expected = {"forecasting", "mpc", "behavior-cloning", "baseline",
                "baseline-evaluation", "fine-tuning", "fine-tuning-evaluation"}
    control = {"baseline", "baseline-evaluation", "fine-tuning", "fine-tuning-evaluation"}
    if (set(stages) != expected
            or any(len(stages[name]) != 1 for name in expected - control)
            or len({len(stages[name]) for name in control}) != 1
            or len(stages["baseline"]) not in (1, 2)
            or any(r["status"] != "completed" for r in selected_records)):
        raise ValueError(f"Missing, duplicate, or incomplete stage timing for seed {seed}")
    for record in selected_records:
        finite_seconds(record["elapsed_seconds"], f"{seed}/{record['stage']}")
    if any(rows != sorted(rows, key=lambda row: row["started_at"])
           for rows in stages.values()):
        raise ValueError(f"Non-chronological stage timing for seed {seed}")
    return stages


def materialization_report(root: Path) -> dict:
    directory = next((path for path in (root / "materialization", root / "Materialization")
                      if path.is_dir()), None)
    reports = []
    if directory is not None:
        for path in directory.glob("*.json"):
            report = read_json(path)
            if report.get("kind") == "hanoi-offline-trajectory-materialization":
                reports.append((path, report))
    if len(reports) != 1:
        raise ValueError(f"Expected one offline materialization report under {root}: "
                         f"found {len(reports)}")
    path, report = reports[0]
    if (report.get("schema_version") != 1 or report.get("status") != "completed"
            or report.get("requested_split") != "all" or not report.get("fresh_generation")
            or {part.get("split") for part in report.get("splits", [])}
            != {"train", "validation", "test"}):
        raise ValueError(f"Incomplete or non-fresh materialization report: {path}")
    if report.get("generated_trajectories") != 2_500:
        raise ValueError(f"Unexpected generated trajectory count: {path}")
    steps = report.get("generated_simulator_steps")
    if steps != 2_500 * 1_200:
        raise ValueError(f"Unexpected generated simulator steps: {path}")
    finite_seconds(report["elapsed_seconds"], str(path))
    if report["elapsed_seconds"] <= 0 or int(report["workers"]) < 1:
        raise ValueError(f"Invalid materialization runtime or worker count: {path}")
    return report


def build_package(args: argparse.Namespace) -> dict[str, object]:
    root = args.results_root
    schema = read_json(ROOT / "paper/schema/datapackage.json")
    if schema.get("version") != "1.1.0":
        raise ValueError("Unexpected paper result schema version")
    hardware_id = "principal-training-pc"
    materialization = materialization_report(root)
    hydraulic_profile_seconds = (
        finite_seconds(materialization["hydraulics_seconds"], "hydraulic profile generation")
        + finite_seconds(materialization["hydraulic_cache_inventory_seconds"], "hydraulic profile inventory")
    )
    trajectory_seconds = finite_seconds(
        materialization["elapsed_seconds"] - hydraulic_profile_seconds,
        "trajectory materialization excluding hydraulic profile preparation",
    )
    costs = [cost_row(
        "offline_trajectory_generation", "shared", "reusable_once",
        "offline-trajectories", trajectory_seconds,
        materialization["execution_device"], materialization["workers"],
        materialization["parallelism"], hardware_id,
        simulator_steps=materialization["generated_simulator_steps"],
        notes="Fresh trajectory materialization excluding hydraulic profile preparation; includes remaining run preparation and validation",
    ), cost_row(
        "hydraulic_profile_generation", "shared", "reusable_once",
        "hydraulic-profiles", hydraulic_profile_seconds,
        materialization["execution_device"], materialization["workers"],
        materialization["parallelism"], hardware_id,
        notes="Shared hydraulic profile generation including hydraulic cache inventory and profile validation",
    )]
    forecast_statistics = []
    learning = []
    for seed in SEEDS:
        plan_dir = root / "pipelines/hanoi-principal-v89" / f"seed-{seed}"
        plan = read_json(plan_dir / "plan.json")
        if plan.get("seed") != seed or plan.get("reward_version") != "v89":
            raise ValueError(f"Wrong Principal plan identity for seed {seed}")
        stages = timed_stages(root, seed)

        jobs = plan["forecasting"]["jobs"]
        if len(jobs) != 5 or {j["name"] for j in jobs} != {
            f"encoder-decoder-lstm-fold-{fold}" for fold in range(5)
        }:
            raise ValueError(f"Incomplete forecast job inventory for seed {seed}")
        for job in jobs:
            fold = int(job["name"].rsplit("-", 1)[1])
            path = forecast_path(root, job)
            require_files(path, "metrics.json", "status.json", "best/model.pt",
                          "checkpoints/normalization.json")
            if read_json(path / "status.json").get("status") != "completed":
                raise ValueError(f"Incomplete forecast job: {path}")
            metrics = read_json(path / "metrics.json")
            if (metrics.get("training_seed"), metrics.get("fold_id")) != (seed, fold):
                raise ValueError(f"Forecast seed/fold mismatch: {path}")
            forecast_statistics.extend(forecast_rows(path))
            runtime = metrics["runtime"]
            workers = runtime["execution_workers"]
            train_seconds = runtime["training_seconds"] + runtime["checkpoint_validation_seconds"]
            evaluation_seconds = sum(runtime["final_evaluation_seconds"][split]
                                     for split in ("train", "evaluation", "test"))
            costs.append(cost_row(
                "forecaster_training", "pretrained", "per_fold",
                f"forecaster-{seed}-fold-{fold}", train_seconds, runtime["device"],
                workers, "Model fitting and checkpoint validation", hardware_id,
                seed=seed, fold=fold,
            ))
            costs.append(cost_row(
                "forecast_evaluation", "pretrained", "per_fold",
                f"forecast-evaluation-{seed}-fold-{fold}", evaluation_seconds,
                runtime["device"], workers, "Final train, validation, and test evaluation",
                hardware_id, seed=seed, fold=fold,
            ))

        for label, key, expected_count in (("training", "teacher", 48_000),
                                           ("validation", "validation_teacher", 6_000)):
            ref = plan[key]
            path = root / "datasets/mpc-teacher-cem" / ref["configuration_hash"]
            require_files(path, "dataset.npz", "manifest.json")
            manifest = read_json(path / "manifest.json")
            if (manifest.get("teacher_configuration_hash") != ref["configuration_hash"]
                    or manifest["arrays"]["first_action"]["shape"][0] != expected_count):
                raise ValueError(f"Invalid {label} teacher dataset for seed {seed}")
        bc = root / "weights/control/mpc-teacher-bc" / plan["behavior_cloning"]["configuration_hash"]
        require_files(bc, "model.zip", "vec_normalize.pkl", "manifest.json", "metrics.json")
        bc_manifest = read_json(bc / "manifest.json")
        if (bc_manifest.get("kind") != "recurrent-ppo-behavior-cloning" or
                bc_manifest["split"]["training_labels"] != 48_000 or
                bc_manifest["split"]["validation_labels"] != 6_000 or
                bc_manifest["split"]["training_trajectories"] != 2_000 or
                Path(bc_manifest["dataset"].replace("\\", "/")).name !=
                    plan["teacher"]["configuration_hash"] or
                Path(bc_manifest["validation_dataset"].replace("\\", "/")).name !=
                    plan["validation_teacher"]["configuration_hash"]):
            raise ValueError(f"Invalid BC artifact for seed {seed}")
        bc_runtime = read_json(bc / "metrics.json")["runtime"]
        costs.extend(mpc_generation_costs(plan, stages["mpc"][0], seed, hardware_id))
        costs.append(cost_row(
            "behavior_cloning", "pretrained", "per_training_seed",
            f"behavior-cloning-{seed}", stages["behavior-cloning"][0]["elapsed_seconds"],
            bc_runtime["device"], stages["behavior-cloning"][0]["settings"]["workers"],
            "BC stage including preparation and validation", hardware_id, seed=seed,
        ))

        for approach, train_stage, eval_stage in (
            ("baseline", "baseline", "baseline-evaluation"),
            ("pretrained", "fine-tuning", "fine-tuning-evaluation"),
        ):
            directory = run_paths(root, plan, approach)
            metrics = read_json(directory / "metrics.json")
            control_key = "baseline" if approach == "baseline" else "mpc_teacher_bc"
            selected_timesteps = validate_control_run(
                directory, metrics, plan["control"][control_key]["job_hash"]
            )
            curves, _ = control_rows(approach, seed, directory, returns_only=True)
            learning.extend(curves)
            runtime = metrics["runtime"]
            costs.append(cost_row(
                "ppo_training", approach, "per_training_seed",
                f"ppo-{approach}-{seed}", stages[train_stage][0]["elapsed_seconds"],
                runtime["device"], runtime["training_environments"],
                "PPO training with parallel environments", hardware_id,
                seed=seed, interactions=int(metrics["final_timesteps"]),
                simulator_steps=int(metrics["final_timesteps"]),
            ))
            costs.append(cost_row(
                "control_evaluation", approach, "per_evaluation",
                f"control-evaluation-{approach}-{seed}",
                stages[eval_stage][0]["elapsed_seconds"], runtime["device"],
                runtime["evaluation_workers"],
                "Checkpoint validation and final validation/test evaluation",
                hardware_id, seed=seed,
            ))

    # Returns and selected-checkpoint provenance remain those of the completed
    # experiments. All published physical values are rescored from raw arrays.
    forecast_summary(forecast_statistics)
    forecast = forecast_projection(forecast_statistics, "experimental")
    control = add_clearance(dict(format_version=FORMAT_VERSION, data_kind="experimental",
                                **control_figure_data(root)), root)
    if len(control["rows"]) != 2 * len(SEEDS) * 10 * len(CONTROL_COMPONENTS):
        raise ValueError("Incomplete scenario projection")
    if len(learning) != 2 * len(SEEDS) * 21:
        raise ValueError("Incomplete learning-curve projection")

    notes = "Offline generation measured; final policy tests use validation-selected checkpoints."
    metadata = dict(zip(METADATA_COLUMNS, (
        "hanoi-principal-v89-five-seed", schema["version"], "experimental",
        datetime.now(timezone.utc).isoformat(),
        "tools/analysis/export_principal_v89_paper_results.py", "",
        "hanoi-principal-v89", notes,
    )))
    return {
        "package_metadata.csv": ([metadata], METADATA_COLUMNS),
        "computation_costs.csv": (costs, COST_COLUMNS),
        "learning-plot-data.json": dict(data_kind="experimental", rows=sorted(
            learning, key=lambda r: (r["approach"], r["training_seed"], r["nominal_interactions"]))),
        "control-data.json": control,
        "forecast-plot-data.json": forecast,
    }


def add_clearance(control: dict, root: Path) -> dict:
    rescored = clearance_projection(root, run_paths)
    control["rows"] = rescored["rows"]
    control["reference_values"] = rescored["reference_values"]
    control["case_burdens"] = rescored["case_burdens"]
    control["clearance_support"] = rescored["support"]
    control["selected"]["clearance_by_seed"] = rescored["selected_cases"]
    control["selected"]["clearance_events_by_seed"] = rescored["selected_events"]
    control["selected"]["burden_by_seed"] = rescored["selected_burdens"]
    control["selected"]["rows"] = rescored["selected_rows"]
    identity = list(control["selected"]["case_ids"])
    expected_identity = ':'.join(rescored["selected_identity"])
    if identity != [expected_identity]:
        raise ValueError("Rescored illustrative case differs from selected case")
    return control


def write_package(package: dict[str, object], output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".principal-results-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        for name in PAPER_FILES:
            content = package[name]
            path = staging / name
            if name.endswith(".csv"):
                rows, columns = content
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                path.write_text(json.dumps(content, separators=(",", ":"),
                                           allow_nan=False) + "\n", encoding="utf-8")
            if path.stat().st_size > 500_000:
                raise ValueError(f"Publication input exceeds 500 kB: {path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in PAPER_FILES:
            os.replace(staging / name, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--control-only", action="store_true", help="Reproject final control metrics, preserving other publication files")
    parser.add_argument("--check-only", action="store_true", help="Validate without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.control_only:
        # Reuse unchanged live publication measurements and provenance.
        verified = control_figure_data(args.results_root)
        control = read_json(args.output_dir / "control-data.json")
        if verified["selected"]["case_ids"] != control["selected"]["case_ids"]:
            raise ValueError("Published selected-case identity does not match final test artifacts")
        control = add_clearance(control, args.results_root)
        if not args.check_only:
            path = args.output_dir / "control-data.json"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                json.dump(control, stream, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                staged = stream.name
            os.replace(staged, path)
            case_path = args.output_dir / "case-trajectory-data.json"
            if case_path.is_file():
                case = read_json(case_path)
                for controller in case["controllers"].values():
                    controller.pop("sustained_chlorine_excess_mg_h_l", None)
                case_path.write_text(json.dumps(case, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        print(f"{'Validated' if args.check_only else 'Wrote'} corrected final-test physical-metric projection; other publication inputs preserved")
        print(json.dumps(control["clearance_support"], indent=2))
        return
    package = build_package(args)
    if not args.check_only:
        write_package(package, args.output_dir)
    action = "Validated" if args.check_only else "Wrote"
    print(f"{action} V89 Principal publication inputs from {args.results_root}")
    for name in PAPER_FILES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
