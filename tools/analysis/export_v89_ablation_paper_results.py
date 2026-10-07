#!/usr/bin/env python3
"""Export four completed V89 ablations and their matched Principal comparators.

No simulation or training is performed. Stored validation returns select
checkpoints. The paper-validation export reads the complete archived history,
including the original checkpoints of the baseline continued to four million.
The separate physical export requires tests of the selected policy bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))

import numpy as np  # noqa: E402

from packages.trainer.evaluation.protocol import (  # noqa: E402
    PROTOCOL_VERSION, scenario_average, scoring_windows)
from paper.tools.results_projection import (CONTROL_COMPONENTS,  # noqa: E402
                                            SEEDS, scenario_metrics)
from tools.analysis.export_evaluation_results import \
    validate_inventory  # noqa: E402
from tools.analysis.paper_arsenic_clearance import (METRIC,  # noqa: E402
                                                    clusters, component_value,
                                                    load_case,
                                                    persistent_clearance,
                                                    physical_projection)

VARIANTS = {
    "baseline-4m": (),
    "without-actor-weights": ("action_std", "observation_normalization"),
    "offline-normalization": ("observation_normalization",),
    "adjusted-action-std": ("action_std",),
}
POLICIES = ("baseline", "pretrained", *VARIANTS)
PAPER_VARIANTS = ("baseline-4m", "without-actor-weights")
LONG_BASELINE_BUDGET = 4_000_000
CONTRASTS = (("pretrained", "baseline"),) + tuple(
    (variant, comparator) for variant in VARIANTS for comparator in ("baseline", "pretrained")
)
PHYSICAL_COMPONENTS = tuple(pair for pair in CONTROL_COMPONENTS if pair[0] != METRIC)
UNITS = {
    "chlorine_shortfall_fraction": "fraction", "normal_chlorine_excess_fraction": "fraction",
    "arsenic_burden": "mg h/L (sum over monitored nodes)",
    "chlorine_ceiling_violation_fraction": "fraction", "chlorine_use": "mg",
    METRIC: "h",
}
PAPER_DISPLAY = {
    metric: dict(unit="%", scale=100) if unit == "fraction" else
    dict(unit="t", scale=1e-9) if metric == "chlorine_use" else dict(unit=unit, scale=1)
    for metric, unit in UNITS.items()
}


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"Invalid {label}: {value}")
    return number


def discover_runs(root: Path, policies: tuple = POLICIES) -> dict:
    """Discover by config identity, rejecting duplicates and missing seeds."""
    runs = {}
    for variant in VARIANTS:
        if variant not in policies:
            continue
        base = root / "weights/control/ablations/v89" / variant / "hanoi-control-v89-ablations" / variant
        for directory in sorted(base.iterdir() if base.is_dir() else []):
            if not directory.is_dir():
                continue
            config = read_json(directory / "config.resolved.json")
            seed = config["job"]["training_seed"]
            if (seed not in SEEDS or (variant, seed) in runs
                    or config.get("experiment") != "hanoi-control-v89-ablations"
                    or config["job"].get("name") != variant):
                raise ValueError(f"Duplicate or incompatible ablation identity: {directory}")
            runs[variant, seed] = directory
    for seed in SEEDS:
        plan = read_json(root / "pipelines/hanoi-principal-v89" / f"seed-{seed}" / "plan.json")
        if plan.get("seed") != seed or plan.get("reward_version") != "v89":
            raise ValueError(f"Wrong Principal plan: seed {seed}")
        for policy, key, kind in (("baseline", "baseline", "baseline"),
                                  ("pretrained", "mpc_teacher_bc", "mpc-teacher-bc")):
            if policy not in policies:
                continue
            runs[policy, seed] = (root / "weights/control/principal/v89" / kind
                                  / "hanoi-principal-v89" / kind / plan["control"][key]["job_hash"])
    if set(runs) != {(policy, seed) for policy in policies for seed in SEEDS}:
        raise ValueError("Every ablation requires exactly the five specified training seeds")
    return runs


def validate_run(directory: Path, policy: str, seed: int, *, validation_only: bool = False) -> tuple[dict | None, list, dict]:
    """Verify selection against history and byte-identical archived policy bundles.

    Legacy final/test manifests have no policy hash. Their policy association is
    the source runner's evaluate_best_model convention, recorded in provenance;
    do not claim that the raw manifest independently certifies that association.
    """
    config = read_json(directory / "config.resolved.json")
    job, job_hash = config["job"], config["job_hash"]
    metrics = read_json(directory / "metrics.json")
    published = read_json(directory / "best/manifest.json")
    status = read_json(directory / "status.json")
    configured_budget = 3_400_000 if policy == "baseline-4m" else 1_000_000
    budget = configured_budget
    state = None
    if policy == "baseline-4m" and (directory / "checkpoints/latest.json").is_file():
        state = read_json(directory / "checkpoints/latest.json")
        budget = state.get("target_timesteps", configured_budget)
        if (budget not in (configured_budget, LONG_BASELINE_BUDGET)
                or state.get("job_hash") != job_hash
                or state.get("completed_timesteps") != metrics.get("final_timesteps")
                or state.get("evaluation_protocol_version") != metrics.get("protocol_version")
                or state.get("history") != metrics.get("history")):
            raise ValueError(f"Incompatible baseline continuation state: {directory}")
    if (config.get("schema_version") != 2
            or not re.fullmatch(r"[0-9a-f]{12}", job_hash) or directory.name != job_hash
            or job.get("training_seed") != seed or job.get("total_timesteps") != configured_budget
            or job.get("kind") != "control" or not job.get("evaluate_test")
            or not job["reward"]["factory"].endswith(":V89Reward")
            or not job["observation"]["factory"].endswith(":V89Observation")
            or status.get("status") != "completed" or metrics.get("status") != "complete"
            or metrics.get("protocol_version") not in (2, PROTOCOL_VERSION)
            or (directory / "evaluation-incomplete.json").exists()
            or published.get("job_hash") != job_hash or published.get("job") != job
            or published.get("schema_version") != 1 or published.get("kind") != "recurrent-ppo"
            or published.get("node_ids") != [str(node) for node in range(2, 33)]):
        raise ValueError(f"Incomplete or incompatible control run: {directory}")
    algorithm = job["algorithm"]
    rollout = algorithm["n_envs"] * algorithm["n_steps"]
    expected_components = VARIANTS.get(policy, () if policy == "baseline" else
                                      ("actor", "action_std", "observation_normalization"))
    # Older Principal configs omit this field: their runner transfers the full
    # actor/std/normalization bundle by default. Ablations declare it explicitly.
    actual_components = algorithm.get("pretrained_components", expected_components if policy == "pretrained" else ())
    if ((policy in ("baseline", "baseline-4m") and algorithm.get("pretrained_policy") is not None)
            or (policy not in ("baseline", "baseline-4m") and
                (not algorithm.get("pretrained_policy") or set(actual_components) != set(expected_components)))):
        raise ValueError(f"Wrong transferred components: {directory}")
    recorded_history = sorted(metrics["history"], key=lambda row: row["timesteps"])
    if not recorded_history or len({row["timesteps"] for row in recorded_history}) != len(recorded_history):
        raise ValueError(f"Empty or duplicate checkpoint history: {directory}")
    published_selected = max(recorded_history, key=lambda row: finite(row["evaluation"]["evaluation_return"], "validation return"))
    if (published.get("evaluation") != published_selected["evaluation"]
            or (metrics.get("final_validation") is not None and metrics["final_validation"] != published_selected["evaluation"])
            or not math.isclose(finite(metrics["best_mean_validation_reward"], "best validation return"),
                                published_selected["evaluation"]["evaluation_return"], rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError(f"Best bundle is not validation-selected: {directory}")
    history = recorded_history
    if budget == LONG_BASELINE_BUDGET:
        history = [read_json(snapshot / "evaluation.json")
                   for snapshot in sorted((directory / "checkpoints/history").iterdir()) if snapshot.is_dir()]
        cutoff = state.get("legacy_evaluation_cutoff_timesteps")
        expected = [row for row in history if cutoff is None or row["timesteps"] > cutoff]
        if (expected != recorded_history
                or (cutoff is not None and cutoff != math.ceil(configured_budget / rollout) * rollout)):
            raise ValueError(f"Archived continuation history disagrees with current metrics: {directory}")
    selected = max(history, key=lambda row: finite(row["evaluation"]["evaluation_return"], "validation return"))
    curves = []
    inventory = None
    history_hashes = []
    for checkpoint in history:
        snapshot = directory / "checkpoints/history" / f"{checkpoint['timesteps']:012d}"
        manifest = read_json(snapshot / "manifest.json")
        evaluation = read_json(snapshot / "evaluation.json")
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "recurrent-ppo-training-checkpoint"
                or manifest.get("job_hash") != job_hash or manifest.get("timesteps") != checkpoint["timesteps"]
                or manifest.get("nominal_thresholds") != checkpoint["nominal_thresholds"]
                or evaluation != checkpoint):
            raise ValueError(f"Checkpoint/evaluation history mismatch: {snapshot}")
        if (checkpoint["evaluation"].get("status") != "complete"
                or checkpoint["evaluation"].get("protocol_version", metrics["protocol_version"]) not in (2, PROTOCOL_VERSION)):
            raise ValueError(f"Incomplete checkpoint evaluation: {snapshot}")
        validate_inventory(checkpoint["evaluation"]["cases"], 5)
        identities = {(row["scenario_id"], row["schedule_hash"]) for row in checkpoint["evaluation"]["cases"]}
        if inventory is not None and identities != inventory:
            raise ValueError(f"Validation cases change between checkpoints: {snapshot}")
        inventory = identities
        for row in checkpoint["evaluation"]["cases"]:
            finite(row["values"]["evaluation_return"], "case validation return")
        recovered_return = scenario_average(checkpoint["evaluation"]["cases"], lambda row: row["values"]["evaluation_return"])
        if not math.isclose(recovered_return, checkpoint["evaluation"]["evaluation_return"], rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError(f"Validation return disagrees with equal-scenario case aggregation: {snapshot}")
        for name in ("model.zip", "vec_normalize.pkl", "trainer_state.json"):
            path = snapshot / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty archived checkpoint artifact: {path}")
        for nominal in checkpoint["nominal_thresholds"]:
            if checkpoint["timesteps"] != math.ceil(nominal / rollout) * rollout:
                raise ValueError(f"Checkpoint does not match nominal rollout budget: {snapshot}")
            curves.append(dict(approach=policy, training_seed=seed, nominal_interactions=nominal,
                               actual_interactions=checkpoint["timesteps"],
                               evaluation_return=checkpoint["evaluation"]["evaluation_return"]))
        history_hashes.append(dict(actual_interactions=checkpoint["timesteps"],
                                   evaluation_sha256=file_hash(snapshot / "evaluation.json")))
    if sorted(row["nominal_interactions"] for row in curves) != list(range(0, budget + 1, 50_000)):
        raise ValueError(f"Incomplete learning-curve thresholds: {directory}")
    actual = metrics["final_timesteps"]
    if actual != math.ceil(budget / rollout) * rollout or history[-1]["timesteps"] != actual:
        raise ValueError(f"Final rollout budget mismatch: {directory}")
    for name in ("model.zip", "vec_normalize.pkl"):
        best_hash = file_hash(directory / "best" / name)
        if best_hash != file_hash(directory / "checkpoints/history" / f"{published_selected['timesteps']:012d}" / name):
            raise ValueError(f"Selected bundle differs from history snapshot: {directory}, {name}")
    provenance = dict(approach=policy, training_seed=seed, run_directory=str(directory), job_hash=job_hash,
                      nominal_budget=budget, configured_budget=configured_budget, actual_interactions=actual,
                      selected_interactions=selected["timesteps"], selected_validation_return=selected["evaluation"]["evaluation_return"],
                      published_selected_interactions=published_selected["timesteps"],
                      transferred_components=list(expected_components),
                      transfer_components_source="resolved configuration" if "pretrained_components" in algorithm else "legacy Principal runner default",
                      selected_bundle_sha256={name: file_hash(directory / "checkpoints/history" / f"{selected['timesteps']:012d}" / name)
                                              for name in ("model.zip", "vec_normalize.pkl")},
                      final_validation_present=metrics.get("final_validation") is not None,
                      metrics_sha256=file_hash(directory / "metrics.json"), config_sha256=file_hash(directory / "config.resolved.json"),
                      history=history_hashes, stored_protocol_version=metrics["protocol_version"], runtime=metrics["runtime"])
    if validation_only:
        provenance["validation_inventory"] = sorted(inventory)
        return None, curves, provenance
    if selected["timesteps"] != published_selected["timesteps"]:
        raise ValueError(f"Final test is not of the full-budget selected policy: {directory}; use --paper-validation for validation-return comparisons")
    test = metrics.get("test")
    if not test or test.get("status") != "complete":
        raise ValueError(f"Missing completed final test: {directory}")
    validate_inventory(test["cases"], 10)
    trace_dir = directory / "evaluation-data/final/test"
    manifest = read_json(trace_dir / "manifest.json")
    if (manifest.get("kind") != "control-evaluation-timeseries" or manifest.get("schema_version") != 1
            or manifest.get("split") != "test" or manifest.get("phase") != "test"
            or manifest.get("duration_seconds") != 25 * 86400
            or manifest.get("hydraulic_timestep_seconds") != 1800
            or manifest.get("node_ids") != published["node_ids"] or manifest.get("case_count") != 100
            or len(manifest.get("cases", [])) != 100):
        raise ValueError(f"Invalid raw final test manifest: {trace_dir}")
    validate_inventory(manifest["cases"], 10)
    if {(row["scenario_id"], row["schedule_hash"]) for row in manifest["cases"]} != {
        (row["scenario_id"], row["schedule_hash"]) for row in test["cases"]
    }:
        raise ValueError(f"Stored/raw test inventories differ: {directory}")
    test_spec = job["dataset"]["test"]
    if (test_spec["scenarios"] != [f"scenario-{i}" for i in range(90, 100)]
            or test_spec["arsenic"]["schedule_count"] != 10 or test_spec["arsenic"]["seed"] != 8130):
        raise ValueError(f"Wrong fixed test definition: {directory}")
    if sorted(row["index"] for row in manifest["cases"]) != list(range(100)):
        raise ValueError(f"Incomplete raw case indexes: {directory}")
    if len({record["file"] for record in manifest["cases"]}) != 100:
        raise ValueError(f"Duplicate raw case file: {directory}")
    for record in manifest["cases"]:
        if (record["scenario_id"] != test_spec["scenarios"][record["index"] % 10]
                or Path(record["file"]).name != record["file"]
                or not (trace_dir / record["file"]).is_file()):
            raise ValueError(f"Wrong raw case identity: {directory}")
    provenance.update(final_test_return=test["evaluation_return"],
                      test_policy_evidence="best bundle matches selected history snapshot; final/test uses source runner evaluate_best_model convention; raw manifest has no checkpoint hash",
                      metrics_sha256=file_hash(directory / "metrics.json"), config_sha256=file_hash(directory / "config.resolved.json"),
                      raw_manifest_sha256=file_hash(trace_dir / "manifest.json"), stored_protocol_version=metrics["protocol_version"],
                      runtime=metrics["runtime"])
    return manifest, curves, provenance


def build_paper_validation(root: Path) -> dict:
    """Export the two paper variants without associating tests with older maxima."""
    policies = ("baseline", "pretrained", *PAPER_VARIANTS)
    runs = discover_runs(root, policies)
    rows, sources, runtime_rows, jobs = [], [], [], {}
    inventory = None
    for policy in policies:
        for seed in SEEDS:
            directory = runs[policy, seed]
            _, curves, source = validate_run(directory, policy, seed, validation_only=True)
            source["run_directory"] = directory.relative_to(root).as_posix()
            expected_budget = LONG_BASELINE_BUDGET if policy == "baseline-4m" else 1_000_000
            if source["nominal_budget"] != expected_budget:
                raise ValueError(f"Paper comparison requires {expected_budget} nominal interactions: {directory}")
            if inventory is not None and source["validation_inventory"] != inventory:
                raise ValueError(f"Validation cases differ between paired runs: {directory}")
            inventory = source.pop("validation_inventory")
            sources.append(source)
            jobs[policy, seed] = read_json(directory / "config.resolved.json")["job"]
            if policy in PAPER_VARIANTS:
                rows.extend(curves)
                runtime = source["runtime"]
                runtime_rows.append(dict(approach=policy, training_seed=seed,
                    training_seconds=finite(runtime["training_seconds"], "training runtime", nonnegative=True),
                    checkpoint_evaluation_seconds=finite(runtime["checkpoint_evaluation_seconds"], "checkpoint runtime", nonnegative=True)))
    validate_matched_jobs(jobs, policies)
    return dict(format_version=1, data_kind="experimental",
                producer="tools/analysis/export_v89_ablation_paper_results.py",
                selection="Highest validation return across the complete archived budget; earliest checkpoint breaks ties.",
                return_scope="Original unnormalized rewards over the main window and complete tail. Physical protocol versions 2 and 3 change reference windows, not this return.",
                rows=rows, source_runs=sources, runtime_rows=runtime_rows)


def write_paper_validation(data: dict) -> Path:
    """Atomically replace only the supplementary validation-return input."""
    path = ROOT / "paper/results/variant-learning-plot-data.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".variant-learning-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def average(values: list) -> float | None:
    valid = [finite(value, "metric") for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def seed_statistics(values: list) -> dict:
    if len(values) != len(SEEDS):
        raise ValueError("Seed statistics require five entries including explicit non-applicability")
    valid = [finite(value, "seed value") for value in values if value is not None]
    return dict(seed_values=values, applicable_seeds=len(valid), mean=average(values),
                sample_sd=float(np.std(valid, ddof=1)) if len(valid) > 1 else None)


def summarize(rows: list, policies: tuple = POLICIES, components: tuple = PHYSICAL_COMPONENTS) -> list:
    summaries = []
    for policy in policies:
        for metric, component in components:
            values = []
            for seed in SEEDS:
                selected = [row for row in rows if (row["approach"], str(row["seed"]), row["metric"], row["component"])
                            == (policy, str(seed), metric, component)]
                if len(selected) != 10 or len({row["scenario"] for row in selected}) != 10:
                    raise ValueError(f"Expected ten scenario means: {policy}, {seed}, {metric}, {component}")
                values.append(average([row["value"] for row in selected]))
            summaries.append(dict(approach=policy, metric=metric, component=component, unit=UNITS[metric],
                                  **seed_statistics(values)))
    return summaries


def paired_summary(variant: dict, comparator: dict, *, nominal_budgets: dict | None = None) -> dict:
    values = [None if v is None or c is None else v - c
              for v, c in zip(variant["seed_values"], comparator["seed_values"], strict=True)]
    difference = seed_statistics(values)
    ci = None
    if difference["applicable_seeds"] == len(SEEDS):
        half = 2.7764451051977987 * difference["sample_sd"] / math.sqrt(len(SEEDS))
        ci = [difference["mean"] - half, difference["mean"] + half]
    return dict(variant=variant["approach"], comparator=comparator["approach"],
                metric=variant["metric"], component=variant["component"], unit=variant["unit"],
                nominal_budgets={policy: nominal_budgets[policy] if nominal_budgets is not None else
                                 3_400_000 if policy == "baseline-4m" else 1_000_000
                                 for policy in (variant["approach"], comparator["approach"])},
                budget_relation="additional PPO training" if "baseline-4m" in (variant["approach"], comparator["approach"]) else "equal PPO budget",
                variant_values=variant["seed_values"], comparator_values=comparator["seed_values"],
                paired_difference=difference, ci95=ci)


def clearance_pair_rows(case: dict, variant: str, comparator: str, support: dict) -> list:
    """Pair eligibility only within this contrast; never filter on a third policy."""
    leads = {variant: [], comparator: []}
    for event in case["clearance_events"]:
        clear = event["policy_clearance_h"]
        support["reference_events"] += 1
        for policy, label in ((variant, "variant"), (comparator, "comparator")):
            support[f"{label}_no_exceedance"] += clear[policy] is None
        eligible = clear[variant] is not None and clear[comparator] is not None
        support["eligible_events" if eligible else "paired_exclusions"] += 1
        if eligible:
            for policy in leads:
                leads[policy].append(event["reference_clearance_h"] - clear[policy])
    support["applicable_cases" if leads[variant] else "no_eligible_cases"] += 1
    return [dict(approach=policy, seed=str(case["training_seed"]), scenario=case["scenario_id"],
                 metric=METRIC, component="total", value=average(values)) for policy, values in leads.items()]


def validate_matched_jobs(jobs: dict, policies: tuple = POLICIES) -> None:
    """Hold all non-ablation settings and the seed's source BC bundle fixed."""
    ignored_job = {"algorithm", "name", "total_timesteps", "hooks", "output"}
    ignored_algorithm = {"pretrained_policy", "pretrained_components"}
    for seed in SEEDS:
        reference = jobs["baseline", seed]
        full_bundle = jobs["pretrained", seed]["algorithm"]["pretrained_policy"].replace("\\", "/")
        for policy in policies:
            job = jobs[policy, seed]
            if ({k: v for k, v in job.items() if k not in ignored_job} !=
                    {k: v for k, v in reference.items() if k not in ignored_job}
                    or {k: v for k, v in job["algorithm"].items() if k not in ignored_algorithm} !=
                    {k: v for k, v in reference["algorithm"].items() if k not in ignored_algorithm}):
                raise ValueError(f"Non-ablation job settings differ: {policy}, seed {seed}")
            if policy in VARIANTS and VARIANTS[policy]:
                bundle = job["algorithm"]["pretrained_policy"].replace("\\", "/")
                if bundle != full_bundle:
                    raise ValueError(f"Source BC bundle differs from Principal pretrained: {policy}, seed {seed}")


def build_package(root: Path) -> dict:
    runs = discover_runs(root)
    manifests, curves, provenance = {}, [], []
    jobs = {}
    test_returns = {}
    budgets = {}
    for policy in POLICIES:
        for seed in SEEDS:
            directory = runs[policy, seed]
            manifest, learning, source = validate_run(directory, policy, seed)
            source["run_directory"] = directory.relative_to(root).as_posix()
            manifests[policy, seed] = {(record["scenario_id"], record["schedule_hash"]): record for record in manifest["cases"]}
            curves.extend(learning)
            provenance.append(source)
            if budgets.setdefault(policy, source["nominal_budget"]) != source["nominal_budget"]:
                raise ValueError(f"Nominal budgets differ between seeds: {policy}")
            jobs[policy, seed] = read_json(directory / "config.resolved.json")["job"]
            test_returns[policy, seed] = {
                (row["scenario_id"], row["schedule_hash"]): row["values"]["evaluation_return"]
                for row in read_json(directory / "metrics.json")["test"]["cases"]
            }
    validate_matched_jobs(jobs)
    inventory = set(manifests["baseline", SEEDS[0]])
    if any(set(records) != inventory for records in manifests.values()):
        raise ValueError("Final test identities differ across policies or training seeds")
    raw_rows, cases, references = [], [], {}
    for seed in SEEDS:
        for identity in sorted(inventory):
            data, scores = {}, {}
            for policy in POLICIES:
                path = runs[policy, seed] / "evaluation-data/final/test"
                record = manifests[policy, seed][identity]
                try:
                    data[policy] = load_case(path, record)
                    rt = data[policy]["reference_time_seconds"]
                    if rt[0] != 0 or rt[-1] != 25 * 86400:
                        raise ValueError("Raw test time grid does not cover the specified episode")
                    score, event_mask = physical_projection(data[policy])
                    with np.load(path / record["file"], allow_pickle=False) as saved:
                        reward = saved["evaluation_reward"]
                    main, tail = scoring_windows(data[policy]["time_seconds"])
                    if reward.shape != (1200,) or not np.isfinite(reward).all() or not math.isclose(
                        float(reward[main | tail].sum()), test_returns[policy, seed][identity], rel_tol=1e-10, abs_tol=1e-7
                    ):
                        raise ValueError("Raw final test return differs from stored test case")
                except ValueError as exc:
                    raise ValueError(f"seed={seed}, policy={policy}, case={identity}: {exc}") from exc
                scores[policy] = {f"{metric}:{component}": component_value(score, metric, component)
                                  for metric, component in PHYSICAL_COMPONENTS}
                raw_rows.extend(dict(approach=policy, seed=str(seed), scenario=identity[0], metric=metric,
                                     component=component, value=scores[policy][f"{metric}:{component}"])
                                for metric, component in PHYSICAL_COMPONENTS)
            reference = {key: value for key, value in data["baseline"].items() if key.startswith("reference_")}
            previous = references.setdefault(identity, reference)
            if any(not np.array_equal(reference[key], previous[key]) for key in reference):
                raise ValueError(f"Reference differs across training seeds: {identity}")
            for policy in POLICIES:
                if (any(not np.array_equal(reference[key], data[policy][key]) for key in reference)
                        or manifests[policy, seed][identity]["index"] != manifests["baseline", seed][identity]["index"]):
                    raise ValueError(f"Reference or schedule index differs: {policy}, {seed}, {identity}")
            events = []
            for index, interval in enumerate(clusters(reference["reference_time_seconds"],
                    reference["reference_transported_arsenic"], reference["reference_event_strength_by_node"])):
                start, stop = interval.source_start_seconds, interval.next_source_start_seconds
                ref_clear = persistent_clearance(reference["reference_time_seconds"],
                    reference["reference_transported_arsenic"].sum(axis=1), start, stop)
                if ref_clear is None:
                    raise ValueError(f"Reference cluster has no exceedance: {identity}, event {index + 1}")
                clear = {}
                for policy in POLICIES:
                    try:
                        value = persistent_clearance(data[policy]["time_seconds"],
                                                      data[policy]["transported_arsenic"].sum(axis=1), start, stop)
                    except ValueError as exc:
                        raise ValueError(f"seed={seed}, policy={policy}, case={identity}, event={index + 1}: {exc}") from exc
                    clear[policy] = value / 3600 if value is not None else None
                events.append(dict(event=index + 1, source_start_h=start / 3600,
                                   reference_clearance_h=ref_clear / 3600, policy_clearance_h=clear))
            rt = reference["reference_time_seconds"]
            window = event_mask & (rt > 2 * 86400) & (rt <= 25 * 86400)
            cases.append(dict(training_seed=seed, scenario_id=identity[0], schedule_hash=identity[1],
                              schedule_id=f"arsenic-{manifests['baseline', seed][identity]['index'] // 10}",
                              reference_arsenic_burden=float(.5 * reference["reference_transported_arsenic"][window].sum()),
                              values=scores, clearance_events=events))
    scenario_rows = scenario_metrics(raw_rows)
    summary = summarize(scenario_rows)
    paired, clearance = [], []
    for variant, comparator in CONTRASTS:
        for metric, component in PHYSICAL_COMPONENTS:
            chosen = [row for row in summary if (row["metric"], row["component"]) == (metric, component)]
            paired.append(paired_summary(next(row for row in chosen if row["approach"] == variant),
                                         next(row for row in chosen if row["approach"] == comparator), nominal_budgets=budgets))
        rows, supports = [], []
        for seed in SEEDS:
            support = dict(training_seed=seed, reference_events=0, eligible_events=0, paired_exclusions=0,
                           variant_no_exceedance=0, comparator_no_exceedance=0,
                           applicable_cases=0, no_eligible_cases=0)
            for case in cases:
                if case["training_seed"] == seed:
                    rows.extend(clearance_pair_rows(case, variant, comparator, support))
            supports.append(support)
        projected = scenario_metrics(rows)
        statistics = summarize(projected, (variant, comparator), ((METRIC, "total"),))
        contrast = paired_summary(statistics[0], statistics[1], nominal_budgets=budgets)
        paired.append(contrast)
        clearance.append(dict(variant=variant, comparator=comparator, rows=projected,
                              support=supports, summary=contrast))
    points = []
    for identity in sorted(inventory):
        selected = [case for case in cases if (case["scenario_id"], case["schedule_hash"]) == identity]
        points.append(dict(scenario_id=identity[0], schedule_hash=identity[1], schedule_id=selected[0]["schedule_id"],
                           reference=selected[0]["reference_arsenic_burden"],
                           **{policy: average([case["values"][policy]["arsenic_burden:total"] for case in selected])
                              for policy in POLICIES}))
    costs = []
    for source in provenance:
        runtime = source["runtime"]
        for field in ("training_seconds", "checkpoint_evaluation_seconds", "final_validation_seconds", "final_test_seconds"):
            costs.append(dict(approach=source["approach"], training_seed=source["training_seed"],
                              stage=field.removesuffix("_seconds"), measured_seconds=finite(runtime[field], field, nonnegative=True),
                              execution_device=runtime["device"], training_environments=runtime["training_environments"],
                              evaluation_workers=runtime["evaluation_workers"],
                              online_interactions=source["actual_interactions"] if field == "training_seconds" else None))
    common = dict(format_version=1, data_kind="experimental")
    metadata = dict(**common, created_at=datetime.now(timezone.utc).isoformat(), results_root=str(root),
                    producer="tools/analysis/export_v89_ablation_paper_results.py", training_seeds=list(SEEDS),
                    policies=list(POLICIES), units=UNITS, paper_display=PAPER_DISPLAY,
                    physical_protocol_version=PROTOCOL_VERSION, exporter_sha256=file_hash(Path(__file__)),
                    aggregation="Equal applicable schedules within scenario; equal applicable scenarios within seed; five independent training seeds for mean and sample SD. Case burdens average each fixed case across seeds.",
                    paired_difference="variant minus comparator; 95% Student t interval across five paired seed differences, df=4; absent if any seed is non-applicable",
                    clearance="Only contrast-specific eligible reference clusters with exceedance in both compared policies; cluster leads -> case means -> scenario means -> seed means. No universal clearance mean.",
                    selection="Highest stored validation evaluation_return; earliest actual checkpoint breaks ties. Budgets count all completed PPO rollouts, tests use the selected best policy.",
                    cost_scope="Measured control-run runtime fields only; training_seconds excludes checkpoint evaluation. These component timings do not reconstruct total elapsed pipeline time or preparation cost; no shared offline cost is reassigned.",
                    protocol="Physical arrays rescored through current paper_arsenic_clearance reference windows/physical_metrics; legacy stored physical scores discarded. Return-based selection and curves preserve original experiment scores. Raw test reward sums agree with each stored case return.",
                    runs=provenance)
    return {
        "metadata.json": metadata,
        "learning-plot-data.json": dict(**common, rows=curves),
        "control-data.json": dict(**common, rows=scenario_rows, case_burdens=points, summary=summary),
        "case-metrics.json": dict(**common, rows=cases),
        "paired-comparisons.json": dict(**common, rows=paired, clearance=clearance),
        "computation-costs.json": dict(**common, rows=costs),
    }


def write_package(package: dict, output_dir: Path) -> None:
    if output_dir.resolve().is_relative_to((ROOT / "paper/results").resolve()):
        raise ValueError("Ablation export must not replace Principal publication inputs")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v89-ablations-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        for name, content in package.items():
            (staging / name).write_text(json.dumps(content, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in package:
            os.replace(staging / name, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/v89-ablation-paper-results")
    parser.add_argument("--check-only", action="store_true", help="Validate and rescore without writing output")
    parser.add_argument("--paper-validation", action="store_true",
                        help="Export the complete four-million baseline and without-actor validation comparisons to paper/results")
    args = parser.parse_args()
    if args.paper_validation:
        data = build_paper_validation(args.results_root)
        if not args.check_only:
            print(f"Wrote {write_paper_validation(data)}")
        print(f"Validated {len(data['source_runs'])} paired runs and {len(data['rows'])} variant checkpoint returns")
        return
    package = build_package(args.results_root)
    if not args.check_only:
        write_package(package, args.output_dir)
    print(f"{'Validated' if args.check_only else 'Wrote'} 30 selected policies, 3,000 rescored test traces, nine paired contrasts")
    print(f"Output: {args.output_dir}")
    for row in package["control-data.json"]["summary"]:
        if (row["metric"], row["component"]) == ("arsenic_burden", "total"):
            print(f"  {row['approach']}: arsenic burden {row['mean']:.6g} +/- {row['sample_sd']:.6g} {row['unit']}")


if __name__ == "__main__":
    main()
