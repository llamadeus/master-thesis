"""Export chlorine-excess contributions after reference clearance and policy-aligned curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.trainer.evaluation.protocol import reference_clusters, reference_windows, scoring_windows
from paper.tools.results_projection import SEEDS
from tools.analysis.paper_arsenic_clearance import load_case, source_intervals, physical_projection
from tools.analysis.export_principal_v89_paper_results import run_paths

APPROACHES = ("baseline", "pretrained")
MAX_STEP = 24


def post_cluster_fractions(arrays):
    """Align every cluster; stop before the next injection or at episode end."""
    times = arrays["reference_time_seconds"]
    events = source_intervals(times, arrays["reference_event_strength_by_node"])
    reference = reference_clusters(times, arrays["reference_transported_arsenic"], events)
    if not reference:
        return []
    transported = np.vstack((np.zeros((1, arrays["transported_arsenic"].shape[1])),
                             arrays["transported_arsenic"]))
    policy = reference_clusters(times, transported, events)
    observed_times = arrays["time_seconds"]
    results = []
    for index, cluster in enumerate(policy):
        clearance = cluster.clearance_time_seconds
        fractions = []
        for step in range(MAX_STEP + 1):
            time = clearance + step * 1800
            selected = np.flatnonzero((observed_times == time)
                                     & (observed_times < cluster.next_source_start_seconds))
            fractions.append(float(np.mean(arrays["chlorine"][selected[0]] > .4))
                             if len(selected) else None)
        results.append(dict(cluster_index=index, clearance_time_seconds=clearance,
                            next_source_start_seconds=cluster.next_source_start_seconds,
                            fractions=fractions))
    return results


def aggregate(records):
    """Average clusters within rollouts, then schedules, scenarios, and seeds."""
    rows = []
    for approach in APPROACHES:
        selected = [r for r in records if r["approach"] == approach]
        for step in range(MAX_STEP + 1):
            rollouts = defaultdict(list)
            for record in selected:
                value = record["fractions"][step]
                if value is not None:
                    rollouts[(record["seed"], record["scenario_id"], record["schedule_hash"])].append(value)
            groups = defaultdict(list)
            for (seed, scenario, _), values in rollouts.items():
                groups[(seed, scenario)].append(float(np.mean(values)))
            seed_values = defaultdict(list)
            for (seed, _), values in groups.items():
                seed_values[seed].append(float(np.mean(values)))
            means = [float(np.mean(seed_values[seed])) for seed in SEEDS if seed in seed_values]
            rows.append(dict(approach=approach, step=step, hours=step / 2,
                             mean=float(np.mean(means)) if means else None,
                             observation_count=sum(len(v) for v in rollouts.values()),
                             rollout_count=len(rollouts),
                             seed_count=len(means)))
    return rows


def excess_partition(arrays):
    """Split reference-normal excess using common 12-hour reference-clearance windows."""
    times = arrays["time_seconds"]
    main, _ = scoring_windows(times)
    reference_times = arrays["reference_time_seconds"]
    events = source_intervals(reference_times, arrays["reference_event_strength_by_node"])
    reference_event, _ = reference_windows(
        reference_times, arrays["reference_transported_arsenic"], events)
    if not np.array_equal(times, reference_times[1:]):
        raise ValueError("Policy and reference reporting grids differ")
    scored = main & ~reference_event[1:]
    followup = np.zeros(len(times), dtype=bool)
    for cluster in reference_clusters(reference_times, arrays["reference_transported_arsenic"], events):
        clearance = cluster.clearance_time_seconds
        followup |= ((times >= clearance) & (times < clearance + MAX_STEP * 1800)
                     & (times < cluster.next_source_start_seconds))
    excess = arrays["chlorine"] > .4
    inside = int(np.count_nonzero(excess[scored & followup]))
    outside = int(np.count_nonzero(excess[scored & ~followup]))
    return dict(inside_count=inside, outside_count=outside, total_count=inside + outside,
                normal_observation_count=int(np.count_nonzero(scored) * excess.shape[1]))


def summarize_excess_partitions(records):
    """Decompose the table metric with its schedule/scenario/seed weighting."""
    summaries = []
    for approach in APPROACHES:
        selected = [r for r in records if r["approach"] == approach]
        inside = sum(r["inside_count"] for r in selected)
        outside = sum(r["outside_count"] for r in selected)
        total = inside + outside
        component_means = {}
        for component in ("inside", "outside"):
            scenarios = defaultdict(list)
            for record in selected:
                denominator = record["normal_observation_count"]
                if denominator:
                    scenarios[(record["seed"], record["scenario_id"])].append(
                        record[f"{component}_count"] / denominator)
            seeds = defaultdict(list)
            for (seed, _), values in scenarios.items():
                seeds[seed].append(float(np.mean(values)))
            means = [float(np.mean(seeds[seed])) for seed in SEEDS if seed in seeds]
            component_means[component] = float(np.mean(means)) if means else None
        metric_total = (component_means["inside"] + component_means["outside"]
                        if component_means["inside"] is not None else None)
        by_seed = [dict(seed=seed,
                        inside_count=sum(r["inside_count"] for r in selected if r["seed"] == seed),
                        outside_count=sum(r["outside_count"] for r in selected if r["seed"] == seed))
                   for seed in SEEDS]
        summaries.append(dict(approach=approach, inside_count=inside, outside_count=outside,
                              total_count=total,
                              inside_fraction=component_means["inside"] / metric_total if metric_total else None,
                              outside_fraction=component_means["outside"] / metric_total if metric_total else None,
                              inside_metric_fraction=component_means["inside"],
                              outside_metric_fraction=component_means["outside"],
                              total_metric_fraction=metric_total,
                              rollout_count=len(selected), by_seed=by_seed))
    return summaries


def export(root):
    records, support, partitions = [], [], []
    inventory = None
    for seed in SEEDS:
        plan = json.loads((root / "pipelines/hanoi-principal-v89" / f"seed-{seed}" / "plan.json").read_text())
        for approach in APPROACHES:
            directory = run_paths(root, plan, approach)
            metrics = json.loads((directory / "metrics.json").read_text())
            published = json.loads((directory / "best/manifest.json").read_text())
            selected = max(sorted(metrics["history"], key=lambda r: r["timesteps"]),
                           key=lambda r: r["evaluation"]["evaluation_return"])
            if published["evaluation"] != selected["evaluation"] or metrics["final_validation"] != selected["evaluation"]:
                raise ValueError(f"Final tests do not use the validation-selected policy: {directory}")
            case_dir = directory / "evaluation-data/final/test"
            manifest = json.loads((case_dir / "manifest.json").read_text())
            identities = {(c["scenario_id"], c["schedule_hash"]) for c in manifest["cases"]}
            if len(identities) != 100 or (inventory is not None and identities != inventory):
                raise ValueError(f"Final test case inventory differs: {case_dir}")
            inventory = identities
            eligible = 0
            cluster_count = 0
            for case in manifest["cases"]:
                try:
                    arrays = load_case(case_dir, case)
                    result = post_cluster_fractions(arrays)
                    partition = excess_partition(arrays)
                    scored, _ = physical_projection(arrays)
                    counts = scored["counts"]["normal_chlorine_excess_fraction"]
                    if (partition["total_count"] != counts["numerator"]
                            or partition["normal_observation_count"] != counts["denominator"]):
                        raise ValueError("Excess partition does not reproduce the primary metric counts")
                except ValueError as exc:
                    raise ValueError(f"seed={seed}, policy={approach}, case={case['index']}: {exc}") from exc
                partitions.append(dict(approach=approach, seed=seed,
                                       scenario_id=case["scenario_id"], **partition))
                if result:
                    eligible += 1
                    cluster_count += len(result)
                    records.extend(dict(approach=approach, seed=seed,
                                        scenario_id=case["scenario_id"],
                                        schedule_hash=case["schedule_hash"], **cluster)
                                   for cluster in result)
            support.append(dict(approach=approach, seed=seed, eligible_rollouts=eligible,
                                cluster_count=cluster_count,
                                excluded_rollouts=100 - eligible))
            print(f"{approach}, seed {seed}: {cluster_count} clusters in {eligible} rollouts", flush=True)
    return dict(format_version=2, data_kind="experimental", timestep_seconds=1800,
                max_step=MAX_STEP, chlorine_threshold_mg_l=.4,
                alignment="all_policy_cluster_clearances", followup_stop="next_source_injection_or_episode_end",
                support=support,
                excess_partition_scope="main_reference_normal_node_time_exceedances",
                excess_partition_alignment="reference_cluster_clearances",
                excess_partition_aggregation="schedule_scenario_seed_metric_contributions",
                excess_partition=summarize_excess_partitions(partitions),
                rows=aggregate(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=ROOT / "paper/results/post-cluster-data.json")
    args = parser.parse_args()
    data = export(args.results_root)
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
