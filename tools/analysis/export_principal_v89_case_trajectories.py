"""Project completed case rollouts into the paper's trajectory data. No simulation."""

import argparse
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np


APPROACHES = ("ppo_initial", "ppo_best", "mpc", "bc_initial", "finetuned_best")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from packages.trainer.evaluation.protocol import reference_windows, physical_metrics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(rollout_dir):
    comparison = read_json(rollout_dir / "comparison.json")
    selected = read_json(ROOT / "paper/results/control-data.json")["selected"]
    assert comparison["status"] == "complete", "Rollouts are incomplete"
    assert comparison["scenario_id"] == selected["scenario_id"]
    assert comparison["schedule_index"] == selected["schedule_index"]
    seed = str(comparison["seed"])
    expected = next(row for row in selected["burden_by_seed"] if row["seed"] == seed)
    time = np.arange(0, comparison["duration_seconds"] + 1, comparison["timestep_seconds"])
    controllers = {}
    common = None
    reference_keys = ("reference_time_seconds", "reference_event", "reference_transported_arsenic")
    for approach in APPROACHES:
        directory = rollout_dir / approach
        run = read_json(directory / "run.json")
        metrics = read_json(directory / "metrics.json")
        metadata = read_json(directory / "metadata.json")
        assert run["status"] == metrics["status"] == "complete"
        assert metadata["scenario_id"] == comparison["scenario_id"]
        assert metadata["arsenic_schedule_hash"] == comparison["schedule_hash"]
        with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory, np.load(
            directory / "evaluation-data/case-00000.npz", allow_pickle=False
        ) as evaluation:
            np.testing.assert_array_equal(trajectory["time_seconds"], time)
            np.testing.assert_array_equal(evaluation["time_seconds"], time[1:])
            np.testing.assert_array_equal(evaluation["chlorine"], trajectory["chlorine"][1:])
            assert np.all(trajectory["arsenic"][0] == 0), "Initial transported burden must be zero"
            chlorine = trajectory["chlorine"].astype(float)
            assert chlorine.shape == (len(time), len(metadata["node_ids"]))
            shared = {key: evaluation[key].copy() for key in reference_keys}
            shared["source"] = trajectory["event_strength_by_node"].copy()
            events = [SimpleNamespace(**event) for event in comparison["schedule_events"]]
            corrected_event, corrected_post = reference_windows(time, shared["reference_transported_arsenic"], events)
            shared["reference_event"] = corrected_event
            rescored = physical_metrics(evaluation["time_seconds"], evaluation["chlorine"],
                evaluation["transported_arsenic"], evaluation["injection"],
                corrected_event[1:], corrected_post[1:], comparison["timestep_seconds"])
            if common is None:
                common = shared
                node_ids = metadata["node_ids"]
            else:
                assert node_ids == metadata["node_ids"]
                for key in shared:
                    np.testing.assert_array_equal(common[key], shared[key])
            transported = np.concatenate(([0.0], evaluation["transported_arsenic"].astype(float).sum(axis=1)))
            arrays = {
                "injection_g_min": trajectory["injection"].astype(float) / 1000,
                "chlorine_mean_mg_l": chlorine.mean(axis=1),
                "chlorine_population_sd_mg_l": chlorine.std(axis=1, ddof=0),
                "transported_arsenic_sum_mg_l": transported,
            }
            assert all(len(a) == len(time) and np.isfinite(a).all() for a in arrays.values())
            controllers[approach] = {
                **{key: value.tolist() for key, value in arrays.items()},
                "arsenic_burden_mg_h_l": rescored["values"]["arsenic_burden"],
                "evaluation_return": metrics["evaluation_return"],
            }
    for approach, policy in (("ppo_best", "baseline"), ("finetuned_best", "pretrained")):
        assert math.isclose(controllers[approach]["arsenic_burden_mg_h_l"], expected[policy],
                            rel_tol=1e-6, abs_tol=1e-4), "Final policy differs from selected paper case"
    np.testing.assert_array_equal(common["reference_time_seconds"], time)
    active = common["reference_event"].astype(bool)
    edges = np.diff(np.concatenate(([False], active, [False])).astype(int))
    windows = [{"start_index": int(start), "end_index": min(int(end), len(time) - 1)}
               for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return {
        "schema_version": 1,
        "source_run": rollout_dir.name,
        "scenario_id": comparison["scenario_id"],
        "schedule_id": selected["schedule_id"],
        "schedule_hash": comparison["schedule_hash"],
        "seed": comparison["seed"],
        "policy_selection": comparison["policy_selection"],
        "node_count": len(node_ids),
        "time_days": (time / 86400).tolist(),
        "source_injection_g_min": (common["source"].astype(float).sum(axis=1) / 1000).tolist(),
        "reference_transported_arsenic_sum_mg_l": common["reference_transported_arsenic"].astype(float).sum(axis=1).tolist(),
        "reference_windows": windows,
        "source_events": comparison["schedule_events"],
        "controllers": controllers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/results/case-trajectory-data.json")
    args = parser.parse_args()
    data = export(args.rollout_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(data['controllers'])} complete rollouts, {len(data['time_days'])} samples each: {args.output}")


if __name__ == "__main__":
    main()
