#!/usr/bin/env python3
"""Project the full network-delay diagnostic into the paper's compact tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/analysis/network-delays/low-source-outflow"
DEFAULT_OUTPUT = ROOT / "paper/results/network-delay-summary.json"
HISTORY_HOURS = (12, 16, 20, 24, 28, 32)
HORIZON_HOURS = (4, 6, 8, 10, 12, 14, 16)


def export(input_dir: Path, output: Path) -> None:
    summary = json.loads((input_dir / "summary.json").read_text())
    counts = summary["counts"]
    with (input_dir / "node_trial_cases.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != counts["node_trial_cases"]:
        raise ValueError("Node-case count differs from the diagnostic summary")

    trials: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        trials.setdefault((row["scenario_id"], int(row["start_step"])), []).append(row)
    if len(trials) != counts["trials"]:
        raise ValueError("Trial count differs from the diagnostic summary")

    threshold = float(summary["design"]["absolute_threshold_mg_l"])
    response_shares = {hours: [] for hours in HISTORY_HOURS}
    for (scenario, start_step), trial_rows in trials.items():
        path = input_dir / "cases/trials" / f"{scenario}-{start_step}.npz"
        with np.load(path, allow_pickle=False) as raw:
            node_ids = [str(node) for node in raw["node_ids"]]
            times = raw["times_seconds"] - raw["times_seconds"][0]
            delta = raw["action_chlorine_mg_l"].astype(np.float64) - raw["reference_chlorine_mg_l"].astype(np.float64)
        delta = np.where(delta > threshold, delta, 0.0)
        by_node = {row["node_id"]: row for row in trial_rows}
        if set(by_node) != set(node_ids):
            raise ValueError(f"Node identities differ for {scenario}-{start_step}")
        for index, node_id in enumerate(node_ids):
            row = by_node[node_id]
            total = float(delta[:, index].sum())
            responsive = row["responsive"] == "True"
            if responsive != (total > 0):
                raise ValueError(f"Response mismatch for {scenario}-{start_step}, node {node_id}")
            if responsive:
                for hours in HISTORY_HOURS:
                    response_shares[hours].append(float(delta[times <= hours * 3600, index].sum()) / total)

    responsive_rows = [row for row in rows if row["responsive"] == "True"]
    if len(responsive_rows) != counts["responsive"]:
        raise ValueError("Responsive-case count differs from the diagnostic summary")
    history = []
    for hours in HISTORY_HOURS:
        still_visible = sum(float(row["last_detectable_hours"]) > hours for row in responsive_rows)
        settled_trials = sum(
            all(row["responsive"] != "True" or float(row["last_detectable_hours"]) <= hours
                for row in trial_rows)
            for trial_rows in trials.values()
        )
        history.append({
            "hours": hours,
            "mean_response_seen_pct": 100.0 * float(np.mean(response_shares[hours])),
            "still_visible_cases": still_visible,
            "all_junctions_settled_trials": settled_trials,
        })
    horizons = []
    for hours in HORIZON_HOURS:
        horizons.append({
            "hours": hours,
            "begun": sum(float(row["onset_hours"]) <= hours for row in responsive_rows),
            "peak_reached": sum(float(row["peak_hours"]) <= hours for row in responsive_rows),
            "settled": sum(float(row["last_detectable_hours"]) <= hours for row in responsive_rows),
        })
    payload = {
        "source": "data/analysis/network-delays/low-source-outflow/summary.json, node_trial_cases.csv, and cases/trials/*.npz",
        "counts": counts,
        "history": history,
        "horizons": horizons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(f"Exported {len(trials)} trials and {len(rows)} node cases to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.input_dir, args.output)


if __name__ == "__main__":
    main()
