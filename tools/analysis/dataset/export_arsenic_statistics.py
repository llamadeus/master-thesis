#!/usr/bin/env python3
"""Export event statistics and spatial reach from one fixed validation catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.forecasting.features import DERIVED_ARSENIC_VERSION, derive_transported_arsenic
from training.datasets.hanoi import control_full
from tools.analysis.dataset.export_arsenic_examples import materialize

THRESHOLD = 0.01


def contamination_clusters(events, times, transported, threshold=THRESHOLD):
    """Group sources until persistent aggregate clearance before the next source.

    Source-only events are retained as zero-reach clusters. This differs from
    a reference-event-window inventory, which would omit them altogether.
    """
    times = np.asarray(times)
    values = np.asarray(transported, dtype=np.float64)
    if not events:
        return []
    if len(times) < 2 or np.any(np.diff(times) != times[1] - times[0]):
        raise ValueError("Expected a regular reporting grid")
    if values.shape[0] != len(times) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid transported arsenic history")
    step = float(times[1] - times[0])
    aggregate = values.sum(axis=1)
    ordered = sorted(enumerate(events), key=lambda pair: pair[1]["start_time_seconds"])
    clusters = []
    position = 0
    while position < len(ordered):
        event_index, event = ordered[position]
        members = [event_index]
        start = event["start_time_seconds"]
        source_end = event["end_time_seconds"]
        position += 1
        while True:
            next_start = ordered[position][1]["start_time_seconds"] if position < len(ordered) else float(times[-1]) + step
            if next_start < source_end:
                merge = True
            else:
                # The last exceedance handles dips followed by a later rebound.
                prior = np.flatnonzero((times >= start) & (times < next_start) & (aggregate > threshold))
                last_exceedance = float(times[prior[-1]]) if len(prior) else start - step
                end = max(float(source_end), last_exceedance + step)
                merge = position < len(ordered) and end >= next_start
            if merge:
                event_index, event = ordered[position]
                members.append(event_index)
                source_end = max(source_end, event["end_time_seconds"])
                position += 1
                continue
            mask = (times >= start) & (times < end)
            if not np.any(mask):
                raise ValueError("Empty contamination cluster")
            censored = end > times[-1]
            peaks = values[mask].max(axis=0)
            clusters.append(dict(event_indices=members, start_time_seconds=start,
                                 end_time_seconds=end, source_end_time_seconds=source_end,
                                 right_censored=bool(censored), node_peak_mg_l=peaks.tolist()))
            break
    return clusters


def validation_catalog(workers):
    runs = materialize(PROJECT_ROOT / "data/runs", workers)
    durations = {int(trajectory.time_seconds[-1]) for _, trajectory in runs}
    if len(durations) != 1:
        raise ValueError("Validation cases must share one complete episode duration")
    records = []
    for record, trajectory in runs:
        transported, _ = derive_transported_arsenic(
            trajectory.arsenic, trajectory.event_strength_by_node, trajectory.hydraulic_outflow,
            trajectory.time_seconds, flow_units=trajectory.flow_units)
        clusters = contamination_clusters(record["events"], trajectory.time_seconds, transported)
        if any(c["right_censored"] for c in clusters):
            raise ValueError(f"Uncleared contamination at episode end: run {record['index']}")
        records.append(dict(run_index=record["index"], scenario_id=record["scenario_id"],
                            schedule_index=record["schedule_index"], schedule_hash=record["schedule_hash"],
                            calendar_start_day=trajectory.calendar_start_day,
                            cache_dir=record["cache_dir"], node_ids=list(trajectory.node_ids),
                            events=record["events"], clusters=clusters))
    return dict(dataset="control_full", split="validation", policy="zero-injection",
                seed=control_full.validation.arsenic.seed,
                schedules_per_scenario=control_full.validation.arsenic.schedule_count,
                duration_seconds=durations.pop(),
                threshold_mg_l=THRESHOLD, derived_arsenic_version=DERIVED_ARSENIC_VERSION,
                runs=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper/dataset/arsenic-statistics.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    catalog = validation_catalog(args.workers)
    payload = dict(schema_version=2, catalog=catalog,
                   generator_sha256=hashlib.sha256((PROJECT_ROOT / 'packages/trainer/schedules/arsenic.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Exported {len(catalog['runs'])} validation schedules, "
          f"{sum(len(r['events']) for r in catalog['runs'])} events and "
          f"{sum(len(r['clusters']) for r in catalog['runs'])} clusters to {args.output}")


if __name__ == "__main__":
    main()
