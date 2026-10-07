#!/usr/bin/env python3
"""Export an isolated zero-injection validation event and its clearance tail."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from packages.trainer.data import resolve_dataset_split
from packages.trainer.evaluation.protocol import reference_windows
from packages.trainer.forecasting.features import derive_transported_arsenic
from packages.trainer.schedules import fixed_schedule
from packages.trainer.simulation import prepare_hydraulic_outflow_cache
from packages.trainer.simulation.epyt import EpytFlowBackend
from training.datasets.hanoi import control_full

SCENARIO_ID = "scenario-85"
SCHEDULE_INDEX = 2
CONTEXT_SECONDS = 4 * 3600


def export(output: Path) -> None:
    catalog = json.loads((ROOT / "paper/dataset/arsenic-statistics.json").read_text())["catalog"]
    record = next(run for run in catalog["runs"]
                  if run["scenario_id"] == SCENARIO_ID and run["schedule_index"] == SCHEDULE_INDEX)
    if (catalog["dataset"], catalog["split"], catalog["policy"]) != (
            "control_full", "validation", "zero-injection"):
        raise ValueError("Expected the fixed control-validation catalog")

    split = replace(control_full.validation, scenarios=(SCENARIO_ID,))
    dataset = resolve_dataset_split(control_full.network, split, split="evaluation",
                                    scenarios_dir=control_full.scenarios_dir,
                                    uncertainty_seed=control_full.uncertainty_seed)
    scenario = dataset.scenarios[0]
    backend = EpytFlowBackend()
    hydraulics = prepare_hydraulic_outflow_cache(dataset, backend, workers=1,
                                                  splits=("evaluation",))
    schedule = fixed_schedule(dataset.schedule, SCENARIO_ID, SCHEDULE_INDEX,
                              scenario.eligible_arsenic_node_ids,
                              hydraulic_profile=hydraulics.profile(SCENARIO_ID))
    if schedule.hash != record["schedule_hash"]:
        raise ValueError("The selected validation schedule differs from the catalog")

    session = backend.create_session(str(scenario.path),
        hydraulic_timestep_seconds=dataset.hydraulic_timestep_seconds,
        calendar_start_day=scenario.calendar_start_day,
        chlorine_source_ids=scenario.chlorine_source_ids,
        monitored_node_ids=scenario.monitored_node_ids)
    try:
        samples = [session.reset(schedule.events)]
        for _ in range(dataset.duration_seconds // dataset.hydraulic_timestep_seconds):
            sample, done = session.step(0.0, 0.0)
            samples.append(sample)
            if done:
                break
        times = np.asarray([sample.time_seconds for sample in samples])
        if not np.array_equal(times, np.arange(0, dataset.duration_seconds + 1,
                                               dataset.hydraulic_timestep_seconds)):
            raise ValueError("The simulated episode has an incomplete reporting grid")
        raw = np.asarray([sample.arsenic for sample in samples])
        chlorine = np.asarray([sample.chlorine for sample in samples])
        injection = np.asarray([sample.injection for sample in samples])
        if np.any(chlorine != 0) or np.any(injection != 0):
            raise ValueError("The reference simulation contains chlorine")
        strengths = np.asarray([sample.event_strength_by_node for sample in samples])
        outflow = np.asarray([sample.hydraulic_outflow for sample in samples])
        transported, local = derive_transported_arsenic(raw, strengths, outflow,
                                                         times, flow_units=session.flow_units)
        event_mask, _ = reference_windows(times, transported, schedule.events)
        starts = np.flatnonzero(event_mask & ~np.r_[False, event_mask[:-1]])
        ends = np.flatnonzero(event_mask & ~np.r_[event_mask[1:], False]) + 1
        windows = [{"start_time_seconds": int(times[start]),
                    "end_time_seconds": int(times[end])}
                   for start, end in zip(starts, ends, strict=True)]
        if len(schedule.events) != 1 or len(windows) != 1:
            raise ValueError("The selected example no longer has one event and one window")
        event = schedule.events[0]
        window = windows[0]
        plot_start = event.start_time_seconds - CONTEXT_SECONDS
        plot_end = window["end_time_seconds"] + CONTEXT_SECONDS
        if plot_start < 0 or plot_end > dataset.duration_seconds:
            raise ValueError("The selected event cannot be shown with four hours of context")
        selection = (times >= plot_start) & (times <= plot_end)
        demand = np.asarray([sample.demand for sample in samples])
        if session.flow_units != "CMH":
            raise ValueError("The demand panel requires cubic metres per hour")
        payload = {
            "schema_version": 2,
            "dataset": "control_full",
            "split": "validation",
            "policy": "zero-injection",
            "scenario_id": SCENARIO_ID,
            "schedule_index": SCHEDULE_INDEX,
            "schedule_hash": schedule.hash,
            "calendar_start_day": scenario.calendar_start_day,
            "timestep_seconds": dataset.hydraulic_timestep_seconds,
            "duration_seconds": dataset.duration_seconds,
            "clearance_threshold_mg_l": catalog["threshold_mg_l"],
            "plot_start_time_seconds": plot_start,
            "plot_end_time_seconds": plot_end,
            "events": record["events"],
            "reference_windows": windows,
            "time_seconds": times[selection].tolist(),
            "demand_mean_m3_h": demand[selection].mean(axis=1).tolist(),
            "demand_sd_m3_h": demand[selection].std(axis=1).tolist(),
            "local_source_arsenic_sum_mg_l": local[selection].sum(axis=1).tolist(),
            "transported_arsenic_sum_mg_l": transported[selection].sum(axis=1).tolist(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        lag_hours = (window["end_time_seconds"] - event.end_time_seconds) / 3600
        print(f"Exported {SCENARIO_ID} schedule {SCHEDULE_INDEX}: "
              f"one event, {lag_hours:g} h after source end to clearance, "
              f"{int(selection.sum())} report times to {output}")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/dataset/reference-window-example.json")
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
