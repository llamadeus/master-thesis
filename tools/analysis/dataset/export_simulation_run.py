#!/usr/bin/env python3
"""Export the four raw arrays used by the paper for one offline trajectory.

The full trajectory remains in the offline cache and is validated before export.
No plotted samples, nodes, or floating-point precision are removed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from packages.trainer.data import resolve_dataset_split
from packages.trainer.forecasting.materialization import ForecastTrajectoryResolver
from packages.trainer.schedules import fixed_schedule
from packages.trainer.schedules.chlorine import fixed_chlorine_schedule
from packages.trainer.simulation import prepare_hydraulic_outflow_cache
from packages.trainer.specs import ForecastTrajectoryDataSpec
from training.datasets.hanoi import forecasting_full

ARRAY_NAMES = ("time_seconds", "pressure", "demand", "flow", "chlorine", "arsenic",
               "injection", "actions", "event_active", "event_strength_by_node",
               "hydraulic_outflow")
# paper/chapters/04_dataset/simulation-run.typ reads only these arrays.
PAPER_ARRAY_NAMES = ("time_seconds", "chlorine", "arsenic", "injection")


def scenario_id(value):
    try:
        number = int(value.removeprefix("scenario-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a scenario identifier from 0 to 99") from exc
    if not 0 <= number <= 99:
        raise argparse.ArgumentTypeError("Use a scenario identifier from 0 to 99")
    return f"scenario-{number}"


def resolve_selection(identifier, seed=None, chlorine_seed=None):
    number = int(identifier.removeprefix("scenario-"))
    partition = "train" if number < 80 else "validation" if number < 90 else "test"
    split = getattr(forecasting_full, partition)
    arsenic_seed = split.arsenic.seed if seed is None else seed
    chlorine_seed = (split.chlorine.seed if seed is None else seed + 1000) if chlorine_seed is None else chlorine_seed
    selected = replace(split, scenarios=(identifier,),
                       arsenic=replace(split.arsenic, seed=arsenic_seed),
                       chlorine=replace(split.chlorine, seed=chlorine_seed))
    return resolve_dataset_split(forecasting_full.network, selected,
                                 scenarios_dir=forecasting_full.scenarios_dir,
                                 uncertainty_seed=forecasting_full.uncertainty_seed,
                                 split="evaluation" if partition == "validation" else partition), partition


def validate_trajectory(trajectory, dataset, arsenic, chlorine):
    """Check the reset sample and every executed interval through the endpoint."""
    times = np.arange(0, dataset.duration_seconds + dataset.hydraulic_timestep_seconds,
                      dataset.hydraulic_timestep_seconds)
    if not np.array_equal(trajectory.time_seconds, times):
        raise ValueError("Expected the standard offline half-hour reporting grid")
    for name in ARRAY_NAMES:
        values = np.asarray(getattr(trajectory, name))
        if values.shape[0] != len(times) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete or non-finite {name}")
        if name in ("pressure", "demand", "chlorine", "arsenic", "event_strength_by_node", "hydraulic_outflow"):
            if values.shape != (len(times), len(trajectory.node_ids)):
                raise ValueError(f"Invalid node dimension for {name}")
    if trajectory.flow.shape != (len(times), len(trajectory.link_ids)):
        raise ValueError("Invalid link dimension for flow")
    np.testing.assert_array_equal(trajectory.injection[1:], chlorine.values)
    if trajectory.injection[0] != 0.0:
        raise ValueError("Initial reset sample must precede the prescribed injection profile")
    expected = np.zeros_like(trajectory.event_strength_by_node)
    for event in arsenic.events:
        active = (times >= event.start_time_seconds) & (times < event.end_time_seconds)
        expected[active, trajectory.node_ids.index(event.node_id)] += event.strength
    np.testing.assert_allclose(trajectory.event_strength_by_node, expected, rtol=1e-6)
    np.testing.assert_array_equal(trajectory.event_active, np.any(expected > 0, axis=1))
    if trajectory.flow_units != "CMH":
        raise ValueError("This Hanoi figure expects native CMH hydraulic units")


def export_run(dataset, partition, reference, trajectory, arsenic, chlorine,
               arsenic_index, chlorine_index, output):
    validate_trajectory(trajectory, dataset, arsenic, chlorine)
    scenario = dataset.scenarios[0]
    cache_dir = reference.cache_dir.resolve()
    payload = dict(
        schema_version=2, scenario_id=scenario.scenario_id, partition=partition,
        calendar_start_day=scenario.calendar_start_day,
        duration_seconds=dataset.duration_seconds,
        timestep_seconds=dataset.hydraulic_timestep_seconds,
        warmup_days=dataset.warmup_days, main_days=dataset.main_days, tail_days=dataset.tail_days,
        node_ids=list(trajectory.node_ids), link_ids=list(trajectory.link_ids),
        source_ids=list(trajectory.source_ids),
        selection=dict(arsenic_seed=dataset.schedule.seed, chlorine_seed=dataset.chlorine_schedule.seed,
                       arsenic_index=arsenic_index, chlorine_index=chlorine_index,
                       schedule_count=dataset.schedule.schedule_count),
        arsenic_events=[asdict(e) for e in arsenic.events],
        chlorine_families=list(chlorine.families),
        units=dict(time_seconds="s", chlorine="mg/L", arsenic="mg/L", injection="mg/min"),
        cache=dict(identity=reference.identity,
                   directory=str(cache_dir.relative_to(PROJECT_ROOT) if cache_dir.is_relative_to(PROJECT_ROOT) else cache_dir),
                   trajectory_sha256=hashlib.sha256((cache_dir / "trajectory.npz").read_bytes()).hexdigest(),
                   arsenic_schedule_hash=arsenic.hash, chlorine_profile_hash=chlorine.hash),
        source_sha256={name: hashlib.sha256((scenario.path / name).read_bytes()).hexdigest()
                       for name in ("scenario.inp", "scenario.msx", "scenario.epytflow_scenario_config", "manifest.json")},
        arrays={name: np.asarray(getattr(trajectory, name)).tolist() for name in PAPER_ARRAY_NAMES},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", type=scenario_id, default="scenario-0")
    parser.add_argument("--seed", type=int, default=42, help="Arsenic seed (default: 42); chlorine defaults to this seed + 1000")
    parser.add_argument("--chlorine-seed", type=int, help="Override the chlorine seed independently")
    parser.add_argument("--arsenic-index", type=int, choices=range(5), default=1)
    parser.add_argument("--chlorine-index", type=int, choices=range(5), default=0)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper/dataset/simulation-run.json")
    args = parser.parse_args()
    dataset, partition = resolve_selection(args.scenario, args.seed, args.chlorine_seed)
    scenario = dataset.scenarios[0]
    hydraulics = prepare_hydraulic_outflow_cache(dataset, None, workers=1, splits=(dataset.split,))
    arsenic = fixed_schedule(dataset.schedule, scenario.scenario_id, args.arsenic_index,
        scenario.eligible_arsenic_node_ids, warmup_days=dataset.warmup_days,
        main_days=dataset.main_days, hydraulic_profile=hydraulics.profile(scenario.scenario_id))
    chlorine = fixed_chlorine_schedule(dataset.chlorine_schedule, scenario.scenario_id, args.chlorine_index,
        step_count=dataset.duration_seconds // dataset.hydraulic_timestep_seconds,
        timestep_seconds=dataset.hydraulic_timestep_seconds)
    # Use the exact offline backend and endpoint-inclusive reporting grid.
    # Resolve only the requested pair instead of materializing all 25 combinations.
    from packages.trainer.forecasting.runner import _SessionForecastBackend
    from packages.trainer.simulation.epyt import EpytFlowBackend
    resolver = ForecastTrajectoryResolver(dataset,
        ForecastTrajectoryDataSpec(backend="epanet", cache_root=args.cache_root.resolve(), materialize_missing=True),
        _SessionForecastBackend(EpytFlowBackend(), dataset), hydraulic_catalog=hydraulics)
    print(f"Resolving {scenario.scenario_id}, arsenic {args.arsenic_index}, chlorine {args.chlorine_index}", flush=True)
    reference = resolver._resolve(scenario, arsenic, chlorine)
    trajectory = reference.load()
    export_run(dataset, partition, reference, trajectory, arsenic, chlorine,
               args.arsenic_index, args.chlorine_index, args.output)
    print(f"Trajectory: {reference.cache_dir}")
    print(f"Exported {len(trajectory.time_seconds)} timestamps, {len(trajectory.node_ids)} nodes, "
          f"{len(trajectory.link_ids)} pipes, {len(arsenic.events)} events to {args.output}")
    for event in arsenic.events:
        print(f"  node {event.node_id}: day {event.start_time_seconds / 86400:.3f}--"
              f"{event.end_time_seconds / 86400:.3f}, {event.strength:.1f} mg/min")


if __name__ == "__main__":
    main()
