#!/usr/bin/env python3
"""Cache chlorine-free Control validation trajectories and export three raw examples."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

# EPANET-MSX and NumPy load separate native runtimes on macOS. Initialize
# their thread limits before importing either library, including spawn workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.data import resolve_dataset_split
from packages.trainer.forecasting.materialization import ForecastTrajectoryResolver
from packages.trainer.schedules import fixed_schedule
from packages.trainer.simulation import Trajectory
from packages.trainer.specs import ForecastTrajectoryDataSpec
from training.datasets.hanoi import control_full

HOUR = 3600
CLEAN_TOLERANCE = 1e-6


class ZeroTrajectoryResolver(ForecastTrajectoryResolver):
    """Keep the final clearance sample in analysis caches, including in workers."""

    def _validate_trajectory(self, trajectory, reference):
        self._validate_identity(trajectory, reference)
        expected = np.arange(0, self.dataset.duration_seconds + 1,
                             self.dataset.hydraulic_timestep_seconds)
        if not np.array_equal(trajectory.time_seconds, expected):
            raise ValueError("Expected the complete analysis grid including the endpoint")
        validate_zero(trajectory)

    def _materialize_parallel(self, missing, workers, progress, interrupted):
        with ProcessPoolExecutor(max_workers=min(workers, len(missing)),
                                 mp_context=mp.get_context("spawn")) as pool:
            pending = iter(missing)
            futures = set()

            def submit_next():
                if interrupted():
                    return
                item = next(pending, None)
                if item is not None:
                    _, scenario, arsenic, chlorine = item
                    futures.add(pool.submit(_materialize_zero, self.dataset, self.spec,
                                            scenario, arsenic, chlorine))

            for _ in range(min(workers, len(missing))):
                submit_next()
            while futures:
                ready = next(as_completed(futures))
                ready.result()
                futures.remove(ready)
                progress.update()
                submit_next()


def _materialize_zero(dataset, spec, scenario, arsenic, chlorine):
    return ZeroTrajectoryResolver(dataset, spec, ZeroBackend(dataset))._resolve(
        scenario, arsenic, chlorine)


def validate_zero(trajectory):
    for name in ("injection", "chlorine", "arsenic"):
        values = np.asarray(getattr(trajectory, name))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid {name} values in {trajectory.scenario_id}")
        if name != "arsenic" and np.any(values != 0):
            raise ValueError(f"Expected exactly zero {name} in {trajectory.scenario_id}")


@dataclass
class ZeroBackend:
    dataset: object

    def simulate(self, scenario, arsenic_schedule, chlorine_schedule, **kwargs):
        from packages.trainer.simulation.epyt import EpytFlowBackend
        session = EpytFlowBackend().create_session(
            str(scenario.path), hydraulic_timestep_seconds=self.dataset.hydraulic_timestep_seconds,
            calendar_start_day=scenario.calendar_start_day,
            chlorine_source_ids=scenario.chlorine_source_ids,
            monitored_node_ids=scenario.monitored_node_ids)
        try:
            samples = [session.reset(arsenic_schedule.events)]
            for _ in range(self.dataset.duration_seconds // self.dataset.hydraulic_timestep_seconds):
                sample, done = session.step(0.0, 0.0)
                samples.append(sample)
                if done:
                    break
            if samples[-1].time_seconds != self.dataset.duration_seconds:
                raise ValueError("Simulation did not complete the full validation episode")
            arrays = {name: np.asarray([getattr(s, name) for s in samples]) for name in (
                "time_seconds", "pressure", "demand", "flow", "chlorine", "arsenic",
                "injection", "event_active", "event_strength_by_node", "hydraulic_outflow")}
            result = Trajectory(
                network=self.dataset.network, scenario_id=scenario.scenario_id,
                calendar_start_day=scenario.calendar_start_day,
                chlorine_profile_hash=chlorine_schedule.hash,
                arsenic_schedule_hash=arsenic_schedule.hash,
                arsenic_schedule_seed=self.dataset.schedule.seed,
                node_ids=session.node_ids, link_ids=session.link_ids,
                source_ids=scenario.chlorine_source_ids,
                actions=np.zeros(len(samples)), simulator_backend="epanet",
                flow_units=session.flow_units,
                reproducibility=(("policy", "zero-injection"),
                                 ("events", [asdict(e) for e in arsenic_schedule.events]),
                                 ("simulator_parameters", kwargs["simulator_parameters"])),
                **arrays)
            validate_zero(result)
            return result
        finally:
            session.close()


def input_fingerprint(dataset):
    digest = hashlib.sha256()
    for scenario in dataset.scenarios:
        for name in ("scenario.inp", "scenario.msx", "scenario.epytflow_scenario_config", "manifest.json"):
            digest.update(f"{scenario.scenario_id}/{name}".encode())
            digest.update((scenario.path / name).read_bytes())
    digest.update((PROJECT_ROOT / "packages/trainer/simulation/epyt.py").read_bytes())
    return digest.hexdigest()


def materialize(cache_root, workers):
    dataset = resolve_dataset_split(control_full.network, control_full.validation, split="evaluation",
        scenarios_dir=control_full.scenarios_dir, uncertainty_seed=control_full.uncertainty_seed)
    spec = ForecastTrajectoryDataSpec(
        backend="epanet", cache_root=cache_root, materialize_missing=True,
        simulator_parameters={"analysis_contract": "zero-control-validation-v1-full-endpoint",
                              "input_sha256": input_fingerprint(dataset)})
    resolver = ZeroTrajectoryResolver(dataset, spec, ZeroBackend(dataset))
    references = list(resolver.references("evaluation", workers=workers,
                                         description="Zero-chlorine Control validation"))
    by_key = {(ref.scenario_id, ref.arsenic_schedule_hash): ref for ref in references}
    runs = []
    # Match the Control evaluator: schedule index first, then scenario index.
    for schedule_index in range(dataset.schedule.schedule_count):
        for scenario in dataset.scenarios:
            schedule = fixed_schedule(
                dataset.schedule, scenario.scenario_id, schedule_index,
                scenario.eligible_arsenic_node_ids,
                hydraulic_profile=resolver.hydraulic_catalog.profile(scenario.scenario_id))
            reference = by_key[(scenario.scenario_id, schedule.hash)]
            trajectory = reference.load()
            validate_zero(trajectory)
            if not np.array_equal(trajectory.time_seconds, np.arange(
                    0, dataset.duration_seconds + 1, dataset.hydraulic_timestep_seconds)):
                raise ValueError("Cached trajectory has incomplete timestamps")
            runs.append((dict(index=len(runs), scenario_id=scenario.scenario_id,
                              schedule_index=schedule_index, schedule_hash=schedule.hash,
                              cache_dir=str(reference.cache_dir.relative_to(PROJECT_ROOT)
                                            if reference.cache_dir.is_relative_to(PROJECT_ROOT)
                                            else reference.cache_dir),
                              events=[asdict(e) for e in schedule.events]), trajectory))
    return runs


def clean_window(trajectory, events, selected):
    """Isolate selected sources in time and reject residual contamination."""
    start = min(events[i]["start_time_seconds"] for i in selected)
    end = max(events[i]["end_time_seconds"] for i in selected)
    left = max(0, start - 6 * HOUR)
    right = min(float(trajectory.time_seconds[-1]), end + 48 * HOUR)
    for i, other in enumerate(events):
        if i in selected:
            continue
        if other["start_time_seconds"] < end and other["end_time_seconds"] > left:
            return None
        if other["start_time_seconds"] >= end:
            right = min(right, other["start_time_seconds"] - 1800)
    if right < end + 6 * HOUR:
        return None
    mask = (trajectory.time_seconds >= left) & (trajectory.time_seconds <= right)
    baseline = mask & (trajectory.time_seconds < start)
    if not np.any(baseline) or np.max(trajectory.arsenic[baseline]) > CLEAN_TOLERANCE:
        return None
    # Reject windows cut short by another event or the episode boundary while
    # a visible contamination tail is still present at any junction.
    values = trajectory.arsenic[mask]
    if np.any(values[-1] > np.maximum(0.0001, values.max(axis=0) * 0.01)):
        return None
    return mask, start


def candidates(record, trajectory):
    events = record["events"]
    result = {"mild": [], "strong": [], "overlap": []}
    for i, event in enumerate(events):
        window = clean_window(trajectory, events, (i,))
        if window is None:
            continue
        mask, origin = window
        values = trajectory.arsenic[mask]
        times = trajectory.time_seconds[mask]
        source = trajectory.node_ids.index(event["node_id"])
        peak = float(values[:, source].max())
        if peak < 0.05:
            continue
        common = dict(run_index=record["index"], event_indices=[i],
                      window_seconds=[float(times[0]), float(times[-1])], origin_seconds=origin)
        others = [j for j in range(len(trajectory.node_ids)) if j != source]
        if event["node_id"] in ("1", "13", "22"):
            continue
        downstream = []
        source_arrival = float(times[np.flatnonzero(values[:, source] >= max(0.01, peak * 0.05))[0]])
        for j in others:
            target_peak = float(values[:, j].max())
            if target_peak < 0.05 or not 0.05 <= target_peak / peak <= 0.9:
                continue
            arrival = float(times[np.flatnonzero(values[:, j] >= max(0.01, target_peak * 0.05))[0]])
            delay = (arrival - source_arrival) / HOUR
            if delay >= 1:
                downstream.append(dict(node=trajectory.node_ids[j], delay_hours=delay,
                                       peak_mg_l=target_peak, peak_ratio=target_peak / peak,
                                       relative_exposure=float(values[:, j].sum() / values[:, source].sum())))
        delayed = [d for d in downstream if d["delay_hours"] >= 2]
        if delayed:
            # Favor sustained, visible transport rather than a very late isolated spike.
            def visibility(d):
                return min(d["delay_hours"], 12) * min(d["peak_mg_l"], 1) * d["relative_exposure"]
            delayed.sort(key=visibility, reverse=True)
            selected = sorted(delayed[:2], key=lambda d: d["delay_hours"])
            candidate = dict(common, nodes=[event["node_id"]] + [d["node"] for d in selected],
                score=visibility(delayed[0]),
                diagnostics={"source_peak_mg_l": peak, "downstream": selected,
                             "arrival_definition": "first sample >= max(0.01 mg/L, 5% of node peak)"})
            result["mild"].append(candidate)
            downstream.sort(key=visibility, reverse=True)
            selected = sorted(downstream[:4], key=lambda d: d["delay_hours"])
            result["strong"].append(dict(candidate,
                nodes=[event["node_id"]] + [d["node"] for d in selected],
                diagnostics=dict(candidate["diagnostics"], downstream=selected)))
    for a, b in itertools.combinations(range(len(events)), 2):
        first, second = events[a], events[b]
        overlap = min(first["end_time_seconds"], second["end_time_seconds"]) - max(
            first["start_time_seconds"], second["start_time_seconds"])
        if overlap < HOUR or first["node_id"] == second["node_id"]:
            continue
        window = clean_window(trajectory, events, (a, b))
        if window is None:
            continue
        mask, origin = window
        values = trajectory.arsenic[mask]
        sources = [first["node_id"], second["node_id"]]
        if min(float(values[:, trajectory.node_ids.index(n)].max()) for n in sources) < 0.05:
            continue
        others = sorted((n for n in trajectory.node_ids if n not in sources),
                        key=lambda n: float(values[:, trajectory.node_ids.index(n)].max()), reverse=True)
        downstream = [n for n in others if values[:, trajectory.node_ids.index(n)].max() >= 0.05][:1]
        result["overlap"].append(dict(run_index=record["index"], event_indices=[a, b],
            window_seconds=[float(trajectory.time_seconds[mask][0]), float(trajectory.time_seconds[mask][-1])],
            origin_seconds=origin, nodes=sources + downstream,
            score=overlap / HOUR + 12 * bool(downstream), diagnostics={"overlap_hours": overlap / HOUR}))
    return result


def select_example(found, kind, requested=None):
    """Compare source peaks within the pool of isolated, visible transport events."""
    if kind in ("mild", "strong") and found:
        quantile = 0.25 if kind == "mild" else 0.75
        boundary = float(np.quantile(
            [candidate["diagnostics"]["source_peak_mg_l"] for candidate in found], quantile))
        found = [candidate for candidate in found if (
            candidate["diagnostics"]["source_peak_mg_l"] <= boundary if kind == "mild"
            else candidate["diagnostics"]["source_peak_mg_l"] >= boundary)]
    found = [candidate for candidate in found
             if requested is None or candidate["run_index"] == requested]
    if kind == "strong":
        broad = [candidate for candidate in found if len(candidate["nodes"]) >= 4]
        if broad:
            found = broad
    return max(found, key=lambda candidate: candidate["score"]) if found else None


def export_examples(runs, choices, output, catalog_path):
    pool = {name: [] for name in choices}
    for record, trajectory in runs:
        for name, found in candidates(record, trajectory).items():
            pool[name].extend(found)
    catalog = {"schema_version": 1, "dataset": "control_full", "split": "validation",
               "policy": "zero-injection", "run_count": len(runs),
               "runs": [record for record, _ in runs], "candidates": pool}
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    panels = []
    for name, requested in choices.items():
        chosen = select_example(pool[name], name, requested)
        if chosen is None:
            raise ValueError(f"No suitable {name} example for run {requested}; inspect {catalog_path}")
        record, trajectory = runs[chosen["run_index"]]
        # Keep the complete cache, but display only the event and its visible tail.
        chosen = dict(chosen)
        origin = chosen["origin_seconds"]
        selected_end = max(record["events"][i]["end_time_seconds"] for i in chosen["event_indices"])
        mask = (trajectory.time_seconds >= origin) & (trajectory.time_seconds <= chosen["window_seconds"][1])
        columns = [trajectory.node_ids.index(n) for n in chosen["nodes"]]
        concentrations = trajectory.arsenic[mask][:, columns]
        thresholds = np.maximum(0.0001, concentrations.max(axis=0) * 0.01)
        visible = np.any(concentrations >= thresholds, axis=1)
        last_visible = float(trajectory.time_seconds[mask][visible][-1])
        end = max(selected_end, last_visible) + 4 * HOUR
        end = origin + np.ceil((end - origin) / (6 * HOUR)) * 6 * HOUR
        chosen["window_seconds"] = [max(0, origin - 3 * HOUR), min(chosen["window_seconds"][1], float(end))]
        panels.append(dict(kind=name, selection=chosen, run=record,
            time_seconds=trajectory.time_seconds.tolist(),
            nodes=[dict(id=node, arsenic=trajectory.arsenic[:, j].tolist())
                   for j, node in enumerate(trajectory.node_ids)]))
        print(f"{name}: run {chosen['run_index']} ({record['scenario_id']}), "
              f"events {chosen['event_indices']}, nodes {chosen['nodes']}, {chosen['diagnostics']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema_version=1, species="AsIII", unit="mg/L",
        timestep_seconds=1800, policy="zero-injection", initial_chlorine_mg_l=0,
        panels=panels), separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported raw node concentrations to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper/dataset/arsenic-examples.json")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/analysis/control-zero-validation/catalog.json")
    count = len(control_full.validation.scenarios) * control_full.validation.arsenic.schedule_count
    for name in ("mild", "strong", "overlap"):
        parser.add_argument(f"--{name}-run", type=int, choices=range(count), metavar=f"0..{count - 1}",
                            help=f"Choose the {name} panel's validation run; default: automatic")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    runs = materialize(args.cache_root.resolve(), args.workers)
    export_examples(runs, {name: getattr(args, f"{name}_run") for name in ("mild", "strong", "overlap")},
                    args.output, args.catalog)


if __name__ == "__main__":
    main()
