#!/usr/bin/env python3
"""Materialize the fixed Hanoi forecasting trajectories without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer.data import resolve_dataset
from packages.trainer.forecasting.materialization import ForecastTrajectoryResolver
from packages.trainer.forecasting.runner import _SessionForecastBackend
from packages.trainer.schedules.arsenic import schedule_source_requires_hydraulics
from packages.trainer.simulation import (DEFAULT_HYDRAULIC_CACHE_ROOT,
                                         prepare_hydraulic_outflow_cache)
from packages.trainer.simulation.epyt import EpytFlowBackend
from packages.trainer.simulation.hydraulic_cache import HydraulicCache
from training.datasets.hanoi import forecasting_full
from training.experiments.forecasting_encoder_decoder_lstm import TRAJECTORY_DATA


class _MeasuredResolver(ForecastTrajectoryResolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_cache_validity = {}

    def _cache_is_valid(self, reference):
        valid = super()._cache_is_valid(reference)
        self.initial_cache_validity.setdefault(reference.identity, valid)
        return valid


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _timed_stage(report: dict, path: Path, split_record: dict, name: str):
    record = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
    split_record[name] = record
    _write_report(path, report)
    started = time.perf_counter()
    try:
        yield record
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        record["status"] = "completed"
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        record["elapsed_seconds"] = time.perf_counter() - started
        _write_report(path, report)


def _hydraulic_cache_counts(dataset, split: str, cache_root: Path) -> tuple[int, int]:
    if not schedule_source_requires_hydraulics(dataset.schedule_for(split)):
        return 0, 0
    scenarios = dataset.scenarios_for(split)
    cache = HydraulicCache(cache_root)
    hits = sum(cache.reference(scenario.path).load() is not None for scenario in scenarios)
    return len(scenarios), hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("train", "evaluation", "test", "all"), default="all")
    parser.add_argument("--cache-root", type=Path,
                        help="optional separate root for hydraulics/ and runs/ (use an empty root for a fresh timing run)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    cache_root = args.cache_root.resolve() if args.cache_root else None
    trajectory_spec = (replace(TRAJECTORY_DATA, cache_root=cache_root / "runs")
                       if cache_root else TRAJECTORY_DATA)
    hydraulic_root = (cache_root / "hydraulics" if cache_root
                      else DEFAULT_HYDRAULIC_CACHE_ROOT).resolve()
    trajectory_root = Path(trajectory_spec.cache_root).resolve()
    output_dir = PROJECT_ROOT / "data" / "materialization"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"{timestamp}_hanoi_offline_trajectories_{os.getpid()}.json"
    report = {
        "schema_version": 1,
        "kind": "hanoi-offline-trajectory-materialization",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "requested_split": args.split,
        "workers": args.workers,
        "parallelism": "spawned processes" if args.workers > 1 else "serial",
        "execution_device": "CPU",
        "hydraulic_cache_root": str(hydraulic_root),
        "trajectory_cache_root": str(trajectory_root),
        "hardware": {
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_scope": "stage times include cache lookup and validation; cache misses identify newly generated data",
        "splits": [],
    }
    _write_report(output, report)
    started = time.perf_counter()
    splits = ("train", "evaluation", "test") if args.split == "all" else (args.split,)
    try:
        dataset = resolve_dataset(forecasting_full)
        steps_per_trajectory = dataset.duration_seconds // dataset.hydraulic_timestep_seconds
        report["dataset"] = {
            "network": dataset.network,
            "duration_seconds": dataset.duration_seconds,
            "hydraulic_timestep_seconds": dataset.hydraulic_timestep_seconds,
            "simulator_steps_per_trajectory": steps_per_trajectory,
        }
        _write_report(output, report)
        for split in splits:
            split_record = {"split": "validation" if split == "evaluation" else split}
            report["splits"].append(split_record)
            with _timed_stage(report, output, split_record, "hydraulic_cache_inventory") as inventory:
                count, hits = _hydraulic_cache_counts(dataset, split, hydraulic_root)
                inventory.update(scenario_count=count, cache_hits_before=hits,
                                 cache_misses_before=count - hits)

            backend = EpytFlowBackend(hydraulic_cache_root=hydraulic_root)
            with _timed_stage(report, output, split_record, "hydraulics") as hydraulics:
                catalog = prepare_hydraulic_outflow_cache(
                    dataset, backend, workers=args.workers, splits=(split,),
                    description=f"{split} hydraulics",
                )
                hydraulics.update(scenario_count=count, generated_count=count - hits,
                                  cache_hits_before=hits)

            resolver = _MeasuredResolver(
                dataset, trajectory_spec, _SessionForecastBackend(backend, dataset),
                hydraulic_catalog=catalog,
            )
            with _timed_stage(report, output, split_record, "trajectories") as trajectories:
                try:
                    validated = sum(1 for _ in resolver.references(split, workers=args.workers))
                finally:
                    validity = resolver.initial_cache_validity.values()
                    trajectories.update(cache_hits_before=sum(validity),
                                        cache_misses_before=sum(not valid for valid in validity))
                trajectories.update(validated_count=validated,
                                    generated_count=trajectories["cache_misses_before"],
                                    generated_simulator_steps=(
                                        trajectories["cache_misses_before"] * steps_per_trajectory
                                    ))
            print(f"{split}: validated {validated} endpoint-inclusive trajectories", flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["hydraulics_seconds"] = sum(
            record.get("hydraulics", {}).get("elapsed_seconds", 0.0) for record in report["splits"]
        )
        report["trajectories_seconds"] = sum(
            record.get("trajectories", {}).get("elapsed_seconds", 0.0) for record in report["splits"]
        )
        report["hydraulic_cache_inventory_seconds"] = sum(
            record.get("hydraulic_cache_inventory", {}).get("elapsed_seconds", 0.0)
            for record in report["splits"]
        )
        report["offline_data_stage_seconds"] = (
            report["hydraulics_seconds"] + report["trajectories_seconds"]
        )
        if report["status"] == "completed":
            report["generated_trajectories"] = sum(
                record["trajectories"]["generated_count"] for record in report["splits"]
            )
            report["generated_simulator_steps"] = sum(
                record["trajectories"]["generated_simulator_steps"] for record in report["splits"]
            )
            report["fresh_generation"] = all(
                record["hydraulics"]["cache_hits_before"] == 0
                and record["trajectories"]["cache_hits_before"] == 0
                for record in report["splits"]
            )
        _write_report(output, report)
        print(f"timing report: {output}", flush=True)


if __name__ == "__main__":
    main()
