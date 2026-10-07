#!/usr/bin/env python3
"""Generate, validate, and merge scenario-wise OOF CEM teacher datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from packages.trainer.experiment import Experiment
from packages.trainer.observations import ControlObservationBuilder
from packages.trainer.serialization import canonical_data, config_hash
from packages.trainer.specs import ForecastTrainingSpec
from tools.analysis.mpc import validate_mpc_teacher_epanet as validator
from tools.mpc import generate_mpc_teacher_dataset_cem as cem
from training.datasets.hanoi import forecasting_full
from training.experiments.forecasting_encoder_decoder_lstm_cross_fit import (
    FOLDS,
    build_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SAMPLES_PER_FOLD = 32
FOLD_DIRECTORY = "folds"


def cross_fit_teacher_configuration(
    experiment: Experiment | None = None,
    *,
    seed: int = cem.DEFAULT_SEED,
) -> dict[str, Any]:
    experiment = build_experiment() if experiment is None else experiment
    fold_jobs = experiment.jobs[: len(FOLDS)]
    observation_spec = cem.V89_OBSERVATION
    observation_size = observation_spec.factory(
        **observation_spec.as_dict()
    ).observation_size(tuple(f"node-{index}" for index in range(31)))
    return {
        "schema_version": cem.DATASET_SCHEMA_VERSION,
        "source_dataset": {
            "network": forecasting_full.network,
            "split": canonical_data(forecasting_full.train),
        },
        "forecast": {
            "strategy": "scenario-out-of-fold",
            "history_steps": cem.HISTORY_STEPS,
            "horizon_steps": cem.HORIZON_STEPS,
            "folds": [
                {
                    "index": fold.index,
                    "held_out": canonical_data(fold.held_out),
                    "job_hash": config_hash(job),
                }
                for fold, job in zip(FOLDS, fold_jobs, strict=True)
            ],
        },
        "selection": {
            "state_count": None,
            "queries_per_trajectory": cem.DEFAULT_QUERIES_PER_TRAJECTORY,
            "seed": seed,
        },
        "action_contract": cem.action_contract(),
        "policy_observation_contract": {
            "spec": canonical_data(observation_spec),
            "observation_size": observation_size,
            "max_injection": cem.MAX_INJECTION,
        },
        "reward": cem.reward_contract(cem.DEFAULT_DISCOUNT_FACTOR),
        "optimizer": {
            "kind": "cem",
            "settings": cem.cem_optimizer_settings(
                cem.DEFAULT_CANDIDATES,
                cem.DEFAULT_ITERATIONS,
                cem.DEFAULT_ELITE_FRACTION,
            ),
        },
    }


DEFAULT_TEACHER_CONFIGURATION = cross_fit_teacher_configuration()
DEFAULT_TEACHER_CONFIGURATION_HASH = config_hash(DEFAULT_TEACHER_CONFIGURATION)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    f"data/datasets/mpc-teacher-cem/{DEFAULT_TEACHER_CONFIGURATION_HASH}"
)


@dataclass(frozen=True)
class CompletedForecast:
    job: ForecastTrainingSpec
    run_dir: Path
    bundle: Path
    metrics_path: Path
    bundle_manifest: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FoldArtifact:
    index: int
    dataset_path: Path
    manifest_path: Path
    validation_dir: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    validation_summary: dict[str, Any]


def expected_run_directory(job: ForecastTrainingSpec, experiment_name: str) -> Path:
    root = Path(job.output.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    group = Path(job.output.group) if job.output.group else Path()
    return root / group / experiment_name / job.name / config_hash(job)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _artifact_path(run_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"completed forecast is missing artifact {label!r}")
    path = Path(value)
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"completed forecast artifact does not exist: {path}")
    return path


def _finite_numeric_leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_finite_numeric_leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_finite_numeric_leaf_count(item) for item in value)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ValueError("forecast metrics contain a non-finite number")
        return 1
    return 0


def preflight_forecast(job: ForecastTrainingSpec, experiment_name: str) -> CompletedForecast:
    run_dir = expected_run_directory(job, experiment_name)
    status = _read_json(run_dir / "status.json")
    if status.get("status") != "completed":
        raise ValueError(f"forecast job is not completed: {job.name}: {run_dir}")
    artifacts = _read_json(run_dir / "artifacts.json").get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"forecast artifacts manifest is invalid: {run_dir}")
    bundle = _artifact_path(run_dir, artifacts.get("best"), "best")
    metrics_path = _artifact_path(run_dir, artifacts.get("metrics"), "metrics")
    for required in ("manifest.json", "model.pt", "normalization.json"):
        if not (bundle / required).is_file():
            raise FileNotFoundError(f"forecast bundle is incomplete: {bundle / required}")
    bundle_manifest = _read_json(bundle / "manifest.json")
    if bundle_manifest.get("job_hash") != config_hash(job):
        raise ValueError(f"forecast bundle job hash mismatch: {job.name}")
    if bundle_manifest.get("job", {}).get("dataset") != canonical_data(job.dataset):
        raise ValueError(f"forecast bundle dataset mismatch: {job.name}")
    metrics = _read_json(metrics_path)
    final = metrics.get("final")
    if not isinstance(final, dict):
        raise ValueError(f"forecast metrics lack final results: {job.name}")
    for split in ("evaluation", "test"):
        if split not in final or _finite_numeric_leaf_count(final[split]) == 0:
            raise ValueError(f"forecast metrics lack finite {split} results: {job.name}")
    return CompletedForecast(job, run_dir, bundle, metrics_path, bundle_manifest, metrics)


def preflight_all_forecasts(
    experiment: Experiment | None = None,
) -> tuple[CompletedForecast, ...]:
    experiment = build_experiment() if experiment is None else experiment
    if len(experiment.jobs) != len(FOLDS):
        raise ValueError(
            f"cross-fit experiment must contain exactly {len(FOLDS)} fold jobs"
        )
    for fold, job in zip(FOLDS, experiment.jobs, strict=True):
        if (
            job.name != f"encoder-decoder-lstm-fold-{fold.index}"
            or job.dataset != fold.forecasting
        ):
            raise ValueError(
                f"cross-fit forecast job {fold.index} does not match its fold"
            )
    return tuple(
        preflight_forecast(job, experiment.name) for job in experiment.jobs
    )


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            return {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid teacher dataset archive: {path}") from exc


def _expected_trajectory_count(fold_index: int) -> int:
    split = FOLDS[fold_index].held_out
    arsenic_count = getattr(split.arsenic, "schedule_count", None)
    chlorine_count = getattr(split.chlorine, "schedule_count", None)
    if not isinstance(arsenic_count, int) or not isinstance(chlorine_count, int):
        raise ValueError("cross-fit held-out schedules must be finite fixed sets")
    return len(split.scenarios) * arsenic_count * chlorine_count


def validate_fold_teacher(
    fold_index: int,
    dataset_path: Path,
    manifest_path: Path,
    forecast: CompletedForecast,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fold = FOLDS[fold_index]
    manifest = _read_json(manifest_path)
    arrays = _load_arrays(dataset_path)
    missing = set(cem.REQUIRED_DATASET_ARRAYS) - set(arrays)
    if missing:
        raise ValueError(f"fold {fold_index} teacher arrays are missing: {sorted(missing)}")
    source = manifest.get("source_dataset", {})
    if source.get("network") != forecasting_full.network or source.get(
        "contract"
    ) != canonical_data(fold.held_out):
        raise ValueError(f"fold {fold_index} teacher source contract mismatch")
    if manifest.get("forecast_job_hash") != config_hash(forecast.job):
        raise ValueError(f"fold {fold_index} teacher forecast hash mismatch")
    if manifest.get("forecast_training_dataset_contract") != canonical_data(
        forecast.job.dataset
    ):
        raise ValueError(f"fold {fold_index} teacher forecast dataset mismatch")
    if Path(str(manifest.get("forecasting_bundle", ""))).resolve() != forecast.bundle.resolve():
        raise ValueError(f"fold {fold_index} teacher bundle path mismatch")
    identity = manifest.get("teacher_configuration")
    if not isinstance(identity, dict) or manifest.get(
        "teacher_configuration_hash"
    ) != config_hash(identity):
        raise ValueError(f"fold {fold_index} teacher configuration hash is stale")
    for name, expected in (
        ("action_contract", cem.action_contract()),
        ("reward", cem.reward_contract(cem.DEFAULT_DISCOUNT_FACTOR)),
    ):
        if canonical_data(manifest.get(name)) != canonical_data(expected):
            raise ValueError(f"fold {fold_index} teacher {name} differs")
    policy = manifest.get("policy_observation_contract", {})
    node_ids = tuple(str(value) for value in manifest.get("node_ids", ()))
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError(f"fold {fold_index} teacher node order is invalid")
    observation_spec = cem.V89_OBSERVATION
    expected_observation_size = ControlObservationBuilder(
        observation_spec,
        node_ids,
        cem.MAX_INJECTION,
    ).shape[0]
    if (
        policy.get("spec") != canonical_data(observation_spec)
        or policy.get("history_steps") != cem.HISTORY_STEPS
        or policy.get("observation_size") != expected_observation_size
        or policy.get("max_injection") != cem.MAX_INJECTION
    ):
        raise ValueError(f"fold {fold_index} teacher observation contract differs")
    oof = manifest.get("out_of_fold", {})
    if (
        oof.get("required") is not True
        or oof.get("source_scenarios") != list(fold.held_out.scenarios)
        or oof.get("forecast_training_scenarios")
        != list(forecast.job.dataset.train.scenarios)
    ):
        raise ValueError(f"fold {fold_index} teacher is not marked out-of-fold")
    held_out = set(fold.held_out.scenarios)
    if held_out & set(forecast.job.dataset.train.scenarios):
        raise ValueError(f"fold {fold_index} forecast training contains held-out scenarios")
    rows = len(arrays["scenario_id"])
    if any(value.ndim == 0 or value.shape[0] != rows for value in arrays.values()):
        raise ValueError(f"fold {fold_index} teacher arrays have inconsistent rows")
    observation_size = int(policy["observation_size"])
    if arrays["policy_observation_history"].shape != (
        rows,
        cem.HISTORY_STEPS,
        observation_size,
    ):
        raise ValueError(f"fold {fold_index} policy observation array differs")
    if arrays["best_raw_actions"].shape != (rows, cem.HORIZON_STEPS):
        raise ValueError(f"fold {fold_index} action-plan array differs")
    if arrays["predicted_chlorine"].ndim != 3 or arrays[
        "predicted_chlorine"
    ].shape != (rows, cem.HORIZON_STEPS, len(node_ids)):
        raise ValueError(f"fold {fold_index} forecast array differs")
    declared_arrays = manifest.get("arrays")
    if not isinstance(declared_arrays, dict) or set(declared_arrays) != set(arrays):
        raise ValueError(f"fold {fold_index} teacher array manifest differs")
    for name, array in arrays.items():
        if declared_arrays.get(name) != {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }:
            raise ValueError(
                f"fold {fold_index} teacher array metadata differs for {name}"
            )
    scenario_ids = [str(value) for value in arrays["scenario_id"]]
    if set(scenario_ids) != held_out:
        raise ValueError(f"fold {fold_index} teacher scenarios do not match held-out fold")
    expected_trajectories = _expected_trajectory_count(fold_index)
    identities = [str(value) for value in arrays["trajectory_identity"]]
    if len(set(identities)) != expected_trajectories:
        raise ValueError(f"fold {fold_index} teacher trajectory coverage is incomplete")
    counts = Counter(identities)
    if set(counts.values()) != {cem.DEFAULT_QUERIES_PER_TRAJECTORY}:
        raise ValueError(f"fold {fold_index} teacher query coverage is incomplete")
    expected_rows = expected_trajectories * cem.DEFAULT_QUERIES_PER_TRAJECTORY
    if rows != expected_rows:
        raise ValueError(f"fold {fold_index} teacher row count is {rows}, expected {expected_rows}")
    keys = list(zip(identities, arrays["query_index"].astype(np.int64), strict=True))
    if len(set(keys)) != rows:
        raise ValueError(f"fold {fold_index} teacher contains duplicate queries")
    identity_scenarios: dict[str, str] = {}
    for scenario, identity in zip(scenario_ids, identities, strict=True):
        previous = identity_scenarios.setdefault(identity, scenario)
        if previous != scenario:
            raise ValueError(
                f"fold {fold_index} trajectory identity spans multiple scenarios"
            )
    expected_per_scenario = expected_trajectories // len(held_out)
    unique_per_scenario = Counter(identity_scenarios.values())
    if set(unique_per_scenario) != held_out or set(unique_per_scenario.values()) != {
        expected_per_scenario
    }:
        raise ValueError(f"fold {fold_index} per-scenario trajectory coverage differs")
    return arrays, manifest


def validate_fold_replay(
    fold_index: int,
    validation_dir: Path,
) -> dict[str, Any]:
    summary = _read_json(validation_dir / "summary.json")
    if (
        summary.get("selected_count") != VALIDATION_SAMPLES_PER_FOLD
        or summary.get("valid_count") != VALIDATION_SAMPLES_PER_FOLD
        or summary.get("failed_count") != 0
        or summary.get("cancelled_count") != 0
        or summary.get("interrupted") is not False
    ):
        raise ValueError(f"fold {fold_index} EPANET validation did not fully pass")
    cases = _load_arrays(validation_dir / "validation_cases.npz")
    scenario_ids = [str(value) for value in cases.get("scenario_id", ())]
    expected = set(FOLDS[fold_index].held_out.scenarios)
    if set(scenario_ids) != expected or set(Counter(scenario_ids).values()) != {2}:
        raise ValueError(
            f"fold {fold_index} validation must contain two cases per held-out scenario"
        )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forecast_bundle_hashes(bundle: Path) -> dict[str, str]:
    return {
        name: _sha256(bundle / name)
        for name in ("manifest.json", "model.pt", "normalization.json")
    }


def _common_array_contract(folds: Iterable[FoldArtifact]) -> None:
    values = tuple(folds)
    reference = values[0]
    ignored_manifest_fields = {
        "arrays",
        "source_dataset",
        "forecasting_run",
        "forecasting_bundle",
        "forecast_job_hash",
        "forecast_training_dataset_contract",
        "teacher_configuration",
        "teacher_configuration_hash",
        "selection",
        "out_of_fold",
        "seed",
        "device",
        "execution",
    }
    reference_contract = {
        key: value
        for key, value in reference.manifest.items()
        if key not in ignored_manifest_fields
    }
    for fold in values[1:]:
        contract = {
            key: value
            for key, value in fold.manifest.items()
            if key not in ignored_manifest_fields
        }
        if canonical_data(contract) != canonical_data(reference_contract):
            raise ValueError(f"fold {fold.index} teacher manifest contract differs")
        if set(fold.arrays) != set(reference.arrays):
            raise ValueError(f"fold {fold.index} teacher array names differ")
        for name in reference.arrays:
            left, right = reference.arrays[name], fold.arrays[name]
            variable_width_bytes = (
                name == "reward_state_zlib_base64"
                and left.ndim == right.ndim == 1
                and left.dtype.kind == right.dtype.kind == "S"
            )
            if left.shape[1:] != right.shape[1:] or (
                left.dtype != right.dtype and not variable_width_bytes
            ):
                raise ValueError(f"fold {fold.index} array contract differs for {name}")


def merge_fold_artifacts(
    folds: tuple[FoldArtifact, ...],
    *,
    teacher_configuration: dict[str, Any] = DEFAULT_TEACHER_CONFIGURATION,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(folds) != len(FOLDS) or tuple(item.index for item in folds) != tuple(
        range(len(FOLDS))
    ):
        raise ValueError("fold artifacts must be complete and ordered")
    _common_array_contract(folds)
    scenario_order = {
        scenario: index for index, scenario in enumerate(forecasting_full.train.scenarios)
    }
    rows: list[tuple[int, str, int, int, int]] = []
    observed_keys: set[tuple[str, int]] = set()
    for fold in folds:
        identities = fold.arrays["trajectory_identity"]
        queries = fold.arrays["query_index"]
        scenarios = fold.arrays["scenario_id"]
        for row in range(len(scenarios)):
            scenario = str(scenarios[row])
            identity = str(identities[row])
            query = int(queries[row])
            key = (identity, query)
            if key in observed_keys:
                raise ValueError(f"duplicate cross-fit teacher query: {key}")
            observed_keys.add(key)
            rows.append((scenario_order[scenario], identity, query, fold.index, row))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    merged: dict[str, np.ndarray] = {}
    for name in folds[0].arrays:
        merged[name] = np.stack(
            [folds[fold_index].arrays[name][row] for *_key, fold_index, row in rows]
        )
    merged["fold_index"] = np.asarray(
        [fold_index for *_key, fold_index, _row in rows], dtype=np.int8
    )

    first = folds[0].manifest
    labeling_inputs = {
        "teacher_configuration": teacher_configuration,
        "folds": [
            {
                "index": fold.index,
                "forecast_job_hash": fold.manifest["forecast_job_hash"],
                "forecast_bundle_hashes": _forecast_bundle_hashes(
                    Path(fold.manifest["forecasting_bundle"])
                ),
                "teacher_configuration_hash": fold.manifest[
                    "teacher_configuration_hash"
                ],
            }
            for fold in folds
        ],
    }
    manifest = {
        key: value
        for key, value in first.items()
        if key
        not in {
            "arrays",
            "source_dataset",
            "forecasting_run",
            "forecasting_bundle",
            "forecast_job_hash",
            "forecast_training_dataset_contract",
            "teacher_configuration",
            "teacher_configuration_hash",
            "selection",
            "out_of_fold",
            "seed",
            "device",
            "execution",
        }
    }
    manifest.update(
        {
            "teacher_configuration": teacher_configuration,
            "teacher_configuration_hash": config_hash(teacher_configuration),
            "aggregate_teacher_hash": config_hash(labeling_inputs),
            "source_dataset": {
                "definition": cem.DEFAULT_SOURCE_DEFINITION,
                "network": forecasting_full.network,
                "contract": canonical_data(forecasting_full.train),
            },
            "selection": {
                "strategy": "complete-scenario-out-of-fold-union",
                "state_count": len(rows),
                "trajectory_count": sum(
                    len(set(str(value) for value in fold.arrays["trajectory_identity"]))
                    for fold in folds
                ),
                "queries_per_trajectory": cem.DEFAULT_QUERIES_PER_TRAJECTORY,
                "selection_seed": teacher_configuration["selection"]["seed"],
            },
            "cross_fitting": {
                "strategy": "scenario-round-robin-declared-order",
                "fold_count": len(folds),
                "reference_dataset": "training.datasets.hanoi.forecasting_full",
                "folds": [
                    {
                        "index": fold.index,
                        "held_out_scenarios": list(FOLDS[fold.index].held_out.scenarios),
                        "labeling_forecast_job_hash": fold.manifest["forecast_job_hash"],
                        "labeling_forecasting_bundle": fold.manifest["forecasting_bundle"],
                        "labeling_forecast_bundle_hashes": labeling_inputs["folds"][
                            fold.index
                        ]["forecast_bundle_hashes"],
                        "teacher_configuration_hash": fold.manifest[
                            "teacher_configuration_hash"
                        ],
                        "dataset_sha256": _sha256(fold.dataset_path),
                        "manifest_sha256": _sha256(fold.manifest_path),
                        "validation_summary": str(
                            (fold.validation_dir / "summary.json").resolve()
                        ),
                        "validation_summary_sha256": _sha256(
                            fold.validation_dir / "summary.json"
                        ),
                        "validation_cases_sha256": _sha256(
                            fold.validation_dir / "validation_cases.npz"
                        ),
                    }
                    for fold in folds
                ],
            },
        }
    )
    return merged, manifest


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_merged_dataset(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.npz"
    manifest_path = output_dir / "manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite merged teacher output: {output_dir}")
    counts = {value.shape[0] for value in arrays.values() if value.ndim > 0}
    if len(counts) != 1 or any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("merged teacher arrays have an invalid storage contract")
    complete = dict(manifest)
    complete["arrays"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    _atomic_npz(dataset_path, arrays)
    try:
        _atomic_json(manifest_path, complete)
    except BaseException:
        dataset_path.unlink(missing_ok=True)
        raise
    return dataset_path, manifest_path


def _aggregate_numeric(values: list[Any]) -> Any:
    if values and all(isinstance(value, dict) for value in values):
        common = set(values[0]).intersection(*(set(value) for value in values[1:]))
        return {
            key: aggregated
            for key in sorted(common)
            if (aggregated := _aggregate_numeric([value[key] for value in values]))
            is not None
        }
    if values and all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in values
    ):
        numeric = np.asarray(values, dtype=np.float64)
        if np.all(np.isfinite(numeric)):
            return {"mean": float(np.mean(numeric)), "std": float(np.std(numeric))}
    return None


def forecast_metrics_report(folds: tuple[CompletedForecast, ...]) -> dict[str, Any]:
    fold_final = [item.metrics["final"] for item in folds]
    return {
        "schema_version": 2,
        "fold_job_hashes": [config_hash(item.job) for item in folds],
        "fold_validation_test_mean_std": _aggregate_numeric(fold_final),
    }


def _generation_args(args: argparse.Namespace, run: CompletedForecast, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        forecasting_run=run.run_dir,
        trajectory_cache=args.trajectory_cache,
        output_dir=output,
        state_count=None,
        queries_per_trajectory=cem.DEFAULT_QUERIES_PER_TRAJECTORY,
        candidates=cem.DEFAULT_CANDIDATES,
        iterations=cem.DEFAULT_ITERATIONS,
        elite_fraction=cem.DEFAULT_ELITE_FRACTION,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        discount_factor=cem.DEFAULT_DISCOUNT_FACTOR,
    )


def _validation_args(args: argparse.Namespace, fold_dir: Path, run: CompletedForecast) -> argparse.Namespace:
    teacher_dir = fold_dir / "teacher"
    return argparse.Namespace(
        sample_count=VALIDATION_SAMPLES_PER_FOLD,
        seed=args.seed + int(fold_dir.name.split("-")[-1]),
        workers=args.validation_workers,
        device=args.device,
        dataset_npz=teacher_dir / "dataset.npz",
        dataset_manifest=teacher_dir / "manifest.json",
        trajectory_cache=args.trajectory_cache,
        forecast_bundle=run.bundle,
        bootstrap_samples=2_000,
        replay_tolerance=1e-5,
        tie_tolerance=1e-6,
        shutdown_timeout=10.0,
        output_dir=fold_dir / "validation",
    )


def execute_pipeline(
    args: argparse.Namespace,
    *,
    experiment: Experiment | None = None,
    measure_fold: Callable[[int], AbstractContextManager[Any]] | None = None,
) -> tuple[Path, Path]:
    experiment = build_experiment() if experiment is None else experiment
    teacher_configuration = cross_fit_teacher_configuration(
        experiment,
        seed=args.seed,
    )
    fold_runs = preflight_all_forecasts(experiment)
    output_dir = (
        PROJECT_ROOT
        / "data/datasets/mpc-teacher-cem"
        / config_hash(teacher_configuration)
        if args.output_dir is None
        else Path(args.output_dir).resolve()
    )
    fold_artifacts: list[FoldArtifact] = []
    for fold, run in zip(FOLDS, fold_runs, strict=True):
        # Include label generation, artifact checks, and EPANET replay for each fold.
        with nullcontext() if measure_fold is None else measure_fold(fold.index):
            fold_dir = output_dir / FOLD_DIRECTORY / f"fold-{fold.index:02d}"
            teacher_dir = fold_dir / "teacher"
            dataset_path, manifest_path = cem.generate_dataset(
                _generation_args(args, run, teacher_dir),
                forecast_job=run.job,
                source_split=fold.held_out,
                source_definition=(
                    "training.experiments.forecasting_encoder_decoder_lstm_cross_fit."
                    f"FOLDS[{fold.index}].held_out"
                ),
                require_out_of_fold=True,
            )
            arrays, manifest = validate_fold_teacher(
                fold.index, dataset_path, manifest_path, run
            )
            validation_args = _validation_args(args, fold_dir, run)
            validator.ensure_fresh_output(validation_args.output_dir)
            cases, summary, interrupted = validator.run_validation(
                validation_args,
                validator.TerminationState(),
                forecast_job=run.job,
                source_split=fold.held_out,
            )
            validator._atomic_npz(
                validation_args.output_dir / "validation_cases.npz",
                validator.cases_to_arrays(cases),
            )
            validator._atomic_json(validation_args.output_dir / "summary.json", summary)
            validator.write_plots(validation_args.output_dir, cases)
            if interrupted:
                raise KeyboardInterrupt
            validation_summary = validate_fold_replay(
                fold.index, validation_args.output_dir
            )
            fold_artifacts.append(
                FoldArtifact(
                    fold.index,
                    dataset_path,
                    manifest_path,
                    validation_args.output_dir,
                    manifest,
                    arrays,
                    validation_summary,
                )
            )
    merged_arrays, merged_manifest = merge_fold_artifacts(
        tuple(fold_artifacts),
        teacher_configuration=teacher_configuration,
    )
    dataset_path, manifest_path = write_merged_dataset(
        output_dir, merged_arrays, merged_manifest
    )
    _atomic_json(
        output_dir / "forecast_metrics.json",
        forecast_metrics_report(fold_runs),
    )
    return dataset_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: content-addressed from the forecast jobs and teacher seed",
    )
    parser.add_argument(
        "--trajectory-cache", type=Path, default=cem.DEFAULT_TRAJECTORY_CACHE
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=cem.DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.validation_workers <= 0:
        raise SystemExit("worker counts must be positive")
    if args.seed < 0:
        raise SystemExit("seed must be non-negative")
    try:
        dataset_path, manifest_path = execute_pipeline(args)
    except KeyboardInterrupt:
        return 130
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"cross-fit teacher generation failed: {exc}")
        return 2
    print(f"dataset: {dataset_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
