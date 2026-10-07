#!/usr/bin/env python3
"""Run selected stages of the replicated V89 principal comparison pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MPC loads PyTorch alongside EPANET; spawned workers inherit these limits.
# Set them before importing either native runtime through the project packages.
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np  # noqa: E402

from packages.trainer import Experiment, OutputSpec  # noqa: E402
from packages.trainer.execution import RunnerRegistry, execute_experiment  # noqa: E402
from packages.trainer.serialization import canonical_data, config_hash  # noqa: E402

CEM_ACTION_STD_SOURCE = "cem-final-iteration-first-action-median"


# Principal-comparison replicates from paper/chapters/06_evaluation/chapter.typ.
SEEDS = (17, 42, 123, 1337, 71026)

REWARD_VERSION = "v89"
PIPELINE_NAME = f"hanoi-principal-{REWARD_VERSION}"
PIPELINE_STATE_ROOT = PROJECT_ROOT / "data/pipelines" / PIPELINE_NAME
STAGES = (
    "forecasting",
    "mpc",
    "behavior-cloning",
    "baseline",
    "fine-tuning",
)


def build_forecasting_experiment(seed: int) -> Experiment:
    from training.experiments.forecasting_encoder_decoder_lstm_cross_fit import (
        build_experiment,
    )

    return build_experiment(training_seed=seed)


def teacher_configuration(experiment: Experiment, seed: int) -> dict[str, Any]:
    from tools.mpc.generate_cross_fit_mpc_teacher_dataset_cem import (
        cross_fit_teacher_configuration,
    )

    return cross_fit_teacher_configuration(experiment, seed=seed)


def validation_teacher_configuration(experiment: Experiment, seed: int) -> dict[str, Any]:
    from tools.mpc.generate_validation_mpc_teacher_dataset_cem import (
        validation_teacher_configuration as configuration,
    )

    return configuration(experiment, seed=seed)


def teacher_output_directory(configuration: dict[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / "data/datasets/mpc-teacher-cem"
        / config_hash(configuration)
    )


def behavior_cloning_configuration(
    seed: int,
    teacher_output: Path,
    validation_output: Path | None = None,
) -> dict[str, Any]:
    """Identity for the BC stage and its content-addressed output directory."""
    teacher_dataset = teacher_output.resolve().relative_to(PROJECT_ROOT.resolve())
    if validation_output is None:
        validation_output = teacher_output_directory(
            validation_teacher_configuration(build_forecasting_experiment(seed), seed)
        )
    validation_dataset = validation_output.resolve().relative_to(PROJECT_ROOT.resolve())
    return {
        "schema_version": 2,
        "kind": "recurrent-ppo-behavior-cloning",
        "reward_version": REWARD_VERSION,
        "seed": seed,
        "teacher_dataset": str(teacher_dataset),
        "validation_teacher_dataset": str(validation_dataset),
        "epochs": 100,
        "trajectory_batch_size": 16,
        "sequence_chunk_length": 48,
        "learning_rate": 3e-4,
        "loss": "mse",
        "initial_action_std_source": CEM_ACTION_STD_SOURCE,
        "patience": 10,
    }


def behavior_cloning_output_directory(configuration: dict[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / "data/weights/control/mpc-teacher-bc"
        / config_hash(configuration)
    )


def build_control_experiment(seed: int, bc_output: Path) -> Experiment:
    """Build the matched baseline and pretrained jobs from one shared template."""
    from training.experiments.control_v89_baseline import (
        build_experiment as build_reference_experiment,
    )

    reference = build_reference_experiment().jobs[0]
    common_algorithm = replace(reference.algorithm, pretrained_policy=None)
    baseline = replace(
        reference,
        name="baseline",
        algorithm=common_algorithm,
        training_seed=seed,
        output=OutputSpec(group=f"control/principal/{REWARD_VERSION}/baseline"),
    )
    fine_tuning = replace(
        reference,
        name="mpc-teacher-bc",
        algorithm=replace(common_algorithm, pretrained_policy=bc_output),
        training_seed=seed,
        output=OutputSpec(
            group=f"control/principal/{REWARD_VERSION}/mpc-teacher-bc"
        ),
    )
    return Experiment(
        name=PIPELINE_NAME,
        jobs=(baseline, fine_tuning),
        description=(
            "Matched ordinary and cross-fitted MPC+BC initialized RecurrentPPO "
            f"training with reward {REWARD_VERSION}."
        ),
    )


def _training_registry(*, on_training_complete=None) -> RunnerRegistry:
    """Keep plots out of pipeline stages without changing checkpoint job hashes."""
    from packages.trainer.control.runner import ControlRunner
    from packages.trainer.forecasting.runner import ForecastRunner

    registry = RunnerRegistry()
    registry.register("forecast", ForecastRunner(render_plots=False))
    registry.register("control", ControlRunner(
        render_plots=False,
        defer_evaluation=True,
        on_training_complete=on_training_complete,
        evaluate_best_model=True,
    ))
    return registry


def _require_matched_teacher_contract(
    teacher: dict[str, Any],
    validation_teacher: dict[str, Any],
    control: Experiment,
) -> None:
    """Require teacher reward and observation contracts to match V89 control."""
    baseline = control.job("baseline")
    expected_reward = {
        "id": baseline.reward.factory.reward_id,
        "parameters": baseline.reward.as_dict(),
    }
    expected_observation = canonical_data(baseline.observation)
    for name, configuration in (
        ("training", teacher),
        ("validation", validation_teacher),
    ):
        reward = configuration["reward"]
        observation = configuration["policy_observation_contract"]
        if (
            reward.get("id") != expected_reward["id"]
            or reward.get("parameters") != expected_reward["parameters"]
            or observation.get("spec") != expected_observation
        ):
            raise RuntimeError(
                f"{name} MPC teacher contract does not match the V89 control "
                "reward and observation; port the MPC and BC stages before "
                "running the principal pretraining pipeline"
            )


def expected_run_directory(experiment: Experiment, job_name: str) -> Path:
    job = experiment.job(job_name)
    root = Path(job.output.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    group = Path(job.output.group) if job.output.group else Path()
    return root / group / experiment.name / job.name / config_hash(job)


def pipeline_plan(seed: int) -> dict[str, Any]:
    forecast = build_forecasting_experiment(seed)
    teacher = teacher_configuration(forecast, seed)
    teacher_output = teacher_output_directory(teacher)
    validation_teacher = validation_teacher_configuration(forecast, seed)
    validation_output = teacher_output_directory(validation_teacher)
    bc = behavior_cloning_configuration(seed, teacher_output, validation_output)
    bc_output = behavior_cloning_output_directory(bc)
    control = build_control_experiment(seed, bc_output)
    return {
        "schema_version": 1,
        "pipeline": PIPELINE_NAME,
        "reward_version": REWARD_VERSION,
        "seed": seed,
        "datasets": {
            "forecasting": "training.datasets.hanoi.forecasting_full",
            "control": "training.datasets.hanoi.control_full",
        },
        "forecasting": {
            "experiment": forecast.name,
            "jobs": [
                {
                    "name": job.name,
                    "training_seed": job.training_seed,
                    "job_hash": config_hash(job),
                }
                for job in forecast.jobs
            ],
        },
        "teacher": {
            "seed": seed,
            "configuration_hash": config_hash(teacher),
            "output": str(teacher_output),
        },
        "validation_teacher": {
            "seed": seed,
            "configuration_hash": config_hash(validation_teacher),
            "output": str(validation_output),
        },
        "behavior_cloning": {
            "seed": seed,
            "configuration_hash": config_hash(bc),
            "output": str(bc_output),
        },
        "control": {
            "experiment": control.name,
            "baseline": {
                "job_hash": config_hash(control.job("baseline")),
                "output": str(expected_run_directory(control, "baseline")),
            },
            "mpc_teacher_bc": {
                "job_hash": config_hash(control.job("mpc-teacher-bc")),
                "output": str(expected_run_directory(control, "mpc-teacher-bc")),
            },
        },
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with open(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_path(seed: int, name: str) -> Path:
    return PIPELINE_STATE_ROOT / f"seed-{seed}" / name


def _set_status(seed: int, phase: str, status: str, **extra: Any) -> None:
    path = _state_path(seed, "status.json")
    previous = _read_json(path)
    started_at = (
        previous.get("started_at")
        if isinstance(previous, dict) and previous.get("started_at")
        else datetime.now(UTC).isoformat()
    )
    _write_json(
        path,
        {
            "schema_version": 1,
            "pipeline": PIPELINE_NAME,
            "seed": seed,
            "phase": phase,
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


@contextmanager
def _measure_stage(
    seed: int, stage: str, *, records: list[dict[str, Any]] | None = None,
    **settings: Any,
):
    """Record wall time, optionally as a component of an enclosing stage."""
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    components: list[dict[str, Any]] = []
    status = "completed"
    error = None
    try:
        yield components
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = str(exc)
        raise
    finally:
        _record_stage_timing(
            seed, stage, started_at, started, status, settings, error,
            components=components, records=records,
        )


def _record_stage_timing(
    seed: int, stage: str, started_at: str, started: float,
    status: str, settings: dict[str, Any], error: str | None = None,
    *, components: list[dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> None:
    record = {
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "settings": settings,
    }
    if error is not None:
        record["error"] = error
    if components:
        record["components"] = components
    if records is not None:
        records.append(record)
        return
    path = _state_path(seed, "stage-timings.json")
    value = _read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        value = {
            "schema_version": 1,
            "pipeline": PIPELINE_NAME,
            "seed": seed,
            "records": [],
        }
    value["records"].append(record)
    _write_json(path, value)


@contextmanager
def _measure_control_stage(seed: int, stage: str, **settings: Any):
    """Split training time from evaluation when the runner finishes learning."""
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    evaluation_started_at = None
    evaluation_started = None

    def training_complete() -> None:
        nonlocal evaluation_started_at, evaluation_started
        if evaluation_started is not None:
            raise RuntimeError("control training completion was reported twice")
        _record_stage_timing(seed, stage, started_at, started, "completed", settings)
        evaluation_started_at = datetime.now(UTC).isoformat()
        evaluation_started = time.perf_counter()

    try:
        yield training_complete
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if evaluation_started is None:
            _record_stage_timing(seed, stage, started_at, started, status, settings, str(exc))
        else:
            _record_stage_timing(seed, f"{stage}-evaluation", evaluation_started_at,
                                 evaluation_started, status, settings, str(exc))
        raise
    else:
        if evaluation_started is None:
            _record_stage_timing(seed, stage, started_at, started, "completed", settings)
        else:
            _record_stage_timing(seed, f"{stage}-evaluation", evaluation_started_at,
                                 evaluation_started, "completed", settings)


def _teacher_is_complete(
    output: Path,
    configuration: dict[str, Any],
) -> bool:
    manifest = _read_json(output / "manifest.json")
    return bool(
        (output / "dataset.npz").is_file()
        and (
            configuration["forecast"]["strategy"] == "five-fold-ensemble-mean-return"
            or (output / "forecast_metrics.json").is_file()
        )
        and isinstance(manifest, dict)
        and manifest.get("teacher_configuration_hash")
        == config_hash(configuration)
        and canonical_data(manifest.get("teacher_configuration"))
        == canonical_data(configuration)
        and _ensemble_inputs_match(manifest)
        and isinstance(manifest.get("arrays"), dict)
        and manifest["arrays"]
    )


def _ensemble_inputs_match(manifest: dict[str, Any]) -> bool:
    if manifest.get("kind") != "mpc-teacher-cem-ensemble":
        return manifest.get("teacher_configuration", {}).get("forecast", {}).get("strategy") != "five-fold-ensemble-mean-return"
    from tools.mpc.generate_cross_fit_mpc_teacher_dataset_cem import _forecast_bundle_hashes

    members = manifest.get("ensemble", {}).get("members", [])
    expected = manifest.get("teacher_configuration", {}).get("forecast", {}).get("job_hashes")
    if len(members) != 5 or [member.get("job_hash") for member in members] != expected:
        return False
    try:
        return all(_forecast_bundle_hashes(Path(member["bundle"])) == member.get("bundle_hashes") for member in members)
    except (OSError, KeyError, TypeError):
        return False


def _manifest_matches(dataset: str, expected: str | None) -> bool:
    try:
        return hashlib.sha256((_project_path(dataset) / "manifest.json").read_bytes()).hexdigest() == expected
    except OSError:
        return False


def _initial_action_std(configuration: dict[str, Any]) -> float:
    if configuration.get("initial_action_std_source") != CEM_ACTION_STD_SOURCE:
        return float(configuration["initial_action_std"])
    dataset = _project_path(configuration["teacher_dataset"]) / "dataset.npz"
    with np.load(dataset, allow_pickle=False) as stored:
        deviations = stored["cem_iteration_action_std"]
        action_count = stored["first_action"].shape[0]
    if deviations.ndim != 3 or deviations.shape[0] != action_count or not all(deviations.shape):
        raise ValueError("teacher CEM action deviations do not match the first-action labels")
    first_action_deviations = deviations[:, -1, 0]
    if not np.all(np.isfinite(first_action_deviations)) or np.any(first_action_deviations <= 0):
        raise ValueError("teacher CEM first-action deviations must be finite and positive")
    return float(np.median(first_action_deviations))


def _bc_is_complete(
    output: Path,
    configuration: dict[str, Any],
) -> bool:
    if not all(
        (output / name).is_file()
        for name in ("model.zip", "vec_normalize.pkl", "metrics.json", "manifest.json")
    ):
        return False
    manifest = _read_json(output / "manifest.json")
    actor_training = manifest.get("actor_training", {}) if isinstance(manifest, dict) else {}
    try:
        initial_action_std = _initial_action_std(configuration)
    except (KeyError, OSError, ValueError):
        return False
    source = configuration.get("initial_action_std_source", "provided")
    return bool(
        isinstance(manifest, dict)
        and manifest.get("kind") == "recurrent-ppo-behavior-cloning"
        and Path(str(manifest.get("dataset", ""))).resolve()
        == _project_path(configuration["teacher_dataset"]).resolve()
        and Path(str(manifest.get("validation_dataset", ""))).resolve()
        == _project_path(configuration["validation_teacher_dataset"]).resolve()
        and manifest.get("split", {}).get("strategy") == "separate-validation-dataset"
        and _manifest_matches(configuration["teacher_dataset"], manifest.get("dataset_manifest_sha256"))
        and _manifest_matches(configuration["validation_teacher_dataset"], manifest.get("validation_dataset_manifest_sha256"))
        and actor_training.get("learning_rate") == configuration["learning_rate"]
        and actor_training.get("early_stopping_patience") == configuration["patience"]
        and manifest.get("policy", {}).get("initial_action_std") == initial_action_std
        and manifest.get("policy", {}).get("initial_action_std_source", "provided") == source
        and actor_training.get("objective") == "mse-sparse-query-deterministic-action-mean"
        and actor_training.get("seed") == configuration["seed"]
        and actor_training.get("maximum_epochs") == configuration["epochs"]
        and actor_training.get("trajectory_batch_size")
        == configuration["trajectory_batch_size"]
        and actor_training.get("sequence_chunk_length")
        == configuration["sequence_chunk_length"]
    )


def _require_forecasting_complete(experiment: Experiment, seed: int) -> None:
    from tools.mpc.generate_cross_fit_mpc_teacher_dataset_cem import (
        preflight_all_forecasts,
    )

    try:
        preflight_all_forecasts(experiment)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"stage 'mpc' requires completed forecasting artifacts for seed {seed}; "
            "run stage 'forecasting' first"
        ) from exc


def _require_teacher_datasets_complete(
    teacher_output: Path,
    teacher: dict[str, Any],
    validation_output: Path,
    validation_teacher: dict[str, Any],
    seed: int,
) -> None:
    missing = []
    if not _teacher_is_complete(teacher_output, teacher):
        missing.append("training teacher dataset")
    if not _teacher_is_complete(validation_output, validation_teacher):
        missing.append("validation teacher dataset")
    if missing:
        raise RuntimeError(
            f"stage 'behavior-cloning' requires completed MPC artifacts for seed {seed}; "
            f"missing or incompatible: {', '.join(missing)}; run stage 'mpc' first"
        )


def _require_behavior_cloning_complete(
    output: Path,
    configuration: dict[str, Any],
    seed: int,
) -> None:
    if not _bc_is_complete(output, configuration):
        raise RuntimeError(
            f"stage 'fine-tuning' requires a completed behavior-cloning bundle "
            f"for seed {seed}; run stage 'behavior-cloning' first"
        )


def _archive_incomplete(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = path.parent / "_attempts" / path.name / stamp
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archive))
    return archive


def _run_teacher(
    experiment: Experiment,
    output: Path,
    *,
    seed: int,
    workers: int,
    validation_workers: int,
    device: str,
    fold_timings: list[dict[str, Any]],
) -> None:
    from tools.mpc import generate_mpc_teacher_dataset_cem as cem
    from tools.mpc import generate_cross_fit_mpc_teacher_dataset_cem as cross_fit

    args = argparse.Namespace(
        output_dir=output,
        trajectory_cache=cem.DEFAULT_TRAJECTORY_CACHE,
        workers=workers,
        validation_workers=validation_workers,
        device=device,
        seed=seed,
    )
    cross_fit.execute_pipeline(
        args, experiment=experiment,
        measure_fold=lambda index: _measure_stage(
            seed, "mpc-teacher-fold", records=fold_timings, fold_index=index,
            workers=workers, validation_workers=validation_workers,
            requested_device=device,
        ),
    )


def _bc_arguments(
    configuration: dict[str, Any],
    output: Path,
    *,
    workers: int,
    device: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=Path(configuration["teacher_dataset"]),
        output=output,
        seed=configuration["seed"],
        validation_dataset=Path(configuration["validation_teacher_dataset"]),
        epochs=configuration["epochs"],
        trajectory_batch_size=configuration["trajectory_batch_size"],
        sequence_chunk_length=configuration["sequence_chunk_length"],
        trajectory_cache_root=None,
        learning_rate=configuration["learning_rate"],
        initial_action_std=_initial_action_std(configuration),
        initial_action_std_source=configuration.get("initial_action_std_source", "provided"),
        patience=configuration["patience"],
        device=device,
        workers=workers,
    )


def run_seed(
    seed: int,
    *,
    workers: int,
    validation_workers: int,
    device: str,
    generate_bc_plots: bool,
    stages: Sequence[str] | None = None,
) -> None:
    selected_stages = _selected_stages(stages)
    forecast = build_forecasting_experiment(seed)
    teacher = teacher_configuration(forecast, seed)
    teacher_output = teacher_output_directory(teacher)
    validation_teacher = validation_teacher_configuration(forecast, seed)
    validation_output = teacher_output_directory(validation_teacher)
    bc = behavior_cloning_configuration(seed, teacher_output, validation_output)
    bc_output = behavior_cloning_output_directory(bc)
    control = build_control_experiment(seed, bc_output)

    if any(stage in selected_stages for stage in ("mpc", "behavior-cloning", "fine-tuning")):
        _require_matched_teacher_contract(teacher, validation_teacher, control)

    plan = pipeline_plan(seed)
    plan["requested_stages"] = list(selected_stages)
    _write_json(_state_path(seed, "plan.json"), plan)
    phase = selected_stages[0]

    def control_training_complete(stage: str, finish_training) -> None:
        nonlocal phase
        finish_training()
        phase = f"{stage}-evaluation"
        _set_status(seed, phase, "running", requested_stages=list(selected_stages))

    try:
        if "forecasting" in selected_stages:
            phase = "forecasting"
            with _measure_stage(
                seed, phase, workers=workers, concurrent_jobs=1,
                device_source="forecast-job-config",
            ):
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                execute_experiment(
                    forecast,
                    source_path=Path(__file__),
                    resume=True,
                    workers=workers,
                    registry=_training_registry(),
                )

        if "mpc" in selected_stages:
            phase = "mpc"
            with _measure_stage(
                seed, phase, workers=workers, validation_workers=validation_workers,
                requested_device=device,
            ) as mpc_timings:
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                _require_forecasting_complete(forecast, seed)
                if _teacher_is_complete(teacher_output, teacher):
                    print(f"Skipping completed MPC teacher dataset: {teacher_output}")
                else:
                    archive = _archive_incomplete(teacher_output)
                    if archive is not None:
                        print(f"Archived incomplete MPC teacher attempt to {archive}")
                    with _measure_stage(
                        seed, "mpc-teacher-training", records=mpc_timings,
                        output=str(teacher_output), workers=workers,
                        validation_workers=validation_workers, requested_device=device,
                    ) as fold_timings:
                        _run_teacher(
                            forecast,
                            teacher_output,
                            seed=seed,
                            workers=workers,
                            validation_workers=validation_workers,
                            device=device,
                            fold_timings=fold_timings,
                        )

                if _teacher_is_complete(validation_output, validation_teacher):
                    print(
                        "Skipping completed validation MPC teacher dataset: "
                        f"{validation_output}"
                    )
                else:
                    archive = _archive_incomplete(validation_output)
                    if archive is not None:
                        print(f"Archived incomplete validation teacher attempt to {archive}")
                    from tools.mpc import generate_validation_mpc_teacher_dataset_cem as ensemble
                    from tools.mpc import generate_mpc_teacher_dataset_cem as cem

                    with _measure_stage(
                        seed, "mpc-teacher-validation", records=mpc_timings,
                        output=str(validation_output), workers=workers,
                        requested_device=device,
                    ):
                        ensemble.execute_pipeline(
                            argparse.Namespace(
                                output_dir=validation_output,
                                trajectory_cache=cem.DEFAULT_TRAJECTORY_CACHE,
                                workers=workers,
                                device=device,
                                seed=seed,
                            ),
                            experiment=forecast,
                        )

        if "behavior-cloning" in selected_stages:
            phase = "behavior-cloning"
            with _measure_stage(
                seed, phase, workers=workers, requested_device=device,
                concurrent_jobs=1,
            ):
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                _require_teacher_datasets_complete(
                    teacher_output,
                    teacher,
                    validation_output,
                    validation_teacher,
                    seed,
                )
                from tools.control.bc import train_mpc_teacher_recurrent_ppo as bc_trainer

                if _bc_is_complete(bc_output, bc):
                    print(f"Skipping completed behavior-cloning bundle: {bc_output}")
                else:
                    archive = _archive_incomplete(bc_output)
                    if archive is not None:
                        print(f"Archived incomplete behavior-cloning attempt to {archive}")
                    # A rebuilt BC bundle invalidates dependent PPO initialization.
                    prior_ppo = _archive_incomplete(
                        expected_run_directory(control, "mpc-teacher-bc")
                    )
                    if prior_ppo is not None:
                        print(f"Archived PPO run with stale BC initialization to {prior_ppo}")
                    bc_trainer.train(
                        _bc_arguments(bc, bc_output, workers=workers, device=device)
                    )
        if "baseline" in selected_stages:
            phase = "baseline"
            with _measure_control_stage(
                seed, phase, workers=workers, concurrent_jobs=1,
                device_source="control-job-config",
            ) as finish_training:
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                execute_experiment(
                    control,
                    selected={"baseline"},
                    source_path=Path(__file__),
                    resume=True,
                    workers=workers,
                    registry=_training_registry(on_training_complete=lambda: control_training_complete(
                        "baseline", finish_training)),
                )

        if "fine-tuning" in selected_stages:
            phase = "fine-tuning"
            with _measure_control_stage(
                seed, phase, workers=workers, concurrent_jobs=1,
                device_source="control-job-config",
            ) as finish_training:
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                _require_behavior_cloning_complete(bc_output, bc, seed)
                execute_experiment(
                    control,
                    selected={"mpc-teacher-bc"},
                    source_path=Path(__file__),
                    resume=True,
                    workers=workers,
                    registry=_training_registry(on_training_complete=lambda: control_training_complete(
                        "fine-tuning", finish_training)),
                )

        if generate_bc_plots and "behavior-cloning" in selected_stages:
            phase = "behavior-cloning-plots"
            with _measure_stage(seed, phase, workers=workers):
                _set_status(seed, phase, "running", requested_stages=list(selected_stages))
                plots = bc_output / "plots/final"
                evaluation = bc_output / "evaluation/final"
                if (
                    not (plots / "validation").is_dir()
                    or not (plots / "test").is_dir()
                    or not (evaluation / "metrics.json").is_file()
                    or not (evaluation / "validation/raw/manifest.json").is_file()
                    or not (evaluation / "test/raw/manifest.json").is_file()
                ):
                    from tools.control.bc import train_mpc_teacher_recurrent_ppo as bc_trainer

                    bc_trainer.generate_final_plots(bc_output, workers=workers)
    except BaseException as exc:
        _set_status(
            seed,
            phase,
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        raise

    final_phase = "complete" if selected_stages == STAGES else selected_stages[-1]
    _set_status(
        seed,
        final_phase,
        "completed",
        requested_stages=list(selected_stages),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run selected replicated pipeline stages")
    run.add_argument(
        "--seed",
        action="append",
        type=_non_negative_integer,
        help="run only this seed; repeatable (default: all paper seeds)",
    )
    run.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help=(
            "run only this stage; repeatable (default: all stages in dependency "
            "order). Dependencies must already be complete or be selected too"
        ),
    )
    run.add_argument(
        "--workers",
        type=_positive_integer,
        default=1,
        help="trajectory, teacher, and evaluation worker processes (default: 1)",
    )
    run.add_argument(
        "--validation-workers",
        type=_positive_integer,
        default=4,
        help="MPC teacher EPANET validation workers (default: 4)",
    )
    run.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    run.add_argument(
        "--generate-bc-plots",
        action="store_true",
        help="generate the behavior-cloned policy's standalone validation/test plots",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print all resolved seeds, hashes, and artifact paths without writing",
    )

    describe = subparsers.add_parser("describe", help="print the resolved pipeline")
    describe.add_argument(
        "--seed",
        action="append",
        type=_non_negative_integer,
        help="describe only this seed; repeatable (default: all paper seeds)",
    )
    return parser


def _selected_seeds(values: Sequence[int] | None) -> tuple[int, ...]:
    return tuple(dict.fromkeys(SEEDS if values is None else values))


def _selected_stages(values: Sequence[str] | None) -> tuple[str, ...]:
    requested = set(STAGES if values is None else values)
    unknown = requested - set(STAGES)
    if unknown:
        raise ValueError(f"unknown pipeline stages: {sorted(unknown)}")
    selected = tuple(stage for stage in STAGES if stage in requested)
    if not selected:
        raise ValueError("at least one pipeline stage must be selected")
    return selected


def _print_plans(seeds: Sequence[int], stages: Sequence[str] = STAGES) -> None:
    payload = {
        "schema_version": 1,
        "default_seeds": list(SEEDS),
        "requested_stages": list(stages),
        "runs": [pipeline_plan(seed) for seed in seeds],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["run"]
    args = build_parser().parse_args(arguments)
    if args.command is None:
        build_parser().print_help()
        return 0
    seeds = _selected_seeds(args.seed)
    stages = _selected_stages(args.stage if args.command == "run" else None)
    if args.command == "describe" or args.dry_run:
        _print_plans(seeds, stages)
        return 0

    try:
        for seed in seeds:
            print(f"\n=== {PIPELINE_NAME}: seed {seed} ===")
            run_seed(
                seed,
                workers=args.workers,
                validation_workers=args.validation_workers,
                device=args.device,
                generate_bc_plots=args.generate_bc_plots,
                stages=stages,
            )
    except KeyboardInterrupt:
        print("Pipeline interrupted; start it again to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
