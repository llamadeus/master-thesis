#!/usr/bin/env python3
"""Label the existing validation trajectories with a five-forecaster CEM teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from packages.trainer import Experiment
from packages.trainer.serialization import canonical_data, config_hash
from tools.mpc import generate_cross_fit_mpc_teacher_dataset_cem as cross_fit
from tools.mpc import generate_mpc_teacher_dataset_cem as cem
from training.datasets.hanoi import forecasting_full
from training.experiments.forecasting_encoder_decoder_lstm_cross_fit import FOLDS, build_experiment


def validation_teacher_configuration(
    experiment: Experiment | None = None, *, seed: int = cem.DEFAULT_SEED,
) -> dict[str, Any]:
    experiment = build_experiment(training_seed=seed) if experiment is None else experiment
    jobs = experiment.jobs[:len(FOLDS)]
    if len(jobs) != 5:
        raise ValueError("validation teacher requires all five cross-fitting jobs")
    for fold, job in zip(FOLDS, jobs, strict=True):
        expected = set(forecasting_full.train.scenarios) - set(fold.held_out.scenarios)
        if set(job.dataset.train.scenarios) != expected:
            raise ValueError("validation teacher forecaster does not match its training fold")
        if set(job.dataset.train.scenarios) & set(forecasting_full.validation.scenarios):
            raise ValueError("validation scenarios occur in forecast training")
        if job.training_seed != seed:
            raise ValueError("validation teacher forecaster training seed mismatch")
    identity = cross_fit.cross_fit_teacher_configuration(experiment, seed=seed)
    identity["source_dataset"]["split"] = canonical_data(forecasting_full.validation)
    identity["forecast"] = {
        "strategy": "five-fold-ensemble-mean-return",
        "history_steps": cem.HISTORY_STEPS,
        "horizon_steps": cem.HORIZON_STEPS,
        "job_hashes": [config_hash(job) for job in jobs],
        "aggregation": "arithmetic-mean-of-separately-scored-returns",
    }
    return identity


DEFAULT_TEACHER_CONFIGURATION = validation_teacher_configuration()
DEFAULT_OUTPUT = cem.PROJECT_ROOT / "data/datasets/mpc-teacher-cem" / config_hash(DEFAULT_TEACHER_CONFIGURATION)


def execute_pipeline(
    args: argparse.Namespace, *, experiment: Experiment | None = None,
) -> tuple[Path, Path]:
    experiment = build_experiment(training_seed=args.seed) if experiment is None else experiment
    identity = validation_teacher_configuration(experiment, seed=args.seed)
    runs = tuple(cross_fit.preflight_forecast(job, experiment.name) for job in experiment.jobs[:5])
    output = DEFAULT_OUTPUT.parent / config_hash(identity) if args.output_dir is None else args.output_dir
    generation = cross_fit._generation_args(args, runs[0], output)
    # Each member keeps its own frozen training normalization and best checkpoint.
    ensemble = {
        "aggregation": identity["forecast"]["aggregation"],
        "members": [
            {"job_hash": config_hash(run.job), "bundle": str(run.bundle),
             "bundle_hashes": cross_fit._forecast_bundle_hashes(run.bundle),
             "training_seed": run.job.training_seed,
             "training_scenarios": list(run.job.dataset.train.scenarios)}
            for run in runs
        ],
        "predicted_chlorine_axes": ["record", "member", "horizon", "node"],
    }
    return cem.generate_dataset(
        generation, forecast_job=runs[0].job,
        source_split=forecasting_full.validation,
        source_definition="training.datasets.hanoi.forecasting_full.validation",
        require_out_of_fold=False, source_phase="evaluation",
        ensemble_bundles=tuple(run.bundle for run in runs),
        manifest_updates={
            "kind": "mpc-teacher-cem-ensemble", "source_split": "validation",
            "teacher_configuration": identity,
            "teacher_configuration_hash": config_hash(identity),
            "ensemble": ensemble,
            "forecasting_run": None, "forecasting_bundle": None,
            "forecast_job_hash": None, "forecast_training_dataset_contract": None,
            "out_of_fold": {"required": True,
                            "source_scenarios": list(forecasting_full.validation.scenarios),
                            "forecast_training_scenarios": list(forecasting_full.train.scenarios)},
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-cache", type=Path, default=cem.DEFAULT_TRAJECTORY_CACHE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=cem.DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = execute_pipeline(args)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
