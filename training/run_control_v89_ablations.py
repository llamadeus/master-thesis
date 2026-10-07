#!/usr/bin/env python3
"""Train V89 initialization and training-budget ablations, then validate checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer import (ControlHooksSpec, Experiment,  # noqa: E402
                              OutputSpec)
from packages.trainer.execution import (  # noqa: E402
    _configure_native_thread_limits, execute_experiment)
from packages.trainer.serialization import config_hash  # noqa: E402
from training import run_principal_v89_pipeline as principal  # noqa: E402

EXPERIMENT_NAME = "hanoi-control-v89-ablations"
LONG_BASELINE_NAME = "baseline-4m"
LONG_BASELINE_TIMESTEPS = 4_000_000
# Keep the original job budget in its hash so the five existing runs retain
# their exact directories and checkpoint identities. The runner extends them.
LONG_BASELINE_ORIGINAL_TIMESTEPS = 3_400_000
VARIANTS = {
    "without-actor-weights": ("action_std", "observation_normalization"),
    "adjusted-action-std": ("action_std",),
    "offline-normalization": ("observation_normalization",),
    LONG_BASELINE_NAME: (),
}


def _bc_inputs(seed: int) -> tuple[Path, dict]:
    forecast = principal.build_forecasting_experiment(seed)
    teacher = principal.teacher_configuration(forecast, seed)
    validation = principal.validation_teacher_configuration(forecast, seed)
    configuration = principal.behavior_cloning_configuration(
        seed,
        principal.teacher_output_directory(teacher),
        principal.teacher_output_directory(validation),
    )
    return principal.behavior_cloning_output_directory(configuration), configuration


def build_experiment(seed: int, bc_output: Path) -> Experiment:
    reference = principal.build_control_experiment(seed, bc_output).job("baseline")
    jobs = tuple(
        replace(
            reference,
            name=name,
            total_timesteps=(LONG_BASELINE_ORIGINAL_TIMESTEPS
                             if name == LONG_BASELINE_NAME else reference.total_timesteps),
            algorithm=replace(
                reference.algorithm,
                pretrained_policy=bc_output,
                pretrained_components=components,
            ) if components else reference.algorithm,
            hooks=ControlHooksSpec(),
            output=OutputSpec(group=f"control/ablations/v89/{name}"),
        )
        for name, components in VARIANTS.items()
    )
    return Experiment(
        EXPERIMENT_NAME,
        jobs,
        "V89 initialization ablations and an ordinary baseline continued to 4 million timesteps.",
    )


def _training_registry():
    from packages.trainer.control.runner import ControlRunner

    registry = principal._training_registry()
    registry.register("control", ControlRunner(
        render_plots=False,
        defer_evaluation=True,
        evaluate_best_model=True,
        timestep_targets={LONG_BASELINE_NAME: LONG_BASELINE_TIMESTEPS},
        resume_legacy_training=True,
    ), replace=True)
    return registry


def _require_long_baseline_checkpoint(experiment: Experiment, registry) -> None:
    job = experiment.job(LONG_BASELINE_NAME)
    output = principal.expected_run_directory(experiment, job.name)
    runner = registry.runner_for("control")
    if not runner.can_resume(output, job):
        raise RuntimeError(f"long baseline requires a compatible existing checkpoint: {output}")


def _preflight(seed: int, bc_output: Path, configuration: dict, experiment: Experiment) -> dict:
    """Validate the seed's BC bundle and the teacher-derived action deviation."""
    _configure_native_thread_limits()
    if not principal._bc_is_complete(bc_output, configuration):
        raise RuntimeError(f"incomplete or incompatible V89 BC bundle for seed {seed}: {bc_output}")
    manifest = principal._read_json(bc_output / "manifest.json")
    policy = manifest["policy"]
    expected_std = principal._initial_action_std(configuration)
    if policy.get("initial_action_std_source") != principal.CEM_ACTION_STD_SOURCE:
        raise RuntimeError(f"BC action-standard-deviation source is incompatible: {bc_output}")

    import numpy as np
    from sb3_contrib import RecurrentPPO

    from packages.trainer.control.runner import \
        _validate_pretrained_policy_contract

    model = RecurrentPPO.load(str(bc_output / "model.zip"), device="cpu")
    log_std = model.policy.log_std.detach().cpu().numpy()
    if (not np.all(np.isfinite(log_std))
            or not np.allclose(log_std, np.log(expected_std), rtol=0, atol=1e-6)):
        raise RuntimeError(f"BC model log_std disagrees with the teacher median: {bc_output}")
    for job in experiment.jobs:
        _validate_pretrained_policy_contract(manifest, job)
    return {
        "bundle": str(bc_output),
        "teacher_first_action_final_cem_std_median": expected_std,
        "initial_action_std_source": principal.CEM_ACTION_STD_SOURCE,
        "model_log_std": log_std.tolist(),
    }


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    for command in ("run", "describe"):
        selected = subparsers.add_parser(command)
        selected.add_argument("--seed", action="append", type=_nonnegative_integer,
                              help="training seed; repeatable (default: five principal seeds)")
        selected.add_argument("--variant", action="append", choices=tuple(VARIANTS),
                              help="ablation; repeatable (default: all four)")
        if command == "run":
            selected.add_argument("--workers", type=_positive_integer, default=1)
            selected.add_argument("--resume", action="store_true",
                                  help="resume existing runs (the long baseline always resumes)")
            selected.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["run"]
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    seeds = tuple(dict.fromkeys(principal.SEEDS if args.seed is None else args.seed))
    variants = tuple(dict.fromkeys(VARIANTS if args.variant is None else args.variant))
    selected = set(variants)
    try:
        prepared = []
        registry = None if args.command == "describe" or args.dry_run else _training_registry()
        for seed in seeds:
            bc_output, configuration = _bc_inputs(seed)
            experiment = build_experiment(seed, bc_output)
            pretrained_jobs = tuple(
                job for job in experiment.jobs
                if job.name in selected and job.algorithm.pretrained_policy is not None
            )
            if args.command == "describe" or args.dry_run:
                print(json.dumps({
                    "seed": seed,
                    "experiment": experiment.name,
                    "bc_bundle": str(bc_output) if pretrained_jobs else None,
                    "initial_action_std_source": (principal.CEM_ACTION_STD_SOURCE
                                                  if pretrained_jobs else None),
                    "jobs": [
                        {
                            "variant": job.name,
                            "components": list(VARIANTS[job.name]),
                            "total_timesteps": (LONG_BASELINE_TIMESTEPS
                                                if job.name == LONG_BASELINE_NAME else job.total_timesteps),
                            "job_hash": config_hash(job),
                            "output": str(principal.expected_run_directory(experiment, job.name)),
                        }
                        for job in experiment.jobs if job.name in selected
                    ],
                }, indent=2, sort_keys=True))
                continue
            if LONG_BASELINE_NAME in selected:
                _require_long_baseline_checkpoint(experiment, registry)
            provenance = (
                _preflight(seed, bc_output, configuration,
                           replace(experiment, jobs=pretrained_jobs))
                if pretrained_jobs else None
            )
            prepared.append((seed, experiment, provenance))
        for seed, experiment, provenance in prepared:
            print(f"\n=== {EXPERIMENT_NAME}: seed {seed} ===")
            if provenance is not None:
                print(f"BC initial action std ({principal.CEM_ACTION_STD_SOURCE}): "
                      f"{provenance['teacher_first_action_final_cem_std_median']:.9g}")
            # Keep ordinary ablation resume behavior; only the extended
            # baseline must always continue a checkpoint, including completed runs.
            selections = [selected - {LONG_BASELINE_NAME}, selected & {LONG_BASELINE_NAME}]
            for selection in selections:
                if selection:
                    execute_experiment(
                        experiment,
                        selected=selection,
                        source_path=Path(__file__),
                        resume=args.resume or LONG_BASELINE_NAME in selection,
                        workers=args.workers,
                        registry=registry,
                    )
    except KeyboardInterrupt:
        print("Ablation interrupted; rerun with --resume.", file=sys.stderr)
        return 130
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Ablation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
