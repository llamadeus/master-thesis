"""Shared command-line entrypoint used by standalone training scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .execution import (RunnerRegistry, _configure_native_thread_limits,
                        execute_experiment)
from .experiment import Experiment, job_dependencies
from .serialization import config_hash


def main(
    experiment: Experiment,
    argv: Sequence[str] | None = None,
    *,
    registry: RunnerRegistry | None = None,
) -> int:
    """Run the shared trainer CLI for one standalone experiment script."""

    parser, command_parsers = _build_parser(experiment)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "help":
        target = command_parsers.get(args.topic) if args.topic else parser
        target.print_help()
        return 0

    selected = set(args.job) if getattr(args, "job", None) else None
    try:
        jobs = experiment.ordered_jobs(selected)
        if args.command == "describe":
            if args.json:
                payload = {
                    "experiment": experiment.to_dict(),
                    "selected_jobs": [job.name for job in jobs],
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _describe(experiment, jobs)
            return 0

        if args.command == "prepare-hydraulics":
            _configure_native_thread_limits()
            from .data import resolve_dataset
            from .simulation import prepare_hydraulic_outflow_cache

            for job in jobs:
                prepare_hydraulic_outflow_cache(
                    resolve_dataset(job.dataset),
                    workers=args.workers,
                    description=f"{job.name} hydraulics",
                )
            return 0

        source_path = _source_path()
        run_dirs = execute_experiment(
            experiment,
            selected=selected,
            registry=registry,
            output_root=args.output_root,
            source_path=source_path,
            dry_run=args.dry_run,
            resume=args.resume,
            restart=args.restart,
            workers=args.workers,
            concurrent=args.concurrent,
        )
        if args.dry_run:
            print("Execution order:")
            for job, run_dir in zip(jobs, run_dirs, strict=True):
                print(f"  {job.name} ({job.kind}) -> {run_dir}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; resume with --resume.",file=sys.stderr)
        return 130
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser(experiment: Experiment) -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description=experiment.description or f"Run the {experiment.name} training experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands: dict[str, argparse.ArgumentParser] = {}

    run = subparsers.add_parser("run", help="run the configured training jobs")
    run.add_argument("--job", action="append", help="run this job and its dependencies; repeatable")
    run.add_argument("--dry-run", action="store_true", help="validate and show execution without writing files")
    run.add_argument("--output-root", type=Path, help="override the configured output roots")
    run.add_argument("--workers", type=_positive_integer, default=1,
                     help="worker processes for forecast trajectory generation and control evaluation (default: 1)")
    run.add_argument("--concurrent", type=_positive_integer, default=1,
                     help="independent training jobs to run concurrently (default: 1)")
    recovery = run.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true",
                          help="continue a compatible incomplete run, or start fresh if no checkpoint exists")
    recovery.add_argument("--restart", action="store_true", help="archive the existing attempt and start cleanly")
    commands["run"] = run

    describe = subparsers.add_parser("describe", help="show the resolved experiment configuration")
    describe.add_argument("--job", action="append", help="describe this job and its dependencies; repeatable")
    describe.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    commands["describe"] = describe

    prepare = subparsers.add_parser(
        "prepare-hydraulics",
        help="materialize hydraulic outflow caches required by selected jobs",
    )
    prepare.add_argument(
        "--job", action="append",
        help="prepare this job and its dependencies; repeatable",
    )
    prepare.add_argument(
        "--workers", type=_positive_integer, default=1,
        help="parallel hydraulic simulation workers (default: 1)",
    )
    commands["prepare-hydraulics"] = prepare

    help_parser = subparsers.add_parser("help", help="show general or command-specific help")
    help_parser.add_argument(
        "topic", nargs="?",
        choices=("run", "describe", "prepare-hydraulics", "help"),
    )
    commands["help"] = help_parser
    return parser, commands


def _describe(experiment: Experiment, jobs: Sequence) -> None:
    print(f"Experiment: {experiment.name}")
    if experiment.description:
        print(experiment.description)
    print(f"Jobs: {len(jobs)}")
    for job in jobs:
        dependencies = job_dependencies(job)
        dependency_text = ", ".join(dependencies) if dependencies else "none"
        print(f"  {job.name} [{job.kind}] hash={config_hash(job)} dependencies={dependency_text}")


def _source_path() -> Path | None:
    if not sys.argv or not sys.argv[0]:
        return None
    path = Path(sys.argv[0])
    return path.resolve() if path.is_file() else None


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result
