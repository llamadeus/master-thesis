"""Dependency-aware, locked, resumable experiment execution."""

from __future__ import annotations

import _thread
import importlib.metadata
import json
import multiprocessing
import os
import queue
import shutil
import signal
import socket
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .experiment import Experiment, job_dependencies
from .serialization import canonical_data, config_hash
from .specs import ArtifactRef, TrainingSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_MANIFEST_VERSION = 2
_PROGRESS_ROWS_PER_JOB = 2
_GRACEFUL_STOP_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RunResult:
    artifacts: Mapping[str, str | Path]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    experiment: Experiment
    job: TrainingSpec
    output_dir: Path
    artifacts: Mapping[str, Mapping[str, Path]]
    source_path: Path | None
    resume: bool = False
    workers: int = 1

    def resolve(self, reference: ArtifactRef) -> Path:
        try:
            return self.artifacts[reference.job][reference.artifact]
        except KeyError as exc:
            raise RuntimeError(f"artifact {reference.job}:{reference.artifact} is unavailable") from exc


class Runner(Protocol):
    def validate(self, job: TrainingSpec) -> None: ...
    def can_resume(self, output_dir: Path, job: TrainingSpec) -> bool: ...
    def run(self, context: RunContext) -> RunResult: ...


class RunnerRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, Runner] = {}

    def register(self, kind: str, runner: Runner, *, replace: bool = False) -> None:
        if not kind or (kind in self._runners and not replace):
            raise ValueError(f"invalid or duplicate runner kind {kind!r}")
        self._runners[kind] = runner

    def runner_for(self, kind: str) -> Runner:
        if kind not in self._runners:
            raise RuntimeError(f"no runner registered for job kind {kind!r}")
        return self._runners[kind]

    def has_runner(self, kind: str) -> bool:
        return kind in self._runners


def default_registry() -> RunnerRegistry:
    """Import heavyweight engines only when execution is requested."""
    # PyTorch and epanet-plus bundle distinct libomp builds on macOS. Both are
    # needed by the control runner, and allowing either runtime to create a
    # thread team can corrupt the other's synchronization state. These values
    # must be established before importing either native extension.
    _configure_native_thread_limits()
    registry = RunnerRegistry()
    from .forecasting.runner import ForecastRunner
    from .control.runner import ControlRunner
    registry.register("forecast", ForecastRunner())
    registry.register("control", ControlRunner())
    return registry


def _configure_native_thread_limits() -> None:
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[name]="1"


def execute_experiment(
    experiment: Experiment,
    *, selected: set[str] | None = None,
    registry: RunnerRegistry | None = None,
    output_root: Path | None = None,
    source_path: Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    restart: bool = False,
    workers: int = 1,
    concurrent: int = 1,
) -> tuple[Path, ...]:
    if resume and restart:
        raise ValueError("resume and restart are mutually exclusive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(concurrent, bool) or not isinstance(concurrent, int) or concurrent < 1:
        raise ValueError("concurrent must be a positive integer")
    jobs = experiment.ordered_jobs(selected)
    active_registry = registry if registry is not None else default_registry()
    for job in jobs:
        active_registry.runner_for(job.kind).validate(job)
    run_dirs = tuple(_run_dir(experiment, job, output_root) for job in jobs)
    if dry_run:
        return run_dirs

    with _sigterm_as_keyboard_interrupt():
        if concurrent > 1 and len(jobs) > 1:
            _execute_concurrently(
                experiment,
                jobs,
                run_dirs,
                registry,
                source_path,
                resume,
                restart,
                workers,
                concurrent,
            )
            return run_dirs

        artifacts: dict[str, Mapping[str, Path]] = {}
        for job, run_dir in zip(jobs, run_dirs, strict=True):
            artifacts[job.name] = _execute_job(
                experiment, job, run_dir, artifacts, active_registry, source_path,
                resume, restart, workers,
            )
    return run_dirs


def _execute_job(
    experiment: Experiment,
    job: TrainingSpec,
    run_dir: Path,
    artifacts: Mapping[str, Mapping[str, Path]],
    registry: RunnerRegistry,
    source_path: Path | None,
    resume: bool,
    restart: bool,
    workers: int,
) -> Mapping[str, Path]:
    runner = registry.runner_for(job.kind)
    job_resume = resume
    status = _read_json(run_dir / "status.json")
    completed = _load_completed_artifacts(run_dir)
    needs_training = getattr(runner, "needs_training", None)
    if resume and completed is not None and callable(needs_training) and needs_training(run_dir, job):
        completed = None
    if completed is not None and not restart:
        completed_resume = getattr(runner, "resume_completed", None)
        if resume and callable(completed_resume):
            with _RunLock(run_dir / ".lock"):
                completed_resume(
                    RunContext(
                        experiment,
                        job,
                        run_dir,
                        artifacts,
                        source_path,
                        resume=True,
                        workers=workers,
                    )
                )
        print(f"Skipping completed job {job.name!r}: {run_dir}")
        return completed
    if run_dir.exists() and restart:
        if (run_dir / ".lock").exists():
            raise RuntimeError(f"cannot restart an actively locked run: {run_dir}")
        _archive_attempt(run_dir)
        status = None
    elif run_dir.exists() and completed is None and not resume:
        raise RuntimeError(f"incomplete run exists at {run_dir}; use --resume or --restart")
    checkpoint_probe = getattr(runner, "has_checkpoint", None)
    has_checkpoint = checkpoint_probe(run_dir, job) if callable(checkpoint_probe) else run_dir.exists()
    if resume and not has_checkpoint:
        warnings.warn(
            f"--resume requested but no checkpoint exists at {run_dir}; starting from scratch",
            RuntimeWarning,
            stacklevel=2,
        )
        job_resume = False
    elif resume and not runner.can_resume(run_dir, job):
        raise RuntimeError(f"run cannot be resumed from a compatible latest checkpoint: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    lock = _RunLock(run_dir / ".lock")
    with lock:
        started = status.get("started_at", _now()) if isinstance(status, dict) and job_resume else _now()
        attempt_started = _now()
        attempt_clock = time.perf_counter()
        if job_resume:
            history = _read_json(run_dir / "resume_history.json")
            if not isinstance(history, list): history = []
            history.append({"resumed_at": _now(), "prior_status": status})
            _write_json(run_dir / "resume_history.json", history)
        manifest = _manifest(experiment, job, source_path)
        _write_json(run_dir / "config.resolved.json", manifest)
        if source_path is not None and source_path.is_file():
            shutil.copy2(source_path, run_dir / "configuration.py")
        _write_json(run_dir / "status.json", {
            "status": "running", "started_at": started, "updated_at": _now(),
            "attempt_started_at": attempt_started, "resumed": job_resume,
            "host": socket.gethostname(), "pid": os.getpid(),
        })
        context = RunContext(experiment, job, run_dir, artifacts, source_path, job_resume, workers)
        try:
            result = runner.run(context)
            resolved = _validate_artifacts(result.artifacts, run_dir)
            metadata = canonical_data(result.metadata)
            _write_json(run_dir / "artifacts.json", {
                "schema_version": 1,
                "artifacts": {key: str(path) for key, path in sorted(resolved.items())},
                "metadata": metadata,
            })
            _write_json(run_dir / "status.json", {
                "status": "completed", "started_at": started, "completed_at": _now(),
                "attempt_started_at": attempt_started,
                "attempt_elapsed_seconds": time.perf_counter() - attempt_clock,
                "resumed": job_resume,
            })
            return resolved
        except KeyboardInterrupt:
            _write_json(run_dir / "status.json", {
                "status": "interrupted", "started_at": started, "interrupted_at": _now(),
                "attempt_started_at": attempt_started,
                "attempt_elapsed_seconds": time.perf_counter() - attempt_clock,
                "resumed": job_resume,
            })
            raise
        except Exception as exc:
            _write_json(run_dir / "status.json", {
                "status": "failed", "started_at": started, "failed_at": _now(),
                "attempt_started_at": attempt_started,
                "attempt_elapsed_seconds": time.perf_counter() - attempt_clock,
                "resumed": job_resume, "error": str(exc), "traceback": traceback.format_exc(),
            })
            raise


def _execute_concurrently(
    experiment: Experiment,
    jobs: tuple[TrainingSpec, ...],
    run_dirs: tuple[Path, ...],
    registry: RunnerRegistry | None,
    source_path: Path | None,
    resume: bool,
    restart: bool,
    workers: int,
    concurrent: int,
) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    progress_lock = context.RLock()
    stop_requested = context.Event()
    pending = list(zip(jobs, run_dirs, strict=True))
    running: dict[str, tuple[multiprocessing.Process, int]] = {}
    available_slots = list(range(concurrent))
    artifacts: dict[str, Mapping[str, Path]] = {}
    try:
        while pending or running:
            while len(running) < concurrent:
                ready_index = next((
                    index for index, (job, _) in enumerate(pending)
                    if set(job_dependencies(job)) <= artifacts.keys()
                ), None)
                if ready_index is None:
                    break
                job, run_dir = pending.pop(ready_index)
                slot = available_slots.pop(0)
                process = context.Process(
                    target=_job_process_entry,
                    name=f"trainer-{job.name}",
                    args=(
                        results, experiment, job, run_dir, dict(artifacts), registry,
                        source_path, resume, restart, workers, progress_lock,
                        slot * _PROGRESS_ROWS_PER_JOB, stop_requested,
                    ),
                )
                process.start()
                running[job.name] = (process, slot)
            if not running:
                raise RuntimeError("concurrent job scheduler could not find a runnable job")
            try:
                state, name, payload = results.get(timeout=0.2)
            except queue.Empty:
                crashed = next((
                    (name, process.exitcode) for name, (process, _) in running.items()
                    if process.exitcode is not None
                ), None)
                if crashed is None:
                    continue
                name, exit_code = crashed
                _interrupt_processes(
                    (process for process, _ in running.values()), stop_requested,
                )
                raise RuntimeError(f"job {name!r} process exited with code {exit_code}")
            process, slot = running.pop(name)
            process.join()
            available_slots.append(slot)
            available_slots.sort()
            if state != "completed":
                _interrupt_processes(
                    (process for process, _ in running.values()), stop_requested,
                )
                raise RuntimeError(f"job {name!r} failed in a worker process: {payload}")
            artifacts[name] = {key: Path(value) for key, value in payload.items()}
    except KeyboardInterrupt:
        _interrupt_processes(
            (process for process, _ in running.values()), stop_requested,
        )
        raise
    finally:
        for process, _ in running.values():
            process.join(timeout=0)
        results.close()
        results.join_thread()


def _job_process_entry(
    results,
    experiment: Experiment,
    job: TrainingSpec,
    run_dir: Path,
    artifacts: Mapping[str, Mapping[str, Path]],
    registry: RunnerRegistry | None,
    source_path: Path | None,
    resume: bool,
    restart: bool,
    workers: int,
    progress_lock,
    progress_position: int,
    stop_requested,
) -> None:
    finished = threading.Event()
    _configure_worker_stopping(stop_requested, finished)
    try:
        from .progress import configure_job_progress
        configure_job_progress(progress_position, progress_lock)
        active_registry = registry if registry is not None else default_registry()
        active_registry.runner_for(job.kind).validate(job)
        resolved = _execute_job(
            experiment, job, run_dir, artifacts, active_registry, source_path,
            resume, restart, workers,
        )
    except KeyboardInterrupt:
        results.put(("interrupted", job.name, "interrupted"))
    except BaseException as exc:
        results.put(("failed", job.name, f"{type(exc).__name__}: {exc}"))
    else:
        results.put(("completed", job.name, {
            key: str(path) for key, path in resolved.items()
        }))
    finally:
        finished.set()


def _configure_worker_stopping(stop_requested, finished: threading.Event) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def interrupt_main(_signum, _frame):
        if not finished.is_set():
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_main)

    def watch_stop_request():
        stop_requested.wait()
        if not finished.is_set():
            _thread.interrupt_main(signal.SIGTERM)

    threading.Thread(
        target=watch_stop_request,
        name="trainer-stop-watcher",
        daemon=True,
    ).start()


def _interrupt_processes(processes, stop_requested) -> None:
    active = tuple(process for process in processes if process.is_alive())
    stop_requested.set()
    deadline = time.monotonic() + _GRACEFUL_STOP_TIMEOUT_SECONDS
    try:
        for process in active:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        _kill_processes(active)
        raise
    _kill_processes(process for process in active if process.is_alive())


def _kill_processes(processes) -> None:
    active = tuple(processes)
    for process in active:
        if process.is_alive():
            if hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
    for process in active:
        process.join()


@contextmanager
def _sigterm_as_keyboard_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _manifest(experiment: Experiment, job: TrainingSpec, source_path: Path | None) -> dict[str, Any]:
    dependencies = {name: config_hash(experiment.job(name)) for name in job_dependencies(job)}
    packages = {}
    for name in ("numpy", "torch", "stable-baselines3", "sb3-contrib", "gymnasium", "epyt-flow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": RUN_MANIFEST_VERSION, "experiment": experiment.name,
        "job": canonical_data(job), "job_hash": config_hash(job), "dependency_hashes": dependencies,
        "source": str(source_path.resolve()) if source_path else None,
        "versions": packages,
    }


class _RunLock:
    def __init__(self, path: Path): self.path = path
    def __enter__(self):
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            owner = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"run is locked: {self.path} ({owner.strip()})") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "created_at": _now()}, handle)
        return self
    def __exit__(self, *_args):
        self.path.unlink(missing_ok=True)


def _archive_attempt(run_dir: Path) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = run_dir.parent / "_attempts" / run_dir.name / stamp
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(archive))


def _run_dir(experiment: Experiment, job: TrainingSpec, override: Path | None) -> Path:
    root = override if override is not None else Path(job.output.root)
    if not root.is_absolute(): root = PROJECT_ROOT / root
    return root / (Path(job.output.group) if job.output.group else Path()) / experiment.name / job.name / config_hash(job)


def _load_completed_artifacts(run_dir: Path) -> Mapping[str, Path] | None:
    status = _read_json(run_dir / "status.json")
    data = _read_json(run_dir / "artifacts.json")
    if not isinstance(status, dict) or status.get("status") != "completed" or not isinstance(data, dict): return None
    raw = data.get("artifacts")
    if not isinstance(raw, dict): return None
    result = {str(key): Path(value) for key, value in raw.items()}
    return result if all(path.exists() for path in result.values()) else None


def _validate_artifacts(values: Mapping[str, str | Path], run_dir: Path) -> Mapping[str, Path]:
    result = {}
    for name, raw in values.items():
        if not isinstance(name, str) or not name: raise ValueError("artifact names must be non-empty strings")
        path = Path(raw); path = path if path.is_absolute() else run_dir / path; path = path.resolve()
        if not path.exists(): raise RuntimeError(f"runner declared missing artifact {name!r}: {path}")
        result[name] = path
    return result


def _read_json(path: Path) -> Any:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError): return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str: return datetime.now(UTC).isoformat()
