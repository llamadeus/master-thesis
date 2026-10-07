"""Experiment graph validation and dependency ordering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .serialization import canonical_data
from .specs import (ArtifactRef, ControlTrainingBaseSpec, ControlTrainingSpec,
                    ForecastTrainingSpec, TrainingSpec, validate_identifier)


@dataclass(frozen=True)
class Experiment:
    name: str
    jobs: tuple[TrainingSpec, ...]
    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.name, "experiment name")
        if not self.jobs:
            raise ValueError("an experiment must contain at least one job")
        for job in self.jobs:
            if isinstance(job, ControlTrainingBaseSpec):
                raise TypeError("ControlTrainingBaseSpec is not runnable; call .build(...) before adding it to an experiment")
            if not isinstance(job, ForecastTrainingSpec | ControlTrainingSpec):
                raise TypeError("experiment jobs must be ForecastTrainingSpec or ControlTrainingSpec instances")
        names = [job.name for job in self.jobs]
        if len(set(names)) != len(names):
            raise ValueError("experiment job names must be unique")
        known = set(names)
        for job in self.jobs:
            missing = set(job_dependencies(job)) - known
            if missing:
                raise ValueError(f"job {job.name!r} has unknown dependencies: {sorted(missing)}")
        self.ordered_jobs()

    def job(self, name: str) -> TrainingSpec:
        for job in self.jobs:
            if job.name == name:
                return job
        raise KeyError(name)

    def ordered_jobs(self, selected: set[str] | None = None) -> tuple[TrainingSpec, ...]:
        by_name = {job.name: job for job in self.jobs}
        targets = set(by_name) if selected is None else set(selected)
        unknown = targets - set(by_name)
        if unknown:
            raise ValueError(f"unknown selected jobs: {sorted(unknown)}")

        required: set[str] = set()

        def collect(name: str) -> None:
            if name in required:
                return
            required.add(name)
            for dependency in job_dependencies(by_name[name]):
                collect(dependency)

        for target in targets:
            collect(target)

        ordered: list[TrainingSpec] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent or name not in required:
                return
            if name in temporary:
                raise ValueError(f"experiment contains a dependency cycle involving {name!r}")
            temporary.add(name)
            for dependency in job_dependencies(by_name[name]):
                visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(by_name[name])

        # Original declaration order is the stable tie-breaker.
        for job in self.jobs:
            visit(job.name)
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)


def job_dependencies(job: TrainingSpec) -> tuple[str, ...]:
    discovered = _artifact_jobs(job)
    return tuple(dict.fromkeys((*job.dependencies, *discovered)))


def _artifact_jobs(value: Any) -> tuple[str, ...]:
    result: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, ArtifactRef):
            result.append(item.job)
        elif isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                walk(nested)
        elif hasattr(item, "__dataclass_fields__"):
            for name in item.__dataclass_fields__:
                walk(getattr(item, name))

    walk(value)
    return tuple(dict.fromkeys(result))
