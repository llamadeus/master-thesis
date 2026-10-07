"""Resolve named Python datasets through per-scenario manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .specs import (ArsenicScheduleSourceSpec, ChlorineInjectionScheduleSourceSpec,
                    DatasetSpec, DatasetSplitSpec)
from .scenarios import (SCENARIO_MANIFEST_VERSION, ensure_scenario,
                        scenario_number, scenario_content_hash, validate_source)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARMUP_DAYS = 2
MAIN_DAYS = 21
TAIL_DAYS = 2
_EPANET_NODE_SECTIONS = {"[JUNCTIONS]", "[RESERVOIRS]", "[TANKS]"}
Split = Literal["train", "evaluation", "test"]


@dataclass(frozen=True)
class ForecastRunEntry:
    cache_dir: Path
    schedule_seed: int
    chlorine_profile_hash: str
    arsenic_schedule_hash: str


@dataclass(frozen=True)
class ScenarioEntry:
    scenario_id: str
    split: Split
    path: Path
    calendar_start_day: int
    chlorine_source_ids: tuple[str, ...]
    monitored_node_ids: tuple[str, ...]
    eligible_arsenic_node_ids: tuple[str, ...]
    forecast_runs: tuple[ForecastRunEntry, ...]
    content_hash: str = ""

    @property
    def eligible_arsenic_nodes(self) -> tuple[str, ...]:
        return self.eligible_arsenic_node_ids


@dataclass(frozen=True)
class ResolvedDataset:
    network: str
    hydraulic_timestep_seconds: int
    duration_seconds: int
    scenarios: tuple[ScenarioEntry, ...]
    train_schedule: ArsenicScheduleSourceSpec | None
    validation_schedule: ArsenicScheduleSourceSpec | None
    test_schedule: ArsenicScheduleSourceSpec | None
    train_chlorine_schedule: ChlorineInjectionScheduleSourceSpec | None = None
    validation_chlorine_schedule: ChlorineInjectionScheduleSourceSpec | None = None
    test_chlorine_schedule: ChlorineInjectionScheduleSourceSpec | None = None

    @property
    def dataset_id(self) -> str:
        return self.network

    @property
    def warmup_days(self) -> int:
        return WARMUP_DAYS

    @property
    def main_days(self) -> int:
        return MAIN_DAYS

    @property
    def tail_days(self) -> int:
        return TAIL_DAYS

    def scenarios_for(self, split: str) -> tuple[ScenarioEntry, ...]:
        return tuple(item for item in self.scenarios if item.split == split)

    def forecast_runs_for(self, split: str) -> tuple[ForecastRunEntry, ...]:
        return tuple(run for scenario in self.scenarios_for(split) for run in scenario.forecast_runs)

    def schedule_for(self, split: str) -> ArsenicScheduleSourceSpec | None:
        if split == "train": return self.train_schedule
        if split == "evaluation": return self.validation_schedule
        if split == "test": return self.test_schedule
        raise ValueError(f"unknown dataset split: {split}")

    def chlorine_schedule_for(self, split: str) -> ChlorineInjectionScheduleSourceSpec | None:
        if split == "train": return self.train_chlorine_schedule
        if split == "evaluation": return self.validation_chlorine_schedule
        if split == "test": return self.test_chlorine_schedule
        raise ValueError(f"unknown dataset split: {split}")


@dataclass(frozen=True)
class ResolvedDatasetSplit:
    """A directly resolved dataset split without placeholder sibling splits."""

    network: str
    hydraulic_timestep_seconds: int
    duration_seconds: int
    split: Split
    scenarios: tuple[ScenarioEntry, ...]
    schedule: ArsenicScheduleSourceSpec | None
    chlorine_schedule: ChlorineInjectionScheduleSourceSpec | None = None

    @property
    def dataset_id(self) -> str:
        return self.network

    @property
    def warmup_days(self) -> int:
        return WARMUP_DAYS

    @property
    def main_days(self) -> int:
        return MAIN_DAYS

    @property
    def tail_days(self) -> int:
        return TAIL_DAYS

    def scenarios_for(self, split: str) -> tuple[ScenarioEntry, ...]:
        return self.scenarios if split == self.split else ()

    def forecast_runs_for(self, split: str) -> tuple[ForecastRunEntry, ...]:
        return tuple(
            run for scenario in self.scenarios_for(split) for run in scenario.forecast_runs
        )

    def schedule_for(self, split: str) -> ArsenicScheduleSourceSpec | None:
        if split != self.split:
            raise ValueError(
                f"resolved dataset split {self.split!r} does not contain {split!r}"
            )
        return self.schedule

    def chlorine_schedule_for(
        self, split: str
    ) -> ChlorineInjectionScheduleSourceSpec | None:
        if split != self.split:
            raise ValueError(
                f"resolved dataset split {self.split!r} does not contain {split!r}"
            )
        return self.chlorine_schedule


def validate_dataset_source(spec: DatasetSpec) -> None:
    """Check the source configuration without downloading or materializing anything."""
    validate_source(spec.network, spec.scenarios_dir, spec.uncertainty_seed)
    for split in (spec.train, spec.validation, spec.test):
        for reference in split.scenarios:
            scenario_number(reference)


def resolve_dataset(spec: DatasetSpec) -> ResolvedDataset:
    validate_dataset_source(spec)
    scenarios=[]
    for split, split_spec in (("train",spec.train),("evaluation",spec.validation),("test",spec.test)):
        scenarios.extend(_scenario(spec.network,reference,split, scenarios_dir=spec.scenarios_dir,
                                   uncertainty_seed=spec.uncertainty_seed) for reference in split_spec.scenarios)
    timestep={item[1] for item in scenarios}
    duration={item[2] for item in scenarios}
    if len(timestep)!=1:raise ValueError("dataset scenarios have different hydraulic timesteps")
    if len(duration)!=1:raise ValueError("dataset scenarios have different durations")
    expected=(WARMUP_DAYS+MAIN_DAYS+TAIL_DAYS)*86400
    if next(iter(duration))!=expected:raise ValueError(f"baseline control requires {expected} second scenarios")
    return ResolvedDataset(spec.network,next(iter(timestep)),next(iter(duration)),tuple(item[0] for item in scenarios),
                           spec.train.arsenic,spec.validation.arsenic,spec.test.arsenic,
                           spec.train.chlorine,spec.validation.chlorine,spec.test.chlorine)


def resolve_dataset_split(
    network: str,
    spec: DatasetSplitSpec,
    *,
    split: Split = "train",
    scenarios_dir: str = "hanoi",
    uncertainty_seed: int = 42,
) -> ResolvedDatasetSplit:
    """Resolve one split directly, without manufacturing a ``DatasetSpec``."""
    scenarios = [_scenario(network, reference, split, scenarios_dir=scenarios_dir,
                          uncertainty_seed=uncertainty_seed) for reference in spec.scenarios]
    timestep, duration = _shared_scenario_timing(scenarios)
    return ResolvedDatasetSplit(
        network=network,
        hydraulic_timestep_seconds=timestep,
        duration_seconds=duration,
        split=split,
        scenarios=tuple(item[0] for item in scenarios),
        schedule=spec.arsenic,
        chlorine_schedule=spec.chlorine,
    )


def _shared_scenario_timing(
    scenarios: list[tuple[ScenarioEntry, int, int]],
) -> tuple[int, int]:
    timestep = {item[1] for item in scenarios}
    duration = {item[2] for item in scenarios}
    if len(timestep) != 1:
        raise ValueError("dataset scenarios have different hydraulic timesteps")
    if len(duration) != 1:
        raise ValueError("dataset scenarios have different durations")
    expected = (WARMUP_DAYS + MAIN_DAYS + TAIL_DAYS) * 86400
    resolved_duration = next(iter(duration))
    if resolved_duration != expected:
        raise ValueError(
            f"baseline control requires {expected} second scenarios"
        )
    return next(iter(timestep)), resolved_duration


def _scenario(network: str, reference: str, split: Split, *, scenarios_dir: str,
              uncertainty_seed: int) -> tuple[ScenarioEntry,int,int]:
    path=ensure_scenario(network, reference, scenarios_dir=scenarios_dir,
                         uncertainty_seed=uncertainty_seed, project_root=PROJECT_ROOT)
    manifest_path=path/"manifest.json"
    try:raw=json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:raise ValueError(f"scenario manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:raise ValueError(f"invalid scenario manifest: {manifest_path}: {exc}") from exc
    if not isinstance(raw,dict):raise ValueError(f"scenario manifest must be an object: {manifest_path}")
    if raw.get("schema_version")!=SCENARIO_MANIFEST_VERSION:raise ValueError(f"unsupported scenario manifest version: {manifest_path}")
    if _string(raw,"network")!=network:raise ValueError(f"scenario network mismatch: {manifest_path}")
    if _string(raw,"id")!=reference:raise ValueError(f"scenario ID mismatch: {manifest_path}")
    sources=tuple(_string_list(raw,"chlorine_source_ids"))
    monitored=tuple(_string_list(raw,"monitored_node_ids"))
    eligible=tuple(_string_list(raw,"eligible_arsenic_node_ids"))
    topology_nodes=_epanet_node_ids(path/"scenario.inp")
    missing_sources=set(sources)-topology_nodes
    if missing_sources:
        raise ValueError(f"scenario chlorine sources are absent from the network topology: {sorted(missing_sources)}: {manifest_path}")
    if not set(eligible).issubset(monitored):
        raise ValueError(f"eligible arsenic nodes must be monitored: {manifest_path}")
    if set(sources)&set(eligible):raise ValueError(f"scenario allows arsenic at a chlorine source: {manifest_path}")
    runs=tuple(_forecast_run(value,path,index) for index,value in enumerate(raw.get("forecast_runs",[])))
    calendar=raw.get("calendar")
    if not isinstance(calendar,dict):raise ValueError(f"scenario calendar must be an object: {manifest_path}")
    start_day=_non_negative_int(calendar,"start_day")
    if start_day!=0:raise ValueError(f"scenario calendar start_day must be zero: {manifest_path}")
    entry=ScenarioEntry(reference,split,path,start_day,sources,monitored,eligible,runs,scenario_content_hash(path))
    return entry,_positive_int(raw,"hydraulic_timestep_seconds"),_positive_int(raw,"duration_seconds")


def _forecast_run(value: Any, base: Path, index: int) -> ForecastRunEntry:
    if not isinstance(value,dict):raise ValueError(f"forecast_runs[{index}] must be an object")
    cache=_existing_directory(base,_string(value,"cache_dir"),f"forecast_runs[{index}].cache_dir")
    if not (cache/"trajectory.npz").is_file() or not (cache/"metadata.json").is_file():
        raise ValueError(f"canonical forecast cache files are missing: {cache}")
    return ForecastRunEntry(cache,_integer(value.get("schedule_seed"),f"forecast_runs[{index}].schedule_seed"),
        _string(value,"chlorine_profile_hash"),_string(value,"arsenic_schedule_hash"))


def _existing_directory(base: Path, raw: str, label: str) -> Path:
    path=Path(raw);path=(path if path.is_absolute() else base/path).resolve()
    if not path.is_dir():raise ValueError(f"{label} does not exist: {path}")
    return path


def _string(value: dict[str,Any], key: str) -> str:
    result=value.get(key)
    if not isinstance(result,str) or not result:raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_list(value: dict[str,Any], key: str) -> list[str]:
    result=value.get(key)
    if not isinstance(result,list) or not result or not all(isinstance(item,str) and item for item in result):
        raise ValueError(f"{key} must be a non-empty array of strings")
    if len(set(result))!=len(result):raise ValueError(f"{key} contains duplicates")
    return result


def _epanet_node_ids(path: Path) -> frozenset[str]:
    try: lines=path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:raise ValueError(f"scenario network topology not found: {path}") from exc
    section=None;nodes=set()
    for raw_line in lines:
        line=raw_line.split(";",1)[0].strip()
        if not line:continue
        if line.startswith("[") and line.endswith("]"):
            section=line.upper();continue
        if section in _EPANET_NODE_SECTIONS:nodes.add(line.split(maxsplit=1)[0])
    if not nodes:raise ValueError(f"scenario network topology contains no nodes: {path}")
    return frozenset(nodes)


def _integer(value: Any, label: str) -> int:
    if isinstance(value,bool) or not isinstance(value,int):raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: dict[str,Any], key: str) -> int:
    result=_integer(value.get(key),key)
    if result<=0:raise ValueError(f"{key} must be positive")
    return result


def _non_negative_int(value: dict[str,Any], key: str) -> int:
    result=_integer(value.get(key),key)
    if result<0:raise ValueError(f"{key} must be non-negative")
    return result
