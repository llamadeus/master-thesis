"""Identity-keyed deterministic arsenic schedule generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from ..serialization import canonical_data
from ..specs import (ArsenicScheduleGeneratorSpec, ArsenicScheduleStreamSpec,
                     FixedArsenicScheduleSetSpec,
                     HydraulicPeakArsenicScheduleGeneratorSpec)


GENERATOR_VERSION = 3
SECONDS_PER_HALF_HOUR = 30 * 60
HALF_HOURS_PER_DAY = 48
HALF_HOURS_PER_WEEK = 7 * HALF_HOURS_PER_DAY
MIN_EVENT_DURATION_HALF_HOURS = 4
MAX_EVENT_DURATION_HALF_HOURS = 144


@dataclass(frozen=True)
class ArsenicEvent:
    node_id: str
    start_time_seconds: int
    end_time_seconds: int
    strength: float
    species: str = "AsIII"
    source_type: str = "MASS"
    target_peak_concentration_mg_per_liter: float | None = None


@dataclass(frozen=True)
class ArsenicSchedule:
    events: tuple[ArsenicEvent, ...]
    hydraulic_profile_identity: str | None = None

    @property
    def hash(self) -> str:
        events = []
        for event in self.events:
            value = asdict(event)
            if event.target_peak_concentration_mg_per_liter is None:
                value.pop("target_peak_concentration_mg_per_liter")
            events.append(value)
        raw: Any = events
        if self.hydraulic_profile_identity is not None:
            raw = {
                "events": events,
                "hydraulic_profile_identity": self.hydraulic_profile_identity,
            }
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class _EventPlacement:
    node_id: str
    start_half_hour: int
    hydraulic_window: tuple[float, float, float, float] | None = None


class ArsenicScheduleStream:
    def __init__(self, spec: ArsenicScheduleStreamSpec, hydraulic_catalog=None):
        self.spec = spec
        self.hydraulic_catalog = (
            hydraulic_catalog if hasattr(hydraulic_catalog, "profile") else None
        )
        self.hydraulic_catalog_hash = (
            hydraulic_catalog if isinstance(hydraulic_catalog, str)
            else getattr(hydraulic_catalog, "catalog_hash", None)
        )
        self.scenario_episode_counts: dict[str, int] = {}

    def next_schedule(
        self,
        eligible_node_ids: tuple[str, ...],
        *,
        scenario_id: str,
        warmup_days: int = 2,
        main_days: int = 21,
        hydraulic_profile=None,
    ) -> ArsenicSchedule:
        if hydraulic_profile is None and self.hydraulic_catalog is not None:
            hydraulic_profile = self.hydraulic_catalog.profile(scenario_id)
        scenario_episode_index=self.scenario_episode_counts.get(scenario_id,0)
        schedule = _generate_schedule(
            eligible_node_ids,
            identity=("stream", self.spec.seed, self.spec.seed_path, scenario_id, scenario_episode_index),
            generator=self.spec.generator,
            warmup_days=warmup_days,
            main_days=main_days,
            hydraulic_profile=hydraulic_profile,
            minimum_event_count=self.spec.minimum_event_count,
            maximum_event_count=self.spec.maximum_event_count,
            minimum_event_duration_steps=self.spec.minimum_event_duration_steps,
            maximum_event_duration_steps=self.spec.maximum_event_duration_steps,
            episode_steps=self.spec.episode_steps,
        )
        self.scenario_episode_counts[scenario_id]=scenario_episode_index+1
        return schedule

    def state_dict(self) -> dict[str, Any]:
        state = {
            "generator_version": GENERATOR_VERSION,
            "seed": self.spec.seed,
            "seed_path": list(self.spec.seed_path),
            "generator": canonical_data(self.spec.generator),
            "minimum_event_count": self.spec.minimum_event_count,
            "maximum_event_count": self.spec.maximum_event_count,
            "minimum_event_duration_steps": self.spec.minimum_event_duration_steps,
            "maximum_event_duration_steps": self.spec.maximum_event_duration_steps,
            "episode_steps": self.spec.episode_steps,
            "scenario_episode_counts": dict(self.scenario_episode_counts),
        }
        if self.hydraulic_catalog_hash is not None:
            state["hydraulic_catalog_hash"] = self.hydraulic_catalog_hash
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(
                "schedule stream identity mismatch: generator version"
            )
        if (int(state["seed"]) != self.spec.seed
                or tuple(state["seed_path"]) != self.spec.seed_path
                or state.get("generator") != canonical_data(self.spec.generator)
                or state.get("minimum_event_count", 1)
                != self.spec.minimum_event_count
                or state.get("maximum_event_count", 4)
                != self.spec.maximum_event_count
                or state.get("minimum_event_duration_steps", 4)
                != self.spec.minimum_event_duration_steps
                or state.get("maximum_event_duration_steps", 144)
                != self.spec.maximum_event_duration_steps
                or state.get("episode_steps") != self.spec.episode_steps
                or state.get("hydraulic_catalog_hash") != self.hydraulic_catalog_hash):
            raise ValueError("schedule stream identity mismatch")
        counts=state["scenario_episode_counts"]
        if not isinstance(counts,dict) or any(not isinstance(key,str) or int(value)<0 for key,value in counts.items()):
            raise ValueError("invalid schedule stream scenario counters")
        self.scenario_episode_counts={key:int(value) for key,value in counts.items()}


class FixedArsenicScheduleIterator:
    """Scenario-bound finite iterator for a fixed schedule set."""

    def __init__(self, spec: FixedArsenicScheduleSetSpec, scenario_id: str,
                 eligible_node_ids: tuple[str, ...], *, warmup_days: int = 2, main_days: int = 21,
                 hydraulic_profile=None):
        self.spec, self.scenario_id, self.eligible_node_ids = spec, scenario_id, eligible_node_ids
        self.warmup_days, self.main_days, self.index = warmup_days, main_days, 0
        self.hydraulic_profile = hydraulic_profile

    def __iter__(self):
        return self

    def __next__(self) -> ArsenicSchedule:
        if self.index >= self.spec.schedule_count:
            raise StopIteration
        result = fixed_schedule(self.spec, self.scenario_id, self.index, self.eligible_node_ids,
                                warmup_days=self.warmup_days, main_days=self.main_days,
                                hydraulic_profile=self.hydraulic_profile)
        self.index += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"generator_version": GENERATOR_VERSION,
                "scenario_id": self.scenario_id, "index": self.index, "spec": asdict(self.spec),
                "hydraulic_profile_identity": _profile_identity(self.hydraulic_profile)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(
                "fixed schedule iterator identity mismatch: generator version"
            )
        if (state.get("scenario_id") != self.scenario_id
                or state.get("spec") != asdict(self.spec)
                or state.get("hydraulic_profile_identity") != _profile_identity(self.hydraulic_profile)):
            raise ValueError("fixed schedule iterator identity mismatch")
        index = state.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= self.spec.schedule_count:
            raise ValueError("invalid fixed schedule iterator index")
        self.index = index


class BoundArsenicScheduleStream:
    """Iterator facade preserving the stateful stream's scenario counters."""

    def __init__(self, stream: ArsenicScheduleStream, scenario_id: str, eligible_node_ids: tuple[str, ...],
                 *, warmup_days: int = 2, main_days: int = 21, hydraulic_profile=None):
        self.stream, self.scenario_id, self.eligible_node_ids = stream, scenario_id, eligible_node_ids
        self.warmup_days, self.main_days = warmup_days, main_days
        self.hydraulic_profile = hydraulic_profile

    def __iter__(self):
        return self

    def __next__(self) -> ArsenicSchedule:
        return self.stream.next_schedule(self.eligible_node_ids, scenario_id=self.scenario_id,
                                         warmup_days=self.warmup_days, main_days=self.main_days,
                                         hydraulic_profile=self.hydraulic_profile)

    def state_dict(self) -> dict[str, Any]:
        return self.stream.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.stream.load_state_dict(state)


def resolve_schedule_source(spec, scenario_id: str, eligible_node_ids: tuple[str, ...],
                            *, warmup_days: int = 2, main_days: int = 21,
                            hydraulic_profile=None):
    _require_profile(spec, hydraulic_profile)
    if isinstance(spec, FixedArsenicScheduleSetSpec):
        return FixedArsenicScheduleIterator(spec, scenario_id, eligible_node_ids,
                                            warmup_days=warmup_days, main_days=main_days,
                                            hydraulic_profile=hydraulic_profile)
    if isinstance(spec, ArsenicScheduleStreamSpec):
        return BoundArsenicScheduleStream(
            ArsenicScheduleStream(spec, _profile_identity(hydraulic_profile)),
            scenario_id, eligible_node_ids, warmup_days=warmup_days,
            main_days=main_days, hydraulic_profile=hydraulic_profile,
        )
    if spec is None:
        return iter((ArsenicSchedule(()),))
    raise TypeError(f"unsupported arsenic schedule source: {type(spec).__name__}")


def spawn_schedule_streams(spec: ArsenicScheduleStreamSpec, count: int, *, hydraulic_catalog=None) -> tuple[ArsenicScheduleStream, ...]:
    if count <= 0:
        raise ValueError("schedule stream count must be positive")
    _require_catalog(spec, hydraulic_catalog)
    return tuple(
        ArsenicScheduleStream(spec.for_environment(index), hydraulic_catalog)
        for index in range(count)
    )


class ArsenicScheduleSource:
    """Control-facing source preserving either finite or unbounded semantics."""
    def __init__(self, spec, hydraulic_catalog=None):
        self.spec = spec
        self.hydraulic_catalog = hydraulic_catalog
        _require_catalog(spec, hydraulic_catalog)
        self.stream = (
            ArsenicScheduleStream(spec, hydraulic_catalog)
            if isinstance(spec, ArsenicScheduleStreamSpec) else None
        )
        self.counts: dict[str, int] = {}
    def next_schedule(self, eligible_node_ids, *, scenario_id, warmup_days=2, main_days=21):
        profile = self.hydraulic_catalog.profile(scenario_id) if self.hydraulic_catalog is not None else None
        if self.stream is not None:
            return self.stream.next_schedule(eligible_node_ids, scenario_id=scenario_id,
                                             warmup_days=warmup_days, main_days=main_days,
                                             hydraulic_profile=profile)
        if isinstance(self.spec, FixedArsenicScheduleSetSpec):
            index = self.counts.get(scenario_id, 0)
            if index >= self.spec.schedule_count: raise StopIteration
            self.counts[scenario_id] = index + 1
            return fixed_schedule(self.spec, scenario_id, index, tuple(eligible_node_ids),
                                  warmup_days=warmup_days, main_days=main_days,
                                  hydraulic_profile=profile)
        return ArsenicSchedule(())
    def state_dict(self):
        return self.stream.state_dict() if self.stream else {
            "generator_version": GENERATOR_VERSION,
            "counts": dict(self.counts), "spec": asdict(self.spec) if self.spec else None,
            "hydraulic_catalog_hash": getattr(self.hydraulic_catalog, "catalog_hash", None),
        }
    def load_state_dict(self, state):
        if self.stream: self.stream.load_state_dict(state); return
        if state.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(
                "schedule source identity mismatch: generator version"
            )
        if (state.get("spec") != (asdict(self.spec) if self.spec else None)
                or state.get("hydraulic_catalog_hash") != getattr(self.hydraulic_catalog, "catalog_hash", None)):
            raise ValueError("schedule source identity mismatch")
        self.counts = {str(key): int(value) for key, value in state.get("counts", {}).items()}


def spawn_schedule_sources(spec, count: int, *, hydraulic_catalog=None):
    if count <= 0: raise ValueError("schedule source count must be positive")
    _require_catalog(spec, hydraulic_catalog)
    if isinstance(spec, ArsenicScheduleStreamSpec):
        return tuple(ArsenicScheduleSource(spec.for_environment(index), hydraulic_catalog) for index in range(count))
    return tuple(ArsenicScheduleSource(spec, hydraulic_catalog) if spec is not None else None for _ in range(count))


def fixed_schedule(
    spec: FixedArsenicScheduleSetSpec,
    scenario_id: str,
    schedule_index: int,
    eligible_node_ids: tuple[str, ...],
    *,
    warmup_days: int = 2,
    main_days: int = 21,
    hydraulic_profile=None,
) -> ArsenicSchedule:
    if not 0 <= schedule_index < spec.schedule_count:
        raise ValueError("fixed arsenic schedule index is out of range")
    return _generate_schedule(
        eligible_node_ids,
        identity=("fixed", spec.seed, scenario_id, schedule_index),
        generator=spec.generator,
        warmup_days=warmup_days,
        main_days=main_days,
        hydraulic_profile=hydraulic_profile,
    )


def _generate_schedule(
    eligible_node_ids,
    *,
    identity,
    generator,
    warmup_days,
    main_days,
    hydraulic_profile=None,
    minimum_event_count=1,
    maximum_event_count=4,
    minimum_event_duration_steps=MIN_EVENT_DURATION_HALF_HOURS,
    maximum_event_duration_steps=MAX_EVENT_DURATION_HALF_HOURS,
    episode_steps=None,
):
    if not eligible_node_ids:
        raise ValueError("eligible_node_ids must not be empty")
    _require_profile_for_generator(generator, hydraulic_profile)
    rng = _DeterministicRandom(_identity_seed(identity))
    event_count = rng.integer(minimum_event_count, maximum_event_count + 1)
    events: list[ArsenicEvent] = []
    main_start_half_hour = warmup_days * HALF_HOURS_PER_DAY
    main_end_half_hour = main_start_half_hour + main_days * HALF_HOURS_PER_DAY
    if episode_steps is not None:
        main_end_half_hour = min(main_end_half_hour, episode_steps)
    available_steps = main_end_half_hour - main_start_half_hour
    maximum_duration_steps = min(maximum_event_duration_steps, available_steps)
    if maximum_duration_steps < minimum_event_duration_steps:
        raise ValueError(
            "arsenic schedule episode is too short for the configured event duration"
        )
    for _ in range(event_count):
        duration_seconds = (
            rng.integer(
                minimum_event_duration_steps,
                maximum_duration_steps + 1,
            )
            * SECONDS_PER_HALF_HOUR
        )
        hydraulic_peak = isinstance(
            generator, HydraulicPeakArsenicScheduleGeneratorSpec
        )
        remaining_week_half_hours = list(range(HALF_HOURS_PER_WEEK))
        placement = None
        rejected_hydraulic_windows = []
        while remaining_week_half_hours:
            slot_index = rng.integer(0, len(remaining_week_half_hours))
            preferred_week_half_hour = remaining_week_half_hours.pop(slot_index)
            placements, rejected = _valid_event_placements(
                events,
                eligible_node_ids,
                main_start_half_hour,
                main_end_half_hour,
                duration_seconds,
                preferred_week_half_hour,
                generator,
                hydraulic_profile,
            )
            rejected_hydraulic_windows.extend(rejected)
            if placements:
                placement = placements[rng.integer(0, len(placements))]
                break
        if placement is None:
            if hydraulic_peak and rejected_hydraulic_windows:
                node_id, start_seconds, minimum_outflow, maximum_outflow, lower, upper = (
                    rejected_hydraulic_windows[0]
                )
                scenario_id = getattr(hydraulic_profile, "scenario_id", "unknown")
                raise ValueError(
                    "could not place a hydraulically feasible arsenic event: "
                    f"scenario={scenario_id}, duration_seconds={duration_seconds}, "
                    f"candidate_nodes={sorted(str(node_id) for node_id in eligible_node_ids)}, "
                    f"rejected_windows={len(rejected_hydraulic_windows)}; "
                    f"example node={node_id}, start_seconds={start_seconds}, "
                    f"minimum_outflow_lpm={minimum_outflow:g}, "
                    f"maximum_outflow_lpm={maximum_outflow:g}, "
                    f"lower_peak_mg_per_liter={lower:g}, "
                    f"upper_peak_mg_per_liter={upper:g}"
                )
            raise RuntimeError("could not place an arsenic event without same-node overlap")
        start_seconds = placement.start_half_hour * SECONDS_PER_HALF_HOUR
        end_seconds = start_seconds + duration_seconds
        if hydraulic_peak:
            assert placement.hydraulic_window is not None
            minimum_outflow, _maximum_outflow, lower_peak, upper_peak = (
                placement.hydraulic_window
            )
            target_peak = _peak_concentration_from_quantile(
                rng.uniform(0.0, 1.0),
                generator,
                lower_bound_mg_per_liter=lower_peak,
                upper_bound_mg_per_liter=upper_peak,
            )
            strength = target_peak * minimum_outflow
        else:
            strength = _sample_strength(rng, generator)
            target_peak = None
        events.append(ArsenicEvent(
            placement.node_id, start_seconds, end_seconds, strength,
            target_peak_concentration_mg_per_liter=target_peak,
        ))
    return ArsenicSchedule(
        tuple(sorted(events,key=lambda event:event.start_time_seconds)),
        _profile_identity(hydraulic_profile),
    )


def _sample_strength(
    rng: _DeterministicRandom,
    generator: ArsenicScheduleGeneratorSpec,
) -> float:
    return _strength_from_quantile(rng.uniform(0.0, 1.0), generator)


def _strength_from_quantile(
    quantile: float,
    generator: ArsenicScheduleGeneratorSpec,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("arsenic strength quantile must be in [0, 1]")
    lower = generator.minimum_strength_mg_per_minute
    upper = generator.maximum_strength_mg_per_minute
    if quantile == 0.0:
        return lower
    if quantile == 1.0:
        return upper
    if generator.strength_distribution == "uniform":
        return lower + (upper - lower) * quantile
    if generator.strength_distribution == "log-uniform":
        return math.exp(math.log(lower) + (math.log(upper) - math.log(lower)) * quantile)
    if generator.strength_distribution == "gaussian":
        distribution = NormalDist(
            generator.strength_mean_mg_per_minute,
            math.sqrt(generator.strength_variance_mg2_per_minute2),
        )
        lower_probability = distribution.cdf(lower)
        probability_span = distribution.cdf(upper) - lower_probability
        if probability_span == 0.0:
            return lower + (upper - lower) * quantile
        probability = lower_probability + probability_span * quantile
        return min(upper, max(lower, distribution.inv_cdf(probability)))
    raise ValueError(f"unsupported arsenic strength distribution: {generator.strength_distribution}")


def _peak_concentration_from_quantile(
    quantile: float,
    generator: HydraulicPeakArsenicScheduleGeneratorSpec,
    *,
    lower_bound_mg_per_liter: float | None = None,
    upper_bound_mg_per_liter: float | None = None,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("arsenic peak-concentration quantile must be in [0, 1]")
    configured_lower = generator.minimum_local_concentration_mg_per_liter
    configured_upper = generator.maximum_local_concentration_mg_per_liter
    lower = (
        configured_lower
        if lower_bound_mg_per_liter is None
        else float(lower_bound_mg_per_liter)
    )
    upper = (
        configured_upper
        if upper_bound_mg_per_liter is None
        else float(upper_bound_mg_per_liter)
    )
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < configured_lower
        or upper > configured_upper
        or lower > upper
    ):
        raise ValueError(
            "arsenic event peak-concentration bounds must be finite, ordered, "
            "and within the configured local-concentration bounds"
        )
    if quantile == 0.0:
        return lower
    if quantile == 1.0:
        return upper
    if lower == upper:
        return lower
    if generator.concentration_distribution == "log-uniform":
        return min(
            upper,
            max(
                lower,
                math.exp(
                    math.log(lower)
                    + (math.log(upper) - math.log(lower)) * quantile
                ),
            ),
        )
    distribution = NormalDist(
        generator.log_peak_mean,
        math.sqrt(generator.log_peak_variance),
    )
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    lower_probability = distribution.cdf(log_lower)
    probability_span = distribution.cdf(log_upper) - lower_probability
    if probability_span == 0.0:
        return lower + (upper - lower) * quantile
    probability = lower_probability + probability_span * quantile
    return min(upper, max(lower, math.exp(distribution.inv_cdf(probability))))


def _generator_requires_hydraulics(generator) -> bool:
    return isinstance(generator, HydraulicPeakArsenicScheduleGeneratorSpec)


def schedule_source_requires_hydraulics(spec) -> bool:
    return isinstance(spec, FixedArsenicScheduleSetSpec | ArsenicScheduleStreamSpec) and _generator_requires_hydraulics(spec.generator)


def _require_profile(spec, hydraulic_profile) -> None:
    if (isinstance(spec, FixedArsenicScheduleSetSpec | ArsenicScheduleStreamSpec)
            and _generator_requires_hydraulics(spec.generator)
            and hydraulic_profile is None):
        raise ValueError("hydraulic arsenic schedule generation requires a hydraulic outflow profile")


def _require_profile_for_generator(generator, hydraulic_profile) -> None:
    if _generator_requires_hydraulics(generator) and hydraulic_profile is None:
        raise ValueError("hydraulic arsenic schedule generation requires a hydraulic outflow profile")


def _require_catalog(spec, hydraulic_catalog) -> None:
    if schedule_source_requires_hydraulics(spec) and hydraulic_catalog is None:
        raise ValueError("hydraulic arsenic schedule generation requires a hydraulic outflow catalog")


def _profile_identity(hydraulic_profile) -> str | None:
    if hydraulic_profile is None:
        return None
    identity = getattr(hydraulic_profile, "identity", None)
    if not isinstance(identity, str) or not identity:
        raise ValueError("hydraulic outflow profile has no valid identity")
    return identity


def _identity_seed(identity) -> bytes:
    payload=json.dumps({"generator_version":GENERATOR_VERSION,"identity":identity},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).digest()


class _DeterministicRandom:
    """Versioned SHA-256 counter RNG with stable sampling operations."""

    def __init__(self, seed: bytes):
        self.seed=seed;self.counter=0

    def _uint64(self) -> int:
        block=hashlib.sha256(self.seed+self.counter.to_bytes(16,"big")).digest()
        self.counter+=1
        return int.from_bytes(block[:8],"big")

    def integer(self, low: int, high: int) -> int:
        span=high-low
        if span<=0:raise ValueError("integer range must be non-empty")
        limit=(1<<64)-((1<<64)%span)
        while True:
            value=self._uint64()
            if value<limit:return low+value%span

    def uniform(self, low: float, high: float) -> float:
        unit=(self._uint64()>>11)*(1.0/(1<<53))
        return low+(high-low)*unit

    def permutation(self, values) -> list[str]:
        result=[str(value) for value in values]
        for index in range(len(result)-1,0,-1):
            other=self.integer(0,index+1)
            result[index],result[other]=result[other],result[index]
        return result


def _valid_start_half_hours(
    events: list[ArsenicEvent],
    node_id: str,
    main_start_half_hour: int,
    main_end_half_hour: int,
    duration_seconds: int,
) -> list[int]:
    result = []
    latest_start_half_hour = (
        main_end_half_hour * SECONDS_PER_HALF_HOUR - duration_seconds
    ) // SECONDS_PER_HALF_HOUR
    for start in range(main_start_half_hour, latest_start_half_hour + 1):
        start_seconds = start * SECONDS_PER_HALF_HOUR
        end_seconds = start_seconds + duration_seconds
        overlaps = any(
            event.node_id == node_id
            and start_seconds < event.end_time_seconds
            and end_seconds > event.start_time_seconds
            for event in events
        )
        if not overlaps:
            result.append(start)
    return result


def _valid_event_placements(
    events: list[ArsenicEvent],
    eligible_node_ids,
    main_start_half_hour: int,
    main_end_half_hour: int,
    duration_seconds: int,
    preferred_week_half_hour: int,
    generator,
    hydraulic_profile=None,
) -> tuple[list[_EventPlacement], list[tuple]]:
    placements = []
    rejected = []
    hydraulic_peak = isinstance(
        generator, HydraulicPeakArsenicScheduleGeneratorSpec
    )
    for node_id in sorted(str(value) for value in eligible_node_ids):
        preferred = [
            half_hour
            for half_hour in _valid_start_half_hours(
                events,
                node_id,
                main_start_half_hour,
                main_end_half_hour,
                duration_seconds,
            )
            if half_hour % HALF_HOURS_PER_WEEK == preferred_week_half_hour
        ]
        if not hydraulic_peak:
            placements.extend(
                _EventPlacement(node_id, half_hour) for half_hour in preferred
            )
            continue
        choices, windows, node_rejected = (
            _hydraulically_feasible_start_half_hours(
                hydraulic_profile,
                node_id,
                preferred,
                duration_seconds,
                generator,
            )
        )
        rejected.extend(node_rejected)
        placements.extend(
            _EventPlacement(node_id, half_hour, windows[half_hour])
            for half_hour in choices
        )
    return placements, rejected


def _hydraulically_feasible_start_half_hours(
    hydraulic_profile,
    node_id: str,
    start_half_hours: list[int],
    duration_seconds: int,
    generator: HydraulicPeakArsenicScheduleGeneratorSpec,
):
    feasible = []
    windows = {}
    rejected = []
    for start_half_hour in start_half_hours:
        start_seconds = start_half_hour * SECONDS_PER_HALF_HOUR
        end_seconds = start_seconds + duration_seconds
        minimum_outflow, maximum_outflow = (
            hydraulic_profile.outflow_range_liters_per_minute(
                node_id, start_seconds, end_seconds
            )
        )
        lower_peak = (
            generator.minimum_local_concentration_mg_per_liter
            * maximum_outflow
            / minimum_outflow
        )
        upper_peak = min(
            generator.maximum_local_concentration_mg_per_liter,
            generator.maximum_mass_rate_mg_per_minute / minimum_outflow,
        )
        window = (
            minimum_outflow,
            maximum_outflow,
            lower_peak,
            upper_peak,
        )
        if lower_peak <= upper_peak:
            feasible.append(start_half_hour)
            windows[start_half_hour] = window
        else:
            rejected.append(
                (
                    node_id,
                    start_seconds,
                    minimum_outflow,
                    maximum_outflow,
                    lower_peak,
                    upper_peak,
                )
            )
    return feasible, windows, rejected
