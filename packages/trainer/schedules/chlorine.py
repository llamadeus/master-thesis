"""Deterministic, specification-driven chlorine injection schedules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from ..constants import V70_MAX_CHLORINE_DELTA
from ..serialization import canonical_data
from ..specs import FixedChlorineInjectionScheduleSetSpec
from .arsenic import _DeterministicRandom


GENERATOR_VERSION = 2
PIECEWISE_CONSTANT_WINDOW_STEPS = 12


@dataclass(frozen=True)
class ChlorineInjectionSchedule:
    values: tuple[float, ...]
    timestep_seconds: int
    families: tuple[str, ...]

    @property
    def hash(self) -> str:
        payload = json.dumps({"version": GENERATOR_VERSION, "values": self.values,
                              "timestep_seconds": self.timestep_seconds, "families": self.families},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class FixedChlorineInjectionScheduleIterator:
    def __init__(self, spec: FixedChlorineInjectionScheduleSetSpec, scenario_id: str,
                 *, step_count: int, timestep_seconds: int):
        if step_count <= 0 or timestep_seconds <= 0:
            raise ValueError("chlorine schedule step count and timestep must be positive")
        self.spec, self.scenario_id = spec, scenario_id
        self.step_count, self.timestep_seconds, self.index = step_count, timestep_seconds, 0

    def __iter__(self): return self

    def __next__(self) -> ChlorineInjectionSchedule:
        if self.index >= self.spec.schedule_count:
            raise StopIteration
        result = fixed_chlorine_schedule(self.spec, self.scenario_id, self.index,
                                         step_count=self.step_count, timestep_seconds=self.timestep_seconds)
        self.index += 1
        return result

    def state_dict(self):
        return {"scenario_id": self.scenario_id, "index": self.index, "spec": canonical_data(self.spec),
                "step_count": self.step_count, "timestep_seconds": self.timestep_seconds}

    def load_state_dict(self, state):
        expected = {"scenario_id": self.scenario_id, "spec": canonical_data(self.spec),
                    "step_count": self.step_count, "timestep_seconds": self.timestep_seconds}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("chlorine schedule iterator identity mismatch")
        index = state.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= self.spec.schedule_count:
            raise ValueError("invalid chlorine schedule iterator index")
        self.index = index


def fixed_chlorine_schedule(spec: FixedChlorineInjectionScheduleSetSpec, scenario_id: str,
                            schedule_index: int, *, step_count: int,
                            timestep_seconds: int) -> ChlorineInjectionSchedule:
    if not 0 <= schedule_index < spec.schedule_count:
        raise ValueError("fixed chlorine schedule index is out of range")
    identity = {"generator_version": GENERATOR_VERSION, "spec": canonical_data(spec),
                "scenario_id": scenario_id, "schedule_index": schedule_index,
                "step_count": step_count, "timestep_seconds": timestep_seconds}
    seed = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).digest()
    rng = _DeterministicRandom(seed)
    generator = spec.generator
    families = generator.families
    if generator.mix_families_within_schedule:
        selected = tuple(rng.permutation(families))
    elif generator.set_coverage == "balanced":
        selected = (families[schedule_index % len(families)],)
    else:
        selected = (families[rng.integer(0, len(families))],)
    lo, hi = float(generator.minimum_injection), float(generator.maximum_injection)
    values = np.empty(step_count, dtype=np.float64)
    segment = max(1, math.ceil(step_count / len(selected)))
    initial_lo, initial_hi = _sampling_interval(generator.sampling_bands, lo, hi, rng)
    previous = rng.uniform(initial_lo, initial_hi)
    for family_index, family in enumerate(selected):
        start, end = family_index * segment, min(step_count, (family_index + 1) * segment)
        if start >= end: break
        family_lo, family_hi = _sampling_interval(
            generator.sampling_bands, lo, hi, rng
        )
        generated = _family_values(
            family, end - start, family_lo, family_hi, rng, previous
        )
        style = generator.transition_style
        use_ramp = "smooth" in style or "ramp" in style
        if "occasional" in style: use_ramp = use_ramp and rng.integer(0, 4) == 0
        if start and use_ramp:
            ramp = min(len(generated), max(1, len(generated) // 4))
            generated[:ramp] = np.linspace(previous, float(generated[ramp - 1]), ramp)
        values[start:end] = generated
        previous = float(generated[-1])
    _limit_step_changes(values, V70_MAX_CHLORINE_DELTA)
    if np.any(~np.isfinite(values)) or np.any(values < lo) or np.any(values > hi):
        raise ValueError("generated chlorine injections violate configured bounds")
    return ChlorineInjectionSchedule(tuple(float(value) for value in values), timestep_seconds, selected)


def _limit_step_changes(values: np.ndarray, maximum_delta: float) -> None:
    for index in range(len(values)):
        previous = 0.0 if index == 0 else values[index - 1]
        lower = np.nextafter(previous - maximum_delta, previous)
        upper = np.nextafter(previous + maximum_delta, previous)
        values[index] = np.clip(values[index], lower, upper)


def _sampling_interval(bands, lo, hi, rng):
    if not bands:
        return lo, hi
    total = sum(float(weight) for _lower, _upper, weight in bands)
    draw = rng.uniform(0.0, total)
    cumulative = 0.0
    for lower, upper, weight in bands:
        cumulative += float(weight)
        if draw <= cumulative:
            return float(lower), float(upper)
    lower, upper, _weight = bands[-1]
    return float(lower), float(upper)


def _family_values(family, count, lo, hi, rng, previous):
    span = hi - lo
    if family == "constant":
        return np.full(count, rng.uniform(lo, hi))
    if family == "periodic":
        period = max(2, rng.integer(2, max(3, count + 1)))
        phase = rng.uniform(0.0, 2 * math.pi)
        center, amplitude = rng.uniform(lo, hi), rng.uniform(0.0, span / 2)
        return np.clip(center + amplitude * np.sin(np.arange(count) * 2 * math.pi / period + phase), lo, hi)
    if family == "piecewise-constant":
        values = np.empty(count); value = previous
        for index in range(count):
            if index % PIECEWISE_CONSTANT_WINDOW_STEPS == 0: value = rng.uniform(lo, hi)
            values[index] = value
        return values
    if family == "stairs":
        end = rng.uniform(lo, hi)
        return np.linspace(previous, end, count)
    if family == "random-walk":
        values = np.empty(count); value = previous
        for index in range(count):
            value = min(hi, max(lo, value + rng.uniform(-span / 10, span / 10)))
            values[index] = value
        return values
    raise ValueError(f"unsupported chlorine schedule family: {family}")
