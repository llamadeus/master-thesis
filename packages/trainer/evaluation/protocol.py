"""Shared physical scoring and case-level reference protocol.

Times are report ends: the sample at the warm-up boundary is excluded, the
sample at the main-window boundary belongs to Main, and the final report is Tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PROTOCOL_VERSION = 3
THRESHOLD = 0.01
PRIMARY_METRICS = (
    "chlorine_shortfall_fraction", "normal_chlorine_excess_fraction",
    "arsenic_burden", "chlorine_ceiling_violation_fraction",
    "sustained_chlorine_excess", "chlorine_use",
)


class ClearanceError(ValueError):
    """A fixed case is incomplete, never silently excluded or replaced."""


@dataclass
class ReferenceEvaluation:
    times: np.ndarray
    event: np.ndarray
    post_source: np.ndarray
    scenario_id: str
    schedule_hash: str
    clearance_time: float
    transported: np.ndarray
    strengths: np.ndarray
    outflow: np.ndarray
    flow_units: str
    raw_arsenic: np.ndarray | None = None


def scoring_windows(times, *, warmup_days=2, main_days=21, duration_seconds=25*86400):
    times = np.asarray(times, dtype=float)
    start = warmup_days * 86400
    boundary = (warmup_days + main_days) * 86400
    return ((times > start) & (times <= boundary),
            (times > boundary) & (times <= duration_seconds))


def clearance_time(times, raw_arsenic, *, tail_start=23*86400, episode_end=25*86400):
    times = np.asarray(times, dtype=float)
    raw = np.asarray(raw_arsenic, dtype=float)
    if (raw.ndim != 2 or len(raw) != len(times) or not len(times)
            or not np.isfinite(raw).all() or times[-1] != episode_end):
        raise ClearanceError("Reference has invalid concentrations or does not cover the episode end")
    below = np.maximum(raw, 0).sum(axis=1) <= THRESHOLD
    persistent = np.logical_and.accumulate(below[::-1])[::-1]
    eligible = np.flatnonzero((times > tail_start) & (times <= episode_end) & persistent)
    if not len(eligible):
        raise ClearanceError(f"No raw As(III) clearance within the existing episode; final sum={np.maximum(raw[-1], 0).sum():.9g} mg/L")
    return float(times[eligible[0]])


@dataclass(frozen=True)
class ReferenceCluster:
    source_start_seconds: float
    next_source_start_seconds: float
    window_start_seconds: float
    clearance_time_seconds: float


def reference_clusters(times, transported, events):
    """Group injections only while transported arsenic has not cleared persistently."""
    times = np.asarray(times, dtype=float)
    if (len(times) < 2 or times[1] <= times[0]
            or not np.allclose(np.diff(times), times[1]-times[0])):
        raise ValueError("Reference requires a regular reporting grid")
    step = times[1]-times[0]
    aggregate = np.asarray(transported, dtype=float).sum(axis=1)
    ordered = sorted(events, key=lambda event: event.start_time_seconds)
    clusters, position = [], 0
    while position < len(ordered):
        start = ordered[position].start_time_seconds
        position += 1
        while True:
            next_start = ordered[position].start_time_seconds if position < len(ordered) else times[-1]+step
            above = np.flatnonzero((times >= start) & (times < next_start) & (aggregate > THRESHOLD))
            end = times[above[-1]]+step if len(above) else start
            if position < len(ordered) and end >= next_start:
                position += 1
                continue
            if len(above):
                if above[-1] == len(times)-1:
                    raise ClearanceError(f"No observed transported As(III) clearance for cluster starting at {start}")
                clusters.append(ReferenceCluster(float(start), float(next_start),
                                                 float(times[above[0]]), float(end)))
            break
    return clusters


def reference_windows(times, transported, events):
    """Continuous threshold-defined windows, retaining temporary concentration dips."""
    times = np.asarray(times, dtype=float)
    events = tuple(events)
    event_mask = np.zeros(len(times), dtype=bool)
    source_active = np.zeros(len(times), dtype=bool)
    for event in events:
        source_active |= (times >= event.start_time_seconds) & (times < event.end_time_seconds)
    for cluster in reference_clusters(times, transported, events):
        event_mask |= (times >= cluster.window_start_seconds) & (times < cluster.clearance_time_seconds)
    return event_mask, event_mask & ~source_active


def physical_metrics(times, chlorine, transported, injection, reference_event, post_source,
                     timestep_seconds, *, warmup_days=2, main_days=21, duration_seconds=25*86400):
    times = np.asarray(times, dtype=float)
    chlorine = np.asarray(chlorine, dtype=float)
    arsenic = np.asarray(transported, dtype=float)
    injection = np.asarray(injection, dtype=float)
    event = np.asarray(reference_event, dtype=bool)
    post = np.asarray(post_source, dtype=bool)
    if (chlorine.ndim != 2 or arsenic.shape != chlorine.shape
            or any(len(x) != len(times) for x in (chlorine, injection, event, post))
            or not all(np.isfinite(x).all() for x in (chlorine, arsenic, injection))):
        raise ValueError("Invalid or unaligned physical metric inputs")
    main, tail = scoring_windows(times, warmup_days=warmup_days, main_days=main_days,
                                 duration_seconds=duration_seconds)
    normal = main & ~event
    counts, values, components = {}, {}, {}
    def fraction(name, numerator, denominator):
        counts[name] = {"numerator": int(numerator), "denominator": int(denominator)}
        return float(numerator/denominator) if denominator else None
    n = chlorine.shape[1]
    values[PRIMARY_METRICS[0]] = fraction(PRIMARY_METRICS[0], np.sum(chlorine[normal] < .2), normal.sum()*n)
    values[PRIMARY_METRICS[1]] = fraction(PRIMARY_METRICS[1], np.sum(chlorine[normal] > .4), normal.sum()*n)
    hours, minutes = timestep_seconds/3600, timestep_seconds/60
    caps = []
    for window, mask in (("main", main), ("tail", tail)):
        selected = mask & event
        components[f"arsenic_burden_{window}"] = float(hours*np.maximum(arsenic[selected], 0).sum())
        components[f"sustained_chlorine_excess_{window}"] = float(hours*np.maximum(chlorine[mask & post]-.4, 0).sum())
        numerator, denominator = np.sum(chlorine[selected] > 5), selected.sum()*n
        caps.append((numerator, denominator))
        key = f"chlorine_ceiling_violation_fraction_{window}"
        components[key] = fraction(key, numerator, denominator)
        components[f"chlorine_use_{window}_event"] = float(minutes*injection[selected].sum())
    components["chlorine_use_main_normal"] = float(minutes*injection[normal].sum())
    for metric in ("arsenic_burden", "sustained_chlorine_excess"):
        values[metric] = components[f"{metric}_main"]+components[f"{metric}_tail"]
    values["chlorine_ceiling_violation_fraction"] = fraction("chlorine_ceiling_violation_fraction", sum(x[0] for x in caps), sum(x[1] for x in caps))
    values["chlorine_use"] = sum(components[f"chlorine_use_{part}"] for part in ("main_normal", "main_event", "tail_event"))
    for key in ("arsenic_burden", "sustained_chlorine_excess", "chlorine_use"):
        gate = (main | tail) & (post if key == "sustained_chlorine_excess" else event)
        if key == "chlorine_use": gate |= main
        counts[key] = {"eligible_observations": int(gate.sum() * (1 if key == "chlorine_use" else n))}
    return {"values": values, "components": components, "counts": counts}


def scenario_average(records, value):
    grouped = {}
    for record in records:
        item = value(record)
        if item is not None:
            grouped.setdefault(record["scenario_id"], []).append(float(item))
    return float(np.mean([np.mean(items) for items in grouped.values()])) if grouped else None


def summarize_cases(records):
    keys = set().union(*(record["values"].keys() | record["components"].keys() for record in records))
    summary = {key: scenario_average(records, lambda r: (r["values"] | r["components"]).get(key)) for key in sorted(keys)}
    summary["applicability"] = {
        key: {"applicable_rollouts": sum(r["values"].get(key) is not None for r in records),
              "not_applicable_rollouts": sum(r["values"].get(key) is None for r in records),
              "eligible_observations": sum(r["counts"].get(key, {}).get("denominator", r["counts"].get(key, {}).get("eligible_observations", 0)) for r in records)}
        for key in PRIMARY_METRICS
    }
    summary["cases"] = records
    return summary


def next_training_chunk(completed, budget, evaluation_frequency, checkpoint_frequency):
    next_eval = (completed//evaluation_frequency+1)*evaluation_frequency
    next_save = (completed//checkpoint_frequency+1)*checkpoint_frequency
    return min(next_eval, next_save, budget)-completed


def crossed_thresholds(previous, current, frequency, budget):
    return [step for step in range(0, budget+1, frequency) if previous < step <= current] + (
        [budget] if budget % frequency and previous < budget <= current else [])
