"""Policy-independent-window control metrics from docs/decisions/evaluation.md."""

from __future__ import annotations

import numpy as np


def control_metrics(chlorine: np.ndarray, arsenic: np.ndarray, injection: np.ndarray,
                    reference_event: np.ndarray, timestep_seconds: int) -> dict[str, float | None]:
    chlorine=np.asarray(chlorine,dtype=float);arsenic=np.asarray(arsenic,dtype=float);injection=np.asarray(injection,dtype=float)
    event=np.asarray(reference_event,dtype=bool);normal=~event;hours=timestep_seconds/3600
    result={}
    for label,mask in (("normal",normal),("event",event)):
        values=chlorine[mask]
        if not len(values):
            for key in ("below_fraction","above_fraction","mean_deficit","mean_excess","worst_node_deficit","worst_node_excess","max_deficit","max_excess"):result[f"{label}_{key}"]=None
            continue
        deficit=np.maximum(.2-values,0);excess=np.maximum(values-.4,0)
        result.update({f"{label}_below_fraction":float(np.mean(values<.2)),f"{label}_above_fraction":float(np.mean(values>.4)),
            f"{label}_mean_deficit":float(deficit.mean()),f"{label}_mean_excess":float(excess.mean()),
            f"{label}_worst_node_deficit":float(deficit.mean(axis=0).max()),f"{label}_worst_node_excess":float(excess.mean(axis=0).max()),
            f"{label}_max_deficit":float(deficit.max()),f"{label}_max_excess":float(excess.max())})
    transitions=normal[:-1]&normal[1:]
    result["normal_temporal_variation"]=float(np.abs(np.diff(chlorine,axis=0))[transitions].mean()) if np.any(transitions) else None
    result["normal_longest_below_hours"]=_longest(chlorine<.2,normal)*hours
    result["normal_longest_above_hours"]=_longest(chlorine>.4,normal)*hours
    burden=np.maximum(arsenic,0).sum(axis=1)
    result["event_mean_arsenic_burden"]=float(burden[event].mean()) if np.any(event) else None
    result["event_max_arsenic_burden"]=float(burden[event].max()) if np.any(event) else None
    result["mean_injection"]=float(injection.mean())
    result["injection_temporal_variation"]=float(np.abs(np.diff(injection)).mean()) if len(injection)>1 else 0.0
    result["total_injection"]=float(injection.sum()*(timestep_seconds/60))
    return result


def _longest(condition: np.ndarray, eligible: np.ndarray) -> int:
    best=0
    for column in range(condition.shape[1]):
        current=0
        for active in condition[:,column]&eligible:
            current=current+1 if active else 0;best=max(best,current)
    return best
