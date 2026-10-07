"""Configuration-driven reducers for arbitrary named node targets."""

from __future__ import annotations

import math
import numpy as np

from ..specs import ForecastEvaluationSpec


class ForecastMetricAccumulator:
    """Incremental physical-unit reducers for bounded-memory evaluation."""

    def __init__(self, spec: ForecastEvaluationSpec, *, node_ids: tuple[str, ...] | None = None):
        self.spec = spec; self.node_ids = node_ids; self._values = {}; self._scenarios = {}

    def update(self, predictions: dict[str, np.ndarray], actuals: dict[str, np.ndarray],
               *, scenario_ids: tuple[str, ...] | None = None):
        for target in self.spec.targets:
            if target not in predictions or target not in actuals:
                raise ValueError(f"evaluation target {target!r} is unavailable")
            prediction, actual = np.asarray(predictions[target]), np.asarray(actuals[target])
            if prediction.shape != actual.shape or prediction.ndim != 3:
                raise ValueError("forecast metric inputs must share (sample, horizon, node) shape")
            error = prediction - actual
            target_values = self._values.setdefault(target, {})
            if self.spec.scenario_weighting == "equal" and scenario_ids is None:
                raise ValueError("Equal scenario weighting requires scenario labels")
            if scenario_ids is not None:
                if len(scenario_ids) != len(error):
                    raise ValueError("Scenario labels must match the batch")
                groups = self._scenarios.setdefault(target, {})
                for label in dict.fromkeys(scenario_ids):
                    selected = error[np.asarray(scenario_ids) == label]
                    reducers = groups.setdefault(label, {})
                    for breakdown in ("overall", "horizon", "node"):
                        axes, _ = _breakdown(breakdown, selected, self.node_ids, None)
                        _update_reducer(reducers.setdefault(breakdown, _Reducer()), selected, axes)
            for breakdown in self.spec.breakdowns:
                if breakdown == "scenario":
                    if scenario_ids is None or len(scenario_ids) != len(error):
                        raise ValueError("scenario breakdown requires one label per sample")
                    groups = target_values.setdefault(breakdown, {})
                    for label in dict.fromkeys(scenario_ids):
                        mask = np.asarray([value == label for value in scenario_ids])
                        _update_reducer(groups.setdefault(label, _Reducer()), error[mask], (0, 1, 2))
                else:
                    axes, _labels = _breakdown(breakdown, error, self.node_ids, None)
                    _update_reducer(target_values.setdefault(breakdown, _Reducer()), error, axes)

    def finalize(self):
        result = {}
        for target in self.spec.targets:
            target_result = {}; values = self._values.get(target, {})
            for breakdown in self.spec.breakdowns:
                value = values.get(breakdown)
                if breakdown == "scenario":
                    target_result[breakdown] = {metric: {label: reducer.metric(metric)
                        for label, reducer in value.items()} for metric in self.spec.metrics}
                else:
                    labels = None
                    if breakdown == "node": labels = self.node_ids or tuple(str(i) for i in range(len(value.count)))
                    elif breakdown == "horizon": labels = tuple(str(i) for i in range(len(value.count)))
                    target_result[breakdown] = {metric: _labeled(
                        np.mean([reducers[breakdown].metric(metric) for reducers in self._scenarios[target].values()], axis=0)
                        if self.spec.scenario_weighting == "equal" else value.metric(metric), labels)
                        for metric in self.spec.metrics}
            if target in self._scenarios:
                target_result["scenario_horizon_statistics"] = {
                    scenario: [{"horizon_step": index+1, "observation_count": int(count),
                                "absolute_error_sum_mg_l": float(reducer.absolute[index]),
                                "squared_error_sum_mg2_l2": float(reducer.squared[index])}
                               for index, count in enumerate(reducer.count)]
                    for scenario, reducers in self._scenarios[target].items()
                    for reducer in [reducers["horizon"]]
                }
            result[target] = target_result
        return result


class _Reducer:
    def __init__(self): self.absolute = None; self.squared = None; self.count = None
    def metric(self, name):
        if name == "mae": return self.absolute / self.count
        if name == "mse": return self.squared / self.count
        if name == "rmse": return np.sqrt(self.squared / self.count)
        raise ValueError(f"unsupported forecast metric: {name}")


def _update_reducer(reducer, error, axes):
    absolute = np.sum(np.abs(error), axis=axes, dtype=np.float64)
    squared = np.sum(np.square(error), axis=axes, dtype=np.float64)
    remaining = tuple(index for index in range(error.ndim) if index not in axes)
    count = np.full(tuple(error.shape[index] for index in remaining),
                    math.prod(error.shape[index] for index in axes), dtype=np.int64)
    reducer.absolute = absolute if reducer.absolute is None else reducer.absolute + absolute
    reducer.squared = squared if reducer.squared is None else reducer.squared + squared
    reducer.count = count if reducer.count is None else reducer.count + count


def evaluate_forecasts(predictions: dict[str, np.ndarray], actuals: dict[str, np.ndarray],
                       spec: ForecastEvaluationSpec, *, node_ids: tuple[str, ...] | None = None,
                       scenario_ids: tuple[str, ...] | None = None):
    accumulator = ForecastMetricAccumulator(spec, node_ids=node_ids)
    accumulator.update(predictions, actuals, scenario_ids=scenario_ids)
    return accumulator.finalize()


def _metric(error, metric, axes):
    if metric == "mae": return np.mean(np.abs(error), axis=axes)
    if metric == "mse": return np.mean(np.square(error), axis=axes)
    if metric == "rmse": return np.sqrt(np.mean(np.square(error), axis=axes))
    raise ValueError(f"unsupported forecast metric: {metric}")


def _breakdown(name, values, node_ids, scenario_ids):
    if name == "overall": return (0, 1, 2), None
    if name == "node":
        labels = node_ids or tuple(str(index) for index in range(values.shape[2]))
        if len(labels) != values.shape[2]: raise ValueError("node labels do not match metric inputs")
        return (0, 1), labels
    if name == "horizon": return (0, 2), tuple(str(index) for index in range(values.shape[1]))
    if name == "scenario":
        if scenario_ids is None or len(scenario_ids) != values.shape[0]: raise ValueError("scenario breakdown requires one label per sample")
        return (1, 2), scenario_ids
    raise ValueError(f"unsupported metric breakdown: {name}")


def _labeled(value, labels):
    array = np.asarray(value)
    if labels is None: return float(array)
    return {label: float(item) for label, item in zip(labels, array, strict=True)}
