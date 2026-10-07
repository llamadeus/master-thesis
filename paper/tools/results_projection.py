"""Small publication inputs, with aggregation below the independent-seed level.

Raw measurements belong in experiment artifacts. Keep scenario means and their
applicability counts for control, and model/horizon errors for forecasting.
"""

from collections import defaultdict
import math


FORMAT_VERSION = 1
SEEDS = (17, 42, 123, 1337, 71026)
CONTROL_COMPONENTS = (
    ("chlorine_shortfall_fraction", "main_normal"),
    ("normal_chlorine_excess_fraction", "main_normal"),
    ("arsenic_burden", "main"),
    ("arsenic_burden", "tail"),
    ("arsenic_burden", "total"),
    ("chlorine_ceiling_violation_fraction", "total"),
    ("arsenic_clearance_lead", "total"),
    ("chlorine_use", "main_normal"),
    ("chlorine_use", "main_event"),
    ("chlorine_use", "tail_event"),
    ("chlorine_use", "total"),
)
PRIMARY_COMPONENTS = tuple(
    pair for pair in CONTROL_COMPONENTS
    if pair[1] == "total" or pair[0] in (
        "chlorine_shortfall_fraction", "normal_chlorine_excess_fraction"
    )
)


def scenario_metrics(rows):
    """Average applicable schedules within scenarios, never pool denominators.

    Input rows have approach, seed, scenario, metric, component, and value.
    None is non-applicable. Entirely non-applicable scenarios remain explicit.
    """
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("approach", "seed", "scenario", "metric", "component"))
        groups[key].append(row["value"])
    result = []
    for key, values in sorted(groups.items()):
        valid = [float(value) for value in values if value is not None]
        if not all(math.isfinite(value) for value in valid):
            raise ValueError("Non-finite control value")
        result.append(dict(
            zip(("approach", "seed", "scenario", "metric", "component"), key),
            value=sum(valid) / len(valid) if valid else None,
            applicable_count=len(valid), not_applicable_count=len(values) - len(valid),
        ))
    return result


def forecast_projection(rows, data_kind):
    """Preserve all 25 model errors and 16 horizons, with equal scenario weights.

    Full-horizon RMSE is computed from each scenario's pooled squared errors
    before averaging scenarios. It is not the mean of step-wise RMSE values.
    """
    if data_kind == "pending":
        if rows:
            raise ValueError("Pending forecast data must be empty")
        return dict(format_version=FORMAT_VERSION, data_kind=data_kind, rows=[])
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        seed, fold, step = (int(row[k]) for k in ("training_seed", "fold_id", "horizon_step"))
        if seed not in SEEDS or fold not in range(5) or step not in range(1, 17):
            raise ValueError("Invalid forecast seed, fold, or horizon")
        steps = groups[(seed, fold)][str(row["scenario_id"])]
        if step in steps:
            raise ValueError("Duplicate forecast scenario/horizon statistic")
        count = int(row["observation_count"])
        absolute = float(row["absolute_error_sum_mg_l"])
        squared = float(row["squared_error_sum_mg2_l2"])
        if count <= 0 or not all(math.isfinite(v) and v >= 0 for v in (absolute, squared)):
            raise ValueError("Invalid forecast sufficient statistics")
        steps[step] = (count, absolute, squared)
    if set(groups) != {(seed, fold) for seed in SEEDS for fold in range(5)}:
        raise ValueError("Forecast data requires all 25 seed/fold models")
    result = []
    for (seed, fold), scenarios in sorted(groups.items()):
        if len(scenarios) != 10 or any(set(steps) != set(range(1, 17)) for steps in scenarios.values()):
            raise ValueError("Each forecast model requires ten scenarios and sixteen steps")
        for step in range(17):
            errors = []
            for steps in scenarios.values():
                chosen = steps.values() if step == 0 else [steps[step]]
                count, absolute, squared = map(sum, zip(*chosen))
                errors.append((absolute / count, math.sqrt(squared / count)))
            result.append(dict(training_seed=seed, fold_id=fold, horizon_step=step,
                               mae=sum(e[0] for e in errors) / len(errors),
                               rmse=sum(e[1] for e in errors) / len(errors)))
    return dict(format_version=FORMAT_VERSION, data_kind=data_kind, rows=result)
