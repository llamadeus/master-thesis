"""Plot simulator-neutral control evaluation trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray

from ..specs import ForecastPlotsSpec


@dataclass(frozen=True)
class ControlEvaluationTrajectory:
    time_seconds: NDArray[np.floating]
    node_ids: tuple[str, ...]
    chlorine: NDArray[np.floating]
    arsenic: NDArray[np.floating]
    injection: NDArray[np.floating]
    actions: NDArray[np.floating]
    event_active: NDArray[np.bool_]
    source_local_arsenic: NDArray[np.bool_] | None = None


def render_control_trajectory(path: Path, trajectory: ControlEvaluationTrajectory, title: str) -> None:
    """Render the three-panel layout used by the legacy reward comparison."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    hours = np.asarray(trajectory.time_seconds, dtype=float) / 3600.0
    chlorine = np.asarray(trajectory.chlorine, dtype=float)
    arsenic = np.asarray(trajectory.arsenic, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(24, 16), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, max(chlorine.shape[1], 1)))
    for index in range(chlorine.shape[1]):
        label = trajectory.node_ids[index] if index < len(trajectory.node_ids) else str(index)
        axes[0].plot(hours, chlorine[:, index], color=colors[index], lw=1.1, alpha=0.8, label=label)
    _set_chlorine_axis_limits(axes[0])
    if arsenic.size:
        colors_as = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, max(arsenic.shape[1], 1)))
        source_local = (np.zeros_like(arsenic, dtype=bool)
                        if trajectory.source_local_arsenic is None
                        else np.asarray(trajectory.source_local_arsenic, dtype=bool))
        if source_local.shape != arsenic.shape:
            raise ValueError("source-local arsenic mask must match arsenic shape")
        _plot_control_arsenic(axes[1], hours, arsenic, source_local,
                              trajectory.node_ids, colors_as)
    axes[2].plot(hours, trajectory.injection, color="#1769aa", lw=2.0, label="Injection")
    action_axis = axes[2].twinx()
    action_axis.plot(hours, trajectory.actions, color="#d1495b", lw=1.0, alpha=0.55, label="Raw action")
    step_hours = float(np.median(np.diff(hours))) if len(hours) > 1 else 0.5
    for axis in axes:
        for start, end in _active_windows(trajectory.event_active):
            end_hour = hours[end] if end < len(hours) else hours[-1] + step_hours
            axis.axvspan(hours[start], end_hour, color="#ef476f", alpha=0.12)
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel("Chlorine concentration")
    axes[1].set_ylabel("AsIII concentration")
    axes[2].set_ylabel("Injection rate")
    action_axis.set_ylabel("Raw action")
    axes[2].set_xlabel("Simulation time (hours)")
    axes[0].set_title(title, fontsize=16, pad=12)
    for axis in axes[:2]:
        if len(axis.lines) <= 32:
            axis.legend(ncol=8, fontsize=7, loc="upper right")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = action_axis.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="upper right")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _set_chlorine_axis_limits(axis) -> None:
    """Start chlorine plots at zero and show at least the 0--1 mg/L range."""
    _, automatic_upper_limit = axis.get_ylim()
    axis.set_ylim(0.0, max(1.0, automatic_upper_limit))


def _plot_control_arsenic(axis, hours: np.ndarray, arsenic: np.ndarray,
                          source_local: np.ndarray, node_ids: tuple[str, ...],
                          colors: np.ndarray) -> None:
    for index in range(arsenic.shape[1]):
        label = node_ids[index] if index < len(node_ids) else str(index)
        local = source_local[:, index]
        axis.plot(hours, np.ma.masked_where(local, arsenic[:, index]),
                  color=colors[index], lw=1.2, alpha=0.85,
                  linestyle="solid", label=f"{label} transported")
        if np.any(local):
            axis.plot(hours, np.ma.masked_where(~local, arsenic[:, index]),
                      color=colors[index], lw=1.2, alpha=0.85,
                      linestyle="dotted", label=f"{label} source-local")


def _active_windows(active: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    values = np.asarray(active, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return tuple(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def render_forecast_targets(root: Path, *, actual: dict[str, np.ndarray],
                            predicted: dict[str, np.ndarray], node_ids: tuple[str, ...],
                            time_seconds: np.ndarray, spec: ForecastPlotsSpec,
                            title: str = "Forecast") -> tuple[Path, ...]:
    """Render arbitrary named node targets according to ForecastPlotsSpec."""
    if set(actual) != set(predicted): raise ValueError("actual and predicted plot targets differ")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = tuple(actual)
    selected = _forecast_plot_nodes(actual, node_ids, spec.node_selection)
    groups = ((target,) for target in targets) if spec.separate_target_files else (targets,)
    paths = []
    for group in groups:
        group = tuple(group); path = Path(root) / ((group[0] if len(group) == 1 else "targets") + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(len(group), 1, figsize=(14, 5 * len(group)), squeeze=False, constrained_layout=True)
        hours = np.asarray(time_seconds, dtype=float) / 3600
        for axis, target in zip(axes[:, 0], group, strict=True):
            observed, forecast = np.asarray(actual[target]), np.asarray(predicted[target])
            if observed.shape != forecast.shape or observed.ndim != 2 or observed.shape[1] != len(node_ids):
                raise ValueError("forecast plot arrays must share (time, node) shape")
            for index in selected:
                axis.plot(hours, observed[:, index], linestyle=spec.actual_line_style,
                          label=f"{node_ids[index]} actual")
                axis.plot(hours, forecast[:, index], linestyle=spec.rolling_prediction_line_style,
                          label=f"{node_ids[index]} prediction")
            axis.set_title(f"{title} · {target}"); axis.grid(True, alpha=.2); axis.legend()
        figure.savefig(path, dpi=170, facecolor="white"); plt.close(figure); paths.append(path)
    return tuple(paths)


def render_scenario_forecast_targets(root: Path, *, actual: dict[str, np.ndarray],
                                     predicted: dict[str, np.ndarray],
                                     node_ids: tuple[str, ...],
                                     forecast_time_seconds: np.ndarray,
                                     spec: ForecastPlotsSpec,
                                     title: str = "Forecast") -> tuple[Path, ...]:
    """Render rolling and random-anchor forecasts for one scenario trajectory.

    Arrays use ``(window, horizon, node)`` order. The rolling plot aligns each
    configured lead with its actual timestamp; the random-anchor plot joins
    preceding one-step predictions to one complete forecast horizon.
    """
    if set(actual) != set(predicted): raise ValueError("actual and predicted plot targets differ")
    times=np.asarray(forecast_time_seconds,dtype=float)
    if times.ndim!=2:raise ValueError("forecast times must have (window, horizon) shape")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared={}
    for target in actual:
        observed=np.asarray(actual[target]);forecast=np.asarray(predicted[target])
        if (observed.shape!=forecast.shape or observed.ndim!=3 or
                observed.shape[:2]!=times.shape or observed.shape[2]!=len(node_ids)):
            raise ValueError("scenario forecast arrays must share (window, horizon, node) shape")
        prepared[target]=(observed,forecast)
    paths=[];root=Path(root);root.mkdir(parents=True,exist_ok=True)
    groups=tuple((target,) for target in actual) if spec.separate_target_files else (tuple(actual),)
    actual_times=np.concatenate((times[:,0],times[-1,1:]))/3600.0
    anchor_candidates=np.arange(len(times))
    if spec.random_horizon_cutoff_day_range is not None:
        start_day,end_day=spec.random_horizon_cutoff_day_range
        cutoff_days=times[:,0]/86400.0
        anchor_candidates=anchor_candidates[
            (cutoff_days>=start_day)&(cutoff_days<=end_day)
        ]
        if not len(anchor_candidates):
            raise ValueError(
                "trajectory has no forecast cutoff in configured random horizon day range"
            )
    anchor=int(np.random.default_rng(spec.random_seed).choice(anchor_candidates))
    for group in groups:
        group_actual={
            target:prepared[target][0].reshape(-1,prepared[target][0].shape[-1])
            for target in group
        }
        selected_nodes=_forecast_plot_nodes(group_actual,node_ids,spec.node_selection)
        node_groups=(tuple((index,) for index in selected_nodes)
                     if spec.separate_node_files else (selected_nodes,))
        for plot_nodes in node_groups:
            for plot_kind in ("rolling","random-horizon"):
                if plot_kind=="random-horizon" and not spec.include_random_horizon_plot:continue
                stem=group[0] if len(group)==1 else "targets"
                if spec.separate_node_files:
                    stem=f"{stem}-{_safe_path_component(node_ids[plot_nodes[0]])}"
                path=root/f"{stem}-{plot_kind}.png"
                figure,axes=plt.subplots(len(group),1,figsize=(15,6*len(group)),squeeze=False,
                                         constrained_layout=True)
                for axis,target in zip(axes[:,0],group,strict=True):
                    observed,forecast=prepared[target]
                    if plot_kind=="rolling":
                        for index in plot_nodes:
                            actual_values=np.concatenate((observed[:,0,index],observed[-1,1:,index]))
                            axis.plot(actual_times,actual_values,linestyle=spec.actual_line_style,
                                      label=f"{node_ids[index]} actual")
                            leads=min(spec.rolling_prediction_steps,forecast.shape[1])
                            for lead in range(leads):
                                suffix=f" (+{lead+1})" if leads>1 else ""
                                axis.plot(times[:,lead]/3600.0,forecast[:,lead,index],
                                          linestyle=spec.rolling_prediction_line_style,
                                          label=f"{node_ids[index]} prediction{suffix}")
                    else:
                        if spec.random_horizon_include_history:
                            joined_times=np.concatenate((times[:anchor,0],times[anchor]))/3600.0
                        else:
                            joined_times=times[anchor]/3600.0
                        for index in plot_nodes:
                            if spec.random_horizon_include_history:
                                joined_actual=np.concatenate(
                                    (observed[:anchor,0,index],observed[anchor,:,index])
                                )
                            else:
                                joined_actual=observed[anchor,:,index]
                            axis.plot(joined_times,joined_actual,linestyle=spec.actual_line_style,
                                      label=f"{node_ids[index]} actual")
                            if (spec.random_horizon_include_history and
                                    spec.random_horizon_include_past_predictions and anchor):
                                axis.plot(times[:anchor,0]/3600.0,forecast[:anchor,0,index],
                                          linestyle=spec.rolling_prediction_line_style,
                                          label=f"{node_ids[index]} past predictions")
                            axis.plot(times[anchor]/3600.0,forecast[anchor,:,index],
                                      linestyle=spec.random_horizon_line_style,
                                      label=f"{node_ids[index]} full horizon")
                        axis.axvline(times[anchor,0]/3600.0,color="black",alpha=.25,lw=1)
                    axis.set_xlabel("Simulation time (hours)");axis.set_ylabel(target)
                    detail=(plot_kind if plot_kind=="rolling" else
                            f"{plot_kind} · cutoff day {times[anchor,0]/86400.0:.2f}")
                    axis.set_title(f"{title} · {target} · {detail}")
                    axis.grid(True,alpha=.2);axis.legend()
                figure.savefig(path,dpi=170,facecolor="white");plt.close(figure);paths.append(path)
    return tuple(paths)


def _safe_path_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def render_forecast_training_history(path: Path, *, history: list[dict],
                                     checkpoint_metric: str) -> Path:
    """Render optimization loss and the validation checkpoint metric by epoch."""
    if not history:
        raise ValueError("forecast training history must not be empty")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train"]["combined_loss"]) for item in history]
    validation_loss = [float(item["validation"]["combined_loss"]) for item in history]
    checkpoint_values = [
        _history_checkpoint_value(item["validation"], checkpoint_metric)
        for item in history
    ]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    figure, loss_axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    loss_axis.plot(epochs, train_loss, label="Training loss", color="#1769aa")
    loss_axis.plot(epochs, validation_loss, label="Validation loss", color="#d1495b")
    loss_axis.set_xlabel("Epoch"); loss_axis.set_ylabel("Normalized objective loss")
    loss_axis.grid(True, alpha=.2)
    metric_axis = loss_axis.twinx()
    metric_axis.plot(epochs, checkpoint_values, label=checkpoint_metric,
                     color="#2a9d8f", linestyle="dashed")
    metric_axis.set_ylabel(checkpoint_metric)
    lines, labels = loss_axis.get_legend_handles_labels()
    metric_lines, metric_labels = metric_axis.get_legend_handles_labels()
    loss_axis.legend(lines + metric_lines, labels + metric_labels, loc="best")
    loss_axis.set_title("Forecast training history")
    figure.savefig(path, dpi=170, facecolor="white"); plt.close(figure)
    return path


def render_forecast_metric_breakdowns(root: Path, *, metrics: dict,
                                      breakdowns: tuple[str, ...]) -> tuple[Path, ...]:
    """Render node and forecast-horizon metric summaries for each target."""
    if not breakdowns:
        return ()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root); root.mkdir(parents=True, exist_ok=True); paths = []
    for target, target_metrics in metrics.items():
        for breakdown in breakdowns:
            values = target_metrics.get(breakdown)
            if not values:
                raise ValueError(f"forecast metrics do not include {breakdown!r} breakdown")
            labels = tuple(next(iter(values.values())))
            if any(tuple(metric_values) != labels for metric_values in values.values()):
                raise ValueError("forecast metric breakdown labels differ between metrics")
            x = np.arange(len(labels))
            figure, axis = plt.subplots(figsize=(max(13, len(labels) * .42), 6),
                                         constrained_layout=True)
            if breakdown == "horizon":
                for metric, metric_values in values.items():
                    axis.plot(x + 1, [metric_values[label] for label in labels], marker="o",
                              label=metric.upper())
                axis.set_xlabel("Forecast step")
                axis.set_xticks(x + 1)
            elif breakdown == "node":
                width = .8 / len(values)
                for index, (metric, metric_values) in enumerate(values.items()):
                    offset = (index - (len(values) - 1) / 2) * width
                    axis.bar(x + offset, [metric_values[label] for label in labels],
                             width=width, label=metric.upper())
                axis.set_xlabel("Node")
                axis.set_xticks(x, labels, rotation=60, ha="right")
            else:
                raise ValueError(f"unsupported forecast metric plot breakdown: {breakdown}")
            axis.set_ylabel("Error in physical units")
            axis.set_title(f"{target} error by {breakdown}")
            axis.grid(True, axis="y", alpha=.2); axis.legend()
            path = root / f"{target}-{breakdown}-metrics.png"
            figure.savefig(path, dpi=170, facecolor="white"); plt.close(figure); paths.append(path)
    return tuple(paths)


def _history_checkpoint_value(validation: dict, checkpoint_metric: str) -> float:
    if checkpoint_metric in validation:
        return float(validation[checkpoint_metric])
    if "." in checkpoint_metric:
        target, metric = checkpoint_metric.split(".", 1)
        return float(validation["metrics"][target]["overall"][metric])
    raise ValueError(f"checkpoint metric {checkpoint_metric!r} is absent from training history")


def _forecast_plot_nodes(actual, node_ids, selection):
    if isinstance(selection, tuple):
        unknown = set(selection) - set(node_ids)
        if unknown: raise ValueError(f"forecast plot references unknown nodes: {sorted(unknown)}")
        return tuple(node_ids.index(value) for value in selection)
    if selection == "all": return tuple(range(len(node_ids)))
    if selection == "largest-actual-variation":
        combined = np.concatenate(tuple(np.asarray(value) for value in actual.values()), axis=0)
        return (int(np.argmax(np.ptp(combined, axis=0))),)
    raise ValueError(f"unsupported forecast node selection: {selection}")
