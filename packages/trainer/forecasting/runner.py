"""Encoder-decoder LSTM forecasting runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import resolve_dataset
from ..execution import RunContext, RunResult
from ..progress import tqdm
from ..serialization import canonical_data, config_hash
from ..simulation import Trajectory
from ..specs import (ArsenicScheduleStreamSpec,
                     AuxiliaryNextLatentLSTMEncoderSpec,
                     AuxiliaryNextLatentObjectiveSpec, DistillationObjectiveSpec,
                     ForecastEvaluationSpec, ForecastTrainingSpec,
                     LatentDynamicsForecastSpec, LatentDynamicsObjectiveSpec,
                     NextLatentLSTMEncoderSpec, NextLatentObjectiveSpec,
                     TrainingSpec)
from .data import (ForecastWindowDataset, NamedNormalization, TrajectoryBatchSampler,
                   build_named_windows, fit_streaming_normalization,
                   normalize_named_windows)
from .features import DERIVED_ARSENIC_VERSION
from .bundle import load_bundle_manifest
from .materialization import ForecastTrajectoryResolver
from .model import (EncoderDecoderLSTM, LatentDynamicsForecaster,
                    NextLatentLSTMEncoder)
from .objectives import (auxiliary_next_latent_objective,
                         distillation_objective, latent_dynamics_objective,
                         named_objective, next_latent_objective)
from .targets import reconstruct_physical
from ..evaluation.forecasting import ForecastMetricAccumulator
from ..evaluation.plots import (render_forecast_metric_breakdowns,
                                render_forecast_training_history,
                                render_scenario_forecast_targets)
from ..simulation import prepare_hydraulic_outflow_cache


FORECAST_CHECKPOINT_SCHEMA_VERSION = 3
_FORECAST_CHECKPOINT_FIELDS = {
    "model", "optimizer", "normalization", "rng", "history", "best_model",
    "best_loss", "best_epoch", "stale", "epoch",
}


class ForecastRunner:
    def __init__(self, backend=None, *, render_plots=True):
        self.backend = backend
        self.render_plots = render_plots

    def validate(self, job: TrainingSpec) -> None:
        if not isinstance(job, ForecastTrainingSpec):
            raise TypeError("ForecastRunner requires ForecastTrainingSpec")
        from ..data import validate_dataset_source
        validate_dataset_source(job.dataset)
        if job.trajectory_data.schedule_pairing == "cartesian" and any(
                isinstance(split.arsenic, ArsenicScheduleStreamSpec)
                for split in (job.dataset.train, job.dataset.validation, job.dataset.test)):
            raise ValueError("cartesian materialization with an unbounded schedule source is not a finite training dataset")

    def has_checkpoint(self, output_dir: Path, job: TrainingSpec) -> bool:
        return any(_epoch_archive_paths(output_dir / "checkpoints"))

    def can_resume(self, output_dir: Path, job: TrainingSpec) -> bool:
        try:
            return _load_newest_forecast_checkpoint(
                output_dir / "checkpoints", config_hash(job), "cpu"
            ) is not None
        except Exception:
            return False

    def run(self, context: RunContext) -> RunResult:
        job = context.job
        if not isinstance(job, ForecastTrainingSpec): raise TypeError("invalid forecast job")
        return self._run_forecast(context)

    def resume_completed(self, context: RunContext) -> None:
        """Regenerate runtime-only plots from a completed forecast bundle."""
        if not self.render_plots:
            return
        job = context.job
        if not isinstance(job, ForecastTrainingSpec):
            raise TypeError("ForecastRunner requires ForecastTrainingSpec")
        plot_spec = job.evaluation.effective_plots
        if plot_spec is None:
            return

        _seed(job.training_seed)
        device = _device(job.loop.device)
        dataset = resolve_dataset(job.dataset)
        hydraulic_catalog = prepare_hydraulic_outflow_cache(
            dataset,
            _session_backend(self.backend),
            workers=context.workers,
            description=f"{job.name} hydraulics",
        )
        backend = self.backend
        if backend is None:
            if job.trajectory_data.materialize_missing:
                from ..simulation.epyt import EpytFlowBackend
                backend = _SessionForecastBackend(EpytFlowBackend(), dataset)
            else:
                backend = lambda *_args, **_kwargs: None
        resolver = ForecastTrajectoryResolver(
            dataset, job.trajectory_data, backend, hydraulic_catalog
        )
        bundle = context.output_dir / "best"
        manifest = load_bundle_manifest(bundle)
        model = _model_from_bundle(bundle, manifest, device)
        normalization = _normalization_from_bundle(bundle)

        split_names = {
            "train": "train",
            "validation": "evaluation",
            "test": "test",
        }
        for plot_split in plot_spec.splits:
            split = split_names[plot_split]
            references = tuple(
                resolver.references(
                    split,
                    workers=context.workers,
                    description=f"{job.name} {split} plot trajectories",
                )
            )
            _render_scenario_plots(
                model=model,
                references=references,
                job=job,
                normalization=normalization,
                device=device,
                root=context.output_dir / "plots" / plot_split,
                split_name=plot_split,
            )

        metrics = json.loads(
            (context.output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        plot_root = context.output_dir / "plots"
        if getattr(plot_spec, "include_training_history_plot", False):
            render_forecast_training_history(
                plot_root / "training-history.png",
                history=metrics["history"],
                checkpoint_metric=job.evaluation.checkpoint_metric,
            )
        for plot_split in plot_spec.splits:
            split = split_names[plot_split]
            render_forecast_metric_breakdowns(
                plot_root / plot_split,
                metrics=metrics["final"][split]["metrics"],
                breakdowns=getattr(plot_spec, "metric_breakdown_plots", ()),
            )
        print(f"{job.name}: regenerated plots from completed best bundle")

    def _run_forecast(self, context: RunContext) -> RunResult:
        job = context.job
        assert isinstance(job, ForecastTrainingSpec)
        _seed(job.training_seed); device = _device(job.loop.device); dataset = resolve_dataset(job.dataset)
        hydraulic_catalog = prepare_hydraulic_outflow_cache(
            dataset,
            _session_backend(self.backend),
            workers=context.workers,
            description=f"{job.name} hydraulics",
        )
        backend = self.backend
        if backend is None:
            if job.trajectory_data.materialize_missing:
                from ..simulation.epyt import EpytFlowBackend
                backend = _SessionForecastBackend(EpytFlowBackend(), dataset)
            else:
                backend = lambda *_args, **_kwargs: None
        resolver = ForecastTrajectoryResolver(
            dataset, job.trajectory_data, backend, hydraulic_catalog
        )
        references = {split: tuple(resolver.references(split, workers=context.workers,
                      description=f"{job.name} {split} trajectories"))
                      for split in ("train", "evaluation", "test")}
        teacher = _load_teacher(context, job, device) if isinstance(job.objective, DistillationObjectiveSpec) else None
        checkpoint_dir = context.output_dir / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epoch_dir = checkpoint_dir / "epochs"; epoch_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best.pt"; latest = checkpoint_dir / "latest.pt"
        resume_checkpoint = None
        if context.resume:
            loaded = _load_newest_forecast_checkpoint(
                checkpoint_dir, config_hash(job), device
            )
            if loaded is None:
                raise RuntimeError("no complete compatible forecast epoch archive exists")
            resume_checkpoint, resume_archive = loaded
            _repair_forecast_checkpoint_aliases(
                resume_checkpoint, resume_archive, latest, best_path
            )
            normalization_raw = resume_checkpoint["normalization"]
            normalization = _normalization_from_raw(normalization_raw)
        else:
            unnormalized = ForecastWindowDataset(references["train"], job.observation, job.model.heads,
                job.model.history_steps, job.model.horizon_steps,
                description=f"{job.name} index train windows")
            independent = getattr(job.objective, "normalize_targets_independently", True)
            normalization = fit_streaming_normalization(
                unnormalized, independently=independent,
                input_mode=job.normalization.inputs,
                description=f"{job.name} fit normalization")
            normalization_raw = _normalization_raw(normalization)
        if teacher is not None: teacher = replace(teacher, student_normalization=normalization)
        datasets = {split: ForecastWindowDataset(items, job.observation, job.model.heads,
                    job.model.history_steps, job.model.horizon_steps, normalization,
                    include_future_histories=isinstance(
                        job.model,
                        LatentDynamicsForecastSpec | NextLatentLSTMEncoderSpec
                        | AuxiliaryNextLatentLSTMEncoderSpec,
                    ),
                    auxiliary_inputs=teacher.inputs if teacher else None,
                    auxiliary_normalization=teacher.normalization if teacher else None,
                    description=f"{job.name} index {split} windows")
                    for split, items in references.items()}
        _compatible_datasets(datasets)
        print(f"{job.name}: ready on {device}; " + ", ".join(
            f"{split}={len(values):,} windows from {len(values.sources):,} trajectories"
            for split, values in datasets.items()))
        sample_contract = datasets["train"].feature_contract
        if isinstance(
            job.model,
            NextLatentLSTMEncoderSpec | AuxiliaryNextLatentLSTMEncoderSpec,
        ):
            model_kwargs = {
                "input_size": datasets["train"].history_size,
                "latent_size": job.model.latent_size,
                "predictor_hidden_size": job.model.predictor_hidden_size,
                "num_layers": job.model.num_layers,
                "dropout": job.model.dropout,
                "ema_decay": job.model.ema_decay,
            }
            if isinstance(job.model, AuxiliaryNextLatentLSTMEncoderSpec):
                model_kwargs.update(
                    decoder_input_size=datasets["train"].future_known_size,
                    output_size=datasets["train"].head_sizes["chlorine"],
                    horizon_steps=job.model.horizon_steps,
                )
            model = NextLatentLSTMEncoder(**model_kwargs).to(device)
        elif isinstance(job.model, LatentDynamicsForecastSpec):
            model_kwargs = {
                "input_size": datasets["train"].history_size,
                "transition_input_size": datasets["train"].future_known_size,
                "hidden_size": job.model.hidden_size,
                "transition_hidden_size": job.model.transition_hidden_size,
                "output_size": datasets["train"].head_sizes["chlorine"],
                "horizon_steps": job.model.horizon_steps,
            }
            model = LatentDynamicsForecaster(**model_kwargs).to(device)
        else:
            model_kwargs = {"input_size": datasets["train"].history_size,
                            "decoder_input_size": datasets["train"].future_known_size,
                            "hidden_size": job.model.hidden_size, "num_layers": job.model.num_layers,
                            "dropout": job.model.dropout, "horizon_steps": job.model.horizon_steps,
                            "head_sizes": tuple((head.name, datasets["train"].head_sizes[head.name]) for head in job.model.heads),
                            "decoder_mode": job.model.decoder_mode}
            model = EncoderDecoderLSTM(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=job.optimizer.learning_rate,
                                      weight_decay=job.optimizer.weight_decay)
        train_sampler = TrajectoryBatchSampler(datasets["train"], job.loop.batch_size, job.training_seed)
        loader_kwargs = {"num_workers": job.loop.num_workers}
        train_loader = DataLoader(datasets["train"], batch_sampler=train_sampler, **loader_kwargs)
        evaluation_loader = DataLoader(datasets["evaluation"], batch_size=job.loop.batch_size,
                                       shuffle=False, **loader_kwargs)
        best, best_epoch, stale, history, best_state, start_epoch = math.inf, 0, 0, [], None, 1
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model"])
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            best = float(resume_checkpoint["best_loss"])
            best_epoch = int(resume_checkpoint["best_epoch"])
            stale = int(resume_checkpoint["stale"])
            history = list(resume_checkpoint["history"])
            best_state = resume_checkpoint["best_model"]
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            _restore_rng(resume_checkpoint["rng"])
        _json(checkpoint_dir / "normalization.json", normalization_raw)
        epochs = range(start_epoch, job.loop.epochs + 1)
        if job.loop.early_stopping.patience and stale >= job.loop.early_stopping.patience:
            epochs = range(0)
        for epoch in epochs:
            train_sampler.epoch = epoch
            training_started = time.perf_counter()
            train = _epoch(model, train_loader, job, normalization, device, optimizer, teacher,
                           None, f"{job.name} epoch {epoch}/{job.loop.epochs} train")
            training_elapsed = time.perf_counter() - training_started
            checkpoint_spec = None
            key = job.evaluation.checkpoint_metric
            if "." in key and isinstance(job.evaluation, ForecastEvaluationSpec):
                target, metric = key.split(".", 1)
                checkpoint_spec = replace(job.evaluation, targets=(target,), metrics=(metric,), breakdowns=("overall",), plots=None)
            validation_started = time.perf_counter()
            validation = _epoch(model, evaluation_loader, job, normalization, device, None, teacher,
                                checkpoint_spec, f"{job.name} epoch {epoch}/{job.loop.epochs} evaluation")
            validation_elapsed = time.perf_counter() - validation_started
            if key in validation: current = float(validation[key])
            elif checkpoint_spec is not None:
                current = float(validation["metrics"][target]["overall"][metric])
            else:
                raise ValueError(f"checkpoint metric {key!r} was not produced by validation")
            improved = current < best - job.loop.early_stopping.min_delta
            if improved:
                best, best_epoch, stale = current, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if best_state is None:
                raise RuntimeError("forecast training produced no best model state")
            print(f"{job.name}: epoch {epoch}/{job.loop.epochs} complete; "
                  f"train loss={train['combined_loss']:.6g}, {key}={current:.6g}, "
                  f"best={best:.6g} at epoch {best_epoch}")
            history.append({"epoch": epoch, "train": train, "validation": validation,
                            "training_elapsed_seconds": training_elapsed,
                            "validation_elapsed_seconds": validation_elapsed,
                            "elapsed_seconds": training_elapsed + validation_elapsed})
            checkpoint = {"schema_version": FORECAST_CHECKPOINT_SCHEMA_VERSION,
                          "job_hash": config_hash(job), "epoch": epoch,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "best_model": best_state,
                          "best_loss": best, "best_epoch": best_epoch, "stale": stale,
                          "history": history, "rng": _rng_state(),
                          "normalization": normalization_raw}
            epoch_path = epoch_dir / f"{epoch:06d}.pt"
            if epoch_path.exists():
                raise RuntimeError(f"refusing to overwrite forecast epoch archive: {epoch_path}")
            _publish_forecast_checkpoint(
                checkpoint, epoch_path, latest, best_path, publish_best=improved
            )
            if job.loop.early_stopping.patience and stale >= job.loop.early_stopping.patience: break
        if best_state is None:
            raise RuntimeError("forecast checkpoint has no best model state")
        model.load_state_dict(best_state)
        metrics_path = context.output_dir / "metrics.json"
        cached_metrics = None
        if context.resume and metrics_path.is_file():
            cached_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final = dict(cached_metrics.get("final", {})) if isinstance(cached_metrics, dict) else {}
        cached_runtime = cached_metrics.get("runtime", {}) if isinstance(cached_metrics, dict) else {}
        final_runtime = dict(cached_runtime.get("final_evaluation_seconds", {})) if isinstance(cached_runtime, dict) else {}
        runtime = {
            "device": str(device),
            "execution_workers": int(context.workers),
            "data_loader_workers": int(job.loop.num_workers),
            "training_seconds": sum(float(item.get("training_elapsed_seconds", 0.0)) for item in history),
            "checkpoint_validation_seconds": sum(float(item.get("validation_elapsed_seconds", 0.0)) for item in history),
            "final_evaluation_seconds": final_runtime,
        }

        def write_metrics():
            _json(metrics_path, {"best_epoch": best_epoch,
                "training_seed": job.training_seed,
                "fold_id": int(job.name.rsplit("-fold-", 1)[1]) if "-fold-" in job.name and job.name.rsplit("-fold-", 1)[1].isdigit() else None,
                "scenario_weighting": getattr(job.evaluation, "scenario_weighting", None),
                "best_validation_metric": best, "history": history, "final": final,
                "runtime": runtime})

        for split, values in datasets.items():
            if split in final:
                continue
            loader = DataLoader(values, batch_size=job.loop.batch_size, shuffle=False, **loader_kwargs)
            evaluation_started = time.perf_counter()
            final[split] = _epoch(model, loader, job, normalization, device, None, teacher,
                                  job.evaluation, f"{job.name} {split}")
            final_runtime[split] = time.perf_counter() - evaluation_started
            write_metrics()
            plot_split = "validation" if split == "evaluation" else split
            if (self.render_plots and job.evaluation.effective_plots is not None and
                    plot_split in job.evaluation.effective_plots.splits):
                _render_scenario_plots(
                    model=model, references=references[split], job=job,
                    normalization=normalization, device=device,
                    root=context.output_dir / "plots" / plot_split,
                    split_name=plot_split,
                )
        plot_spec = job.evaluation.effective_plots
        if self.render_plots and plot_spec is not None:
            plot_root = context.output_dir / "plots"
            if getattr(plot_spec, "include_training_history_plot", False):
                render_forecast_training_history(
                    plot_root / "training-history.png",
                    history=history,
                    checkpoint_metric=job.evaluation.checkpoint_metric,
                )
            for split, result in final.items():
                plot_split = "validation" if split == "evaluation" else split
                if plot_split in plot_spec.splits:
                    render_forecast_metric_breakdowns(
                        plot_root / plot_split,
                        metrics=result["metrics"],
                        breakdowns=getattr(plot_spec, "metric_breakdown_plots", ()),
                    )
        write_metrics()
        bundle = context.output_dir / "best"; bundle.mkdir(exist_ok=True); torch.save(model.state_dict(), bundle / "model.pt")
        _json(bundle / "normalization.json", normalization_raw)
        manifest_kind = (
            "auxiliary-next-latent-lstm-encoder"
            if isinstance(job.model, AuxiliaryNextLatentLSTMEncoderSpec)
            else "next-latent-lstm-encoder"
            if isinstance(job.model, NextLatentLSTMEncoderSpec)
            else "latent-dynamics-forecast"
            if isinstance(job.model, LatentDynamicsForecastSpec)
            else "encoder-decoder-lstm"
        )
        manifest = {"kind": manifest_kind,
            "job_hash": config_hash(job), "job": canonical_data(job), "model_kwargs": canonical_data(model_kwargs),
            "node_ids": list(datasets["train"].node_ids), "link_ids": list(datasets["train"].link_ids),
            "source_ids": list(datasets["train"].source_ids),
            "hydraulic_timestep_seconds": dataset.hydraulic_timestep_seconds,
            "calendar_requirement": "scenario-start-day", "history_feature_contract": canonical_data({
                "names": sample_contract["history_names"], "expansions": sample_contract["history_expansions"],
                "transforms": sample_contract["history_transforms"]}),
            "future_known_feature_contract": canonical_data({"names": sample_contract["future_known_names"],
                "expansions": sample_contract["future_known_expansions"], "transforms": sample_contract["future_known_transforms"]}),
            "heads": [canonical_data(head) for head in job.model.heads], "normalization": normalization_raw,
            "derived_feature_versions": {"transported-arsenic": DERIVED_ARSENIC_VERSION}}
        if isinstance(job.model, LatentDynamicsForecastSpec):
            manifest["latent_contract"] = {
                "name": "z_t",
                "size": job.model.hidden_size,
                "source": "encoder-top-hidden-state",
                "history_steps": job.model.history_steps,
            }
        elif isinstance(
            job.model,
            NextLatentLSTMEncoderSpec | AuxiliaryNextLatentLSTMEncoderSpec,
        ):
            manifest["latent_contract"] = {
                "name": "z_t",
                "size": job.model.latent_size,
                "source": "layer-normalized-encoder-top-hidden-state",
                "history_steps": job.model.history_steps,
            }
        _json(bundle / "manifest.json", manifest)
        return RunResult({"best": bundle, "latest": latest, "metrics": metrics_path},
                         {"device": str(device), "workers": context.workers,
                          "data_loader_workers": job.loop.num_workers,
                          "node_ids": list(datasets["train"].node_ids), "best_epoch": best_epoch})


def _epoch(model, loader, job, normalization, device, optimizer, teacher, metric_spec, description):
    model.train(optimizer is not None); totals = {"combined_loss": 0.0, "outcome_loss": 0.0}; samples = 0
    metrics = (ForecastMetricAccumulator(metric_spec, node_ids=loader.dataset.node_ids)
               if isinstance(metric_spec, ForecastEvaluationSpec) else None)
    progress = tqdm(loader, total=len(loader), desc=description, unit="batch", leave=False,
                    dynamic_ncols=True)
    for batch in progress:
        history, future = batch["history"].to(device), batch["future_known"].to(device)
        targets = {name: value.to(device) for name, value in batch["targets"].items()}
        if isinstance(job.objective, AuxiliaryNextLatentObjectiveSpec):
            future_histories = batch["future_histories"].to(device)
            current_states, predicted_states, predictions = (
                model.predict_next_and_forecast(history, future)
            )
            with torch.no_grad():
                target_states = model.encode_target(future_histories[:, 0])
            combined, components = auxiliary_next_latent_objective(
                current_states,
                predicted_states,
                target_states,
                predictions,
                targets,
                job.objective,
            )
        elif isinstance(job.objective, NextLatentObjectiveSpec):
            future_histories = batch["future_histories"].to(device)
            if future_histories.shape[1] != 1:
                raise ValueError("next-latent training requires exactly one shifted history")
            current_states, predicted_states = model.predict_next(history)
            with torch.no_grad():
                target_states = model.encode_target(future_histories[:, 0])
            combined, components = next_latent_objective(
                current_states,
                predicted_states,
                target_states,
                job.objective,
            )
            predictions = {}
        elif isinstance(job.objective, LatentDynamicsObjectiveSpec):
            predictions, predicted_states = model.rollout(history, future)
            future_histories = batch["future_histories"].to(device)
            batch_size, horizon, history_steps, feature_count = future_histories.shape
            with torch.no_grad():
                target_states = model.encode(future_histories.reshape(
                    batch_size * horizon, history_steps, feature_count)).reshape(
                        batch_size, horizon, -1)
            combined, components = latent_dynamics_objective(
                predictions, targets, predicted_states, target_states, job.objective)
        else:
            predictions = model(history, future)
        if isinstance(job.objective, DistillationObjectiveSpec):
            if teacher is None: raise RuntimeError("distillation teacher was not loaded")
            teacher_predictions = {job.objective.teacher_head: teacher.predict(batch, device)}
            combined, components = distillation_objective(predictions, targets, teacher_predictions, job.objective)
        elif not isinstance(
            job.objective,
            LatentDynamicsObjectiveSpec | NextLatentObjectiveSpec
            | AuxiliaryNextLatentObjectiveSpec,
        ):
            combined, components = named_objective(predictions, targets, job.objective)
        if optimizer is not None:
            optimizer.zero_grad(); combined.backward(); optimizer.step()
            if isinstance(
                job.objective,
                NextLatentObjectiveSpec | AuxiliaryNextLatentObjectiveSpec,
            ):
                model.update_target_encoder()
        samples += len(history)
        totals["combined_loss"] += float(combined.detach()) * len(history)
        totals["outcome_loss"] += float(components["outcome_loss"].detach()) * len(history)
        for key, value in components.items():
            if key in ("combined_loss", "outcome_loss"): continue
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(history)
        progress.set_postfix(loss=f"{totals['combined_loss'] / samples:.4g}")
        if metrics is not None:
            physical_prediction, physical_actual = _physical_batch(
                predictions, batch["targets"], batch["previous_levels"], job, normalization)
            indexes = batch["trajectory_index"].tolist()
            scenario_ids = tuple(loader.dataset.scenario_ids[index] for index in indexes)
            metrics.update(physical_prediction, physical_actual, scenario_ids=scenario_ids)
    result = {key: value / samples for key, value in totals.items()}
    if metrics is not None: result["metrics"] = metrics.finalize()
    return result


def _physical_batch(predictions, targets, previous_levels, job, normalization):
    physical_prediction, physical_actual = {}, {}
    for head in job.model.heads:
        mean, std = normalization.targets[head.name]
        predicted = predictions[head.name].detach().cpu().numpy() * std + mean
        actual = targets[head.name].detach().cpu().numpy() * std + mean
        previous = previous_levels[head.name].detach().cpu().numpy()
        physical_prediction[head.name] = reconstruct_physical(predicted, previous, head)
        physical_actual[head.name] = reconstruct_physical(actual, previous, head)
    return physical_prediction, physical_actual


def _model_from_bundle(bundle, manifest, device):
    model_kwargs = dict(manifest["model_kwargs"])
    kind = manifest["kind"]
    if kind == "encoder-decoder-lstm":
        model = EncoderDecoderLSTM(**model_kwargs)
    elif kind == "latent-dynamics-forecast":
        model = LatentDynamicsForecaster(**model_kwargs)
    elif kind in {
        "next-latent-lstm-encoder",
        "auxiliary-next-latent-lstm-encoder",
    }:
        model = NextLatentLSTMEncoder(**model_kwargs)
    else:
        raise ValueError(f"unsupported forecast bundle kind: {kind!r}")
    model.load_state_dict(
        torch.load(bundle / "model.pt", map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def _normalization_from_bundle(bundle):
    raw = json.loads((bundle / "normalization.json").read_text(encoding="utf-8"))
    return _normalization_from_raw(raw)


def _normalization_from_raw(raw):
    return NamedNormalization(
        np.asarray(raw["history_mean"], dtype=np.float32),
        np.asarray(raw["history_std"], dtype=np.float32),
        np.asarray(raw["future_mean"], dtype=np.float32),
        np.asarray(raw["future_std"], dtype=np.float32),
        {
            name: (
                np.asarray(values["mean"], dtype=np.float32),
                np.asarray(values["std"], dtype=np.float32),
            )
            for name, values in raw["targets"].items()
        },
    )


def _normalization_raw(normalization):
    return {
        "history_mean": normalization.history_mean.tolist(),
        "history_std": normalization.history_std.tolist(),
        "future_mean": normalization.future_mean.tolist(),
        "future_std": normalization.future_std.tolist(),
        "targets": {
            name: {"mean": values[0].tolist(), "std": values[1].tolist()}
            for name, values in normalization.targets.items()
        },
    }


def _render_scenario_plots(*,model,references,job,normalization,device,root,split_name):
    spec=job.evaluation.effective_plots
    if spec is None:return
    selected=[reference.load() for reference in _select_plot_references(references,spec)]
    for trajectory_index,trajectory in enumerate(selected):
        raw=build_named_windows([trajectory],job.observation,job.model.heads,
                                job.model.history_steps,job.model.horizon_steps)
        normalized=normalize_named_windows(raw,normalization)
        model.eval()
        with torch.no_grad():
            predictions=model(torch.from_numpy(normalized["history"]).to(device),
                              torch.from_numpy(normalized["future_known"]).to(device))
        predicted,actual=_physical_batch(predictions,
            {name:torch.from_numpy(value) for name,value in normalized["targets"].items()},
            {name:torch.from_numpy(value) for name,value in normalized["previous_levels"].items()},
            job,normalization)
        names=job.evaluation.targets
        count=len(raw.history);starts=range(job.model.history_steps,job.model.history_steps+count)
        forecast_times=np.stack([
            np.asarray(trajectory.time_seconds[start:start+job.model.horizon_steps],dtype=float)
            for start in starts
        ])
        suffix="" if spec.one_plot_per_scenario else f"-{trajectory_index:03d}"
        render_scenario_forecast_targets(
            Path(root)/f"{_safe_path_component(trajectory.scenario_id)}{suffix}",
            actual={name:actual[name] for name in names},
            predicted={name:predicted[name] for name in names},node_ids=trajectory.node_ids,
            forecast_time_seconds=forecast_times,
            spec=replace(spec,random_seed=spec.random_seed+int(hashlib.sha256(
                trajectory.scenario_id.encode()).hexdigest()[:8],16)),
            title=f"{job.name} · {split_name} · {trajectory.scenario_id}",
        )


def _select_plot_references(references,spec):
    if not spec.one_plot_per_scenario:return list(references)
    groups={}
    for reference in references:groups.setdefault(reference.scenario_id,[]).append(reference)
    if spec.scenario_trajectory_selection in ("first","first-fixed-schedule-pair"):
        return [items[0] for items in groups.values()]
    if spec.scenario_trajectory_selection=="random":
        rng=random.Random(spec.random_seed)
        return [rng.choice(items) for items in groups.values()]
    raise ValueError(f"unsupported forecast trajectory selection: {spec.scenario_trajectory_selection}")


def _safe_path_component(value):
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


@dataclass(frozen=True)
class _TeacherRuntime:
    model: EncoderDecoderLSTM
    inputs: object
    normalization: NamedNormalization
    source: object
    destination: object
    student_normalization: NamedNormalization | None = None

    def predict(self, batch, device):
        if self.student_normalization is None: raise RuntimeError("student normalization is unavailable")
        with torch.no_grad():
            prediction = self.model(batch["auxiliary_history"].to(device),
                                    batch["auxiliary_future_known"].to(device))[self.source.name]
        source_mean, source_std = self.normalization.targets[self.source.name]
        represented = prediction * torch.as_tensor(source_std, device=device) + torch.as_tensor(source_mean, device=device)
        physical = _inverse_transform_tensor(represented, self.source.transform, self.source.scale)
        destination = _apply_transform_tensor(physical, self.destination.transform, self.destination.scale)
        mean, std = self.student_normalization.targets[self.destination.name]
        return (destination - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)


def _load_teacher(context, job, device):
    from ..specs import ForecastFeatureSpec, ForecastHeadSpec, ForecastInputSpec
    bundle = context.resolve(job.objective.teacher); manifest = load_bundle_manifest(bundle)
    source_raw = next((item for item in manifest["heads"] if item["name"] == job.objective.teacher_head), None)
    destination = next((item for item in job.model.heads if item.name == job.objective.student_head), None)
    if source_raw is None or destination is None: raise ValueError("distillation head contract is unavailable")
    source = ForecastHeadSpec(**{key: value for key, value in source_raw.items() if key in ForecastHeadSpec.__dataclass_fields__})
    if source.target != destination.target or source.prediction != destination.prediction:
        raise ValueError("distillation heads require compatible target and prediction representations")
    teacher_model_spec = manifest["job"]["model"]
    if (int(teacher_model_spec["history_steps"]) != job.model.history_steps or
            int(teacher_model_spec["horizon_steps"]) != job.model.horizon_steps):
        raise ValueError("distillation teacher and student require matching history and horizon steps")
    observation_raw = manifest["job"]["observation"]
    def feature(raw):
        values = dict(raw); values["parameters"] = values.get("parameters", {})
        values["transform_parameters"] = values.get("transform_parameters", {})
        values["cycles"] = tuple(values.get("cycles", ()))
        return ForecastFeatureSpec(**values)
    teacher_input = ForecastInputSpec(tuple(feature(item) for item in observation_raw["history"]),
                                      tuple(feature(item) for item in observation_raw["future_known"]))
    norm = json.loads((bundle / "normalization.json").read_text())
    normalization = NamedNormalization(np.asarray(norm["history_mean"], dtype=np.float32),
        np.asarray(norm["history_std"], dtype=np.float32), np.asarray(norm["future_mean"], dtype=np.float32),
        np.asarray(norm["future_std"], dtype=np.float32), {name: (np.asarray(values["mean"], dtype=np.float32),
        np.asarray(values["std"], dtype=np.float32)) for name, values in norm["targets"].items()})
    model_kwargs = dict(manifest["model_kwargs"]); model_kwargs["head_sizes"] = tuple(tuple(item) for item in model_kwargs["head_sizes"])
    teacher_model = EncoderDecoderLSTM(**model_kwargs).to(device)
    teacher_model.load_state_dict(torch.load(bundle / "model.pt", map_location=device, weights_only=True))
    teacher_model.eval()
    for parameter in teacher_model.parameters(): parameter.requires_grad_(False)
    return _TeacherRuntime(teacher_model, teacher_input, normalization, source, destination)


def _inverse_transform_tensor(values, transform, scale):
    if transform == "identity": return values
    if transform == "log1p": return torch.expm1(values)
    if transform == "divide-by": return values * float(scale)
    raise ValueError(f"unsupported transform: {transform}")


def _apply_transform_tensor(values, transform, scale):
    if transform == "identity": return values
    if transform == "log1p": return torch.log1p(values)
    if transform == "divide-by": return values / float(scale)
    raise ValueError(f"unsupported transform: {transform}")


class _SessionForecastBackend:
    """Adapt the simulator session protocol to the materializer protocol."""
    def __init__(self, backend, dataset): self.backend, self.dataset = backend, dataset
    def simulate(self, scenario, arsenic_schedule, chlorine_schedule, **_kwargs):
        session = self.backend.create_session(str(scenario.path),
            hydraulic_timestep_seconds=self.dataset.hydraulic_timestep_seconds,
            calendar_start_day=scenario.calendar_start_day,
            chlorine_source_ids=scenario.chlorine_source_ids,
            monitored_node_ids=scenario.monitored_node_ids)
        try:
            values = chlorine_schedule.values
            if len(values) == 0:
                raise ValueError("chlorine schedule must contain at least one value")
            first = session.reset(arsenic_schedule.events)
            samples = [first]
            for index, injection in enumerate(values):
                sample, done = session.step(float(injection), 0.0); samples.append(sample)
                if done and index != len(values) - 1:
                    raise RuntimeError(
                        f"EPANET terminated {scenario.scenario_id} after "
                        f"{index + 1}/{len(values)} control intervals"
                    )
        finally: session.close()
        node_ids, link_ids = samples[0].node_ids, samples[0].link_ids
        zeros = lambda name: np.stack([np.asarray(getattr(sample, name)) for sample in samples])
        return Trajectory(self.dataset.network, scenario.scenario_id, scenario.calendar_start_day,
            chlorine_schedule.hash, arsenic_schedule.hash, -1, node_ids, link_ids, scenario.chlorine_source_ids,
            np.asarray([sample.time_seconds for sample in samples]), zeros("pressure"), zeros("demand"), zeros("flow"),
            zeros("chlorine"), zeros("arsenic"), np.asarray([sample.injection for sample in samples]),
            np.asarray([sample.action for sample in samples]), np.asarray([sample.event_active for sample in samples], dtype=bool),
            np.stack([np.asarray(sample.event_strength_by_node) for sample in samples]),
            hydraulic_outflow=np.stack([np.asarray(sample.hydraulic_outflow) for sample in samples]),
            simulator_backend="epanet", simulator_config_hash="", flow_units=session.flow_units)


def _session_backend(backend):
    if backend is None:
        return None
    if hasattr(backend, "create_session"):
        return backend
    candidate = getattr(backend, "backend", None)
    return candidate if hasattr(candidate, "create_session") else None


def _compatible_datasets(groups):
    first = next(iter(groups.values()))
    identity = (first.node_ids, first.link_ids, first.source_ids, first.feature_contract,
                first.history_size, first.future_known_size, first.head_sizes)
    if any((item.node_ids, item.link_ids, item.source_ids, item.feature_contract,
            item.history_size, item.future_known_size, item.head_sizes) != identity
           for item in groups.values()):
        raise ValueError("forecast splits have incompatible feature, target, or topology contracts")


def _epoch_archive_paths(checkpoint_dir):
    root = Path(checkpoint_dir) / "epochs"
    if not root.is_dir():
        return ()
    return tuple(sorted(
        (
            path for path in root.glob("*.pt")
            if len(path.stem) == 6 and path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
        reverse=True,
    ))


def _load_newest_forecast_checkpoint(checkpoint_dir, expected_job_hash, map_location):
    for path in _epoch_archive_paths(checkpoint_dir):
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            _validate_forecast_checkpoint(
                checkpoint, expected_job_hash, expected_epoch=int(path.stem)
            )
        except Exception:
            continue
        return checkpoint, path
    return None


def _validate_forecast_checkpoint(checkpoint, expected_job_hash, *, expected_epoch):
    if not isinstance(checkpoint, dict):
        raise ValueError("forecast checkpoint is not a mapping")
    if checkpoint.get("schema_version") != FORECAST_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported forecast checkpoint schema")
    if checkpoint.get("job_hash") != expected_job_hash:
        raise ValueError("forecast checkpoint job identity mismatch")
    if not _FORECAST_CHECKPOINT_FIELDS <= checkpoint.keys():
        raise ValueError("forecast checkpoint is incomplete")
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    best_epoch = checkpoint["best_epoch"]
    if not isinstance(epoch, int) or epoch != expected_epoch or epoch < 1:
        raise ValueError("forecast checkpoint epoch identity mismatch")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[-1], dict)
        or history[-1].get("epoch") != epoch
    ):
        raise ValueError("forecast checkpoint history is incomplete")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epoch:
        raise ValueError("forecast checkpoint best epoch is invalid")
    if not isinstance(checkpoint["stale"], int) or checkpoint["stale"] < 0:
        raise ValueError("forecast checkpoint early-stopping state is invalid")
    for name in ("model", "optimizer", "normalization", "rng", "best_model"):
        if not isinstance(checkpoint[name], dict):
            raise ValueError(f"forecast checkpoint {name} state is invalid")


def _publish_forecast_checkpoint(
    checkpoint, epoch_path, latest_path, best_path, *, publish_best
):
    _atomic_torch_save(checkpoint, epoch_path)
    if publish_best:
        _atomic_torch_save(checkpoint["best_model"], best_path)
    _atomic_copy(epoch_path, latest_path)


def _repair_forecast_checkpoint_aliases(
    checkpoint, epoch_path, latest_path, best_path
):
    _atomic_torch_save(checkpoint["best_model"], best_path)
    _atomic_copy(epoch_path, latest_path)


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        torch.save(value, temporary)
        # Windows implements fsync via _commit(), which rejects read-only file
        # descriptors with EBADF.  Open read/write even though no further data
        # is written so the completed checkpoint can be flushed durably before
        # it is published by the atomic replace.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _device(value):
    if value != "auto": return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
