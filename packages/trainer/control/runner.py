"""Recurrent-PPO runner for baseline and forecast-assisted control."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import signal
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)

from ..data import WARMUP_DAYS, resolve_dataset
from ..evaluation import control_metrics
from ..evaluation.protocol import (PROTOCOL_VERSION, ClearanceError, ReferenceEvaluation,
    clearance_time, reference_windows, physical_metrics, summarize_cases, scenario_average,
    scoring_windows, next_training_chunk, crossed_thresholds)
from ..forecasting.features import derive_transported_arsenic
from ..evaluation.plots import (ControlEvaluationTrajectory,
                                render_control_trajectory)
from ..execution import RunContext, RunResult
from ..observations import ControlObservationBuilder
from ..observations.control import base_control_observation_spec
from ..progress import has_shared_progress_lock, tqdm
from ..schedules import (ArsenicEvent, ArsenicSchedule, fixed_schedule,
                         spawn_schedule_sources, spawn_schedule_streams)
from ..serialization import canonical_data, config_hash
from ..simulation import prepare_hydraulic_outflow_cache
from ..simulation.epyt import EpytFlowBackend
from ..specs import (ArsenicScheduleStreamSpec, ArtifactRef,
                     ControlPlotsActionSpec, ControlTrainingSpec,
                     DynamicsAuxiliaryPPOSpec, FixedArsenicScheduleSetSpec,
                     FixedPCALatentAssistedControlObservationSpec,
                     ForecastAssistedControlObservationSpec,
                     LatentAssistedControlObservationSpec,
                     NoiseForecastProviderSpec,
                     ProjectedLatentAssistedControlObservationSpec,
                     SharedDynamicsEncoderControlObservationSpec,
                     TrainableNextLatentControlObservationSpec, TrainingSpec)
from .algorithms import DynamicsAuxiliaryPPO
from .environment import ControlEnv
from .features import (FixedPCALatentFeatureExtractor,
                       ProjectedLatentFeatureExtractor,
                       SharedDynamicsEncoderFeatureExtractor,
                       TrainableNextLatentFeatureExtractor,
                       UngatedProjectedLatentFeatureExtractor)
from .normalization import BaseOnlyVecNormalize
from .providers import (make_latent_history_provider, make_latent_provider,
                        make_provider, resolve_latent_bundle)
from .rewards import create_reward

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _PendingCheckpointEvaluation:
    process: object
    receive: object
    cancellation: object
    snapshot_dir: Path
    timesteps: int


class ControlRunner:
    def __init__(self, backend=None, *, render_plots=True,
                 defer_evaluation=False, on_training_complete=None,
                 evaluate_best_model=False, timestep_targets=None,
                 resume_legacy_training=False):
        if on_training_complete is not None and not defer_evaluation:
            raise ValueError("training completion callback requires deferred evaluation")
        if resume_legacy_training and not defer_evaluation:
            raise ValueError("legacy training continuation requires deferred evaluation")
        self.backend = backend or EpytFlowBackend()
        self.render_plots = render_plots
        self.defer_evaluation = defer_evaluation
        self.on_training_complete = on_training_complete
        self.evaluate_best_model = evaluate_best_model
        # Extend existing runs without changing their configuration hashes.
        self.timestep_targets = dict(timestep_targets or {})
        self.resume_legacy_training = resume_legacy_training
    def training_target(self, job: TrainingSpec) -> int:
        target = self.timestep_targets.get(job.name, job.total_timesteps)
        if isinstance(target, bool) or not isinstance(target, int) or target < job.total_timesteps:
            raise ValueError("training target must be an integer at least as large as the original budget")
        return target
    def needs_training(self, output_dir: Path, job: TrainingSpec) -> bool:
        if job.name not in self.timestep_targets:
            return False
        state = _read(output_dir / "checkpoints" / "latest.json")
        return (not isinstance(state, dict)
                or int(state.get("completed_timesteps", 0)) < self.training_target(job))
    def validate(self, job: TrainingSpec) -> None:
        if not isinstance(job, ControlTrainingSpec): raise TypeError("ControlRunner requires ControlTrainingSpec")
        self.training_target(job)
        from ..data import validate_dataset_source
        validate_dataset_source(job.dataset)
        assisted_observation = isinstance(
            job.observation,
            (
                ForecastAssistedControlObservationSpec,
                LatentAssistedControlObservationSpec,
                ProjectedLatentAssistedControlObservationSpec,
                FixedPCALatentAssistedControlObservationSpec,
                SharedDynamicsEncoderControlObservationSpec,
                TrainableNextLatentControlObservationSpec,
            ),
        )
        if assisted_observation:
            ControlObservationBuilder(
                base_control_observation_spec(job.observation),
                job.observation.node_ids,
                job.max_injection,
            )
        create_reward(job.reward)
        shared = isinstance(
            job.observation, SharedDynamicsEncoderControlObservationSpec
        )
        auxiliary = isinstance(job.algorithm, DynamicsAuxiliaryPPOSpec)
        if shared != auxiliary:
            raise ValueError(
                "shared dynamics observations and dynamics-auxiliary PPO must be configured together"
            )
        if shared and job.algorithm.normalize_observations:
            raise ValueError(
                "shared dynamics histories already use forecast normalization; PPO observation normalization must be disabled"
            )
        if (
            isinstance(job.observation, FixedPCALatentAssistedControlObservationSpec)
            and not job.algorithm.normalize_observations
        ):
            raise ValueError(
                "fixed PCA control requires base-observation normalization"
            )
        if job.algorithm.pretrained_policy is not None and not job.algorithm.normalize_observations:
            raise ValueError(
                "behavior-cloned PPO continuation requires VecNormalize observations"
            )
    def has_checkpoint(self, output_dir: Path, job: TrainingSpec) -> bool:
        return (output_dir/"checkpoints"/"latest.zip").is_file()
    def can_resume(self, output_dir: Path, job: TrainingSpec) -> bool:
        state=_read(output_dir/"checkpoints"/"latest.json")
        return isinstance(state,dict) and state.get("job_hash")==config_hash(job) and (output_dir/"checkpoints"/"latest.zip").is_file() and (output_dir/"checkpoints"/"vec_normalize.pkl").is_file()
    def run(self, context: RunContext) -> RunResult:
        job=context.job
        if not isinstance(job,ControlTrainingSpec):raise TypeError("invalid control job")
        total_timesteps = self.training_target(job)
        if job.name in self.timestep_targets and (not context.resume or not self.can_resume(context.output_dir, job)):
            raise RuntimeError("extended training requires a compatible existing checkpoint")
        legacy_evaluation = False
        if context.resume:
            saved=_read(context.output_dir/"checkpoints"/"latest.json")
            saved_version = saved.get("evaluation_protocol_version") if isinstance(saved,dict) else None
            if isinstance(saved,dict) and saved_version != PROTOCOL_VERSION:
                legacy_evaluation = (self.resume_legacy_training
                    and job.name in self.timestep_targets
                    and (saved_version is None or (type(saved_version) is int and saved_version < PROTOCOL_VERSION)))
            if not isinstance(saved,dict) or (saved_version != PROTOCOL_VERSION and not legacy_evaluation):
                raise ValueError("Cannot resume legacy evaluation history under the new protocol; use a separate run")
        _seed(job.training_seed); dataset=resolve_dataset(job.dataset); algo=job.algorithm
        if any(len(item.chlorine_source_ids)!=1 for item in dataset.scenarios):
            raise ValueError("trainer2 recurrent control requires exactly one chlorine source per scenario")
        hydraulic_catalog=prepare_hydraulic_outflow_cache(
            dataset,self.backend,workers=context.workers,
            splits=("train",) if self.defer_evaluation else ("train","evaluation","test"),
            description=f"{job.name} hydraulics")
        assisted = isinstance(job.observation,(
            ForecastAssistedControlObservationSpec,
            LatentAssistedControlObservationSpec,
            ProjectedLatentAssistedControlObservationSpec,
            FixedPCALatentAssistedControlObservationSpec,
            SharedDynamicsEncoderControlObservationSpec,
            TrainableNextLatentControlObservationSpec,
        ))
        node_ids = job.observation.node_ids if assisted else dataset.scenarios_for("train")[0].monitored_node_ids
        if any(item.monitored_node_ids != node_ids for item in dataset.scenarios):
            raise ValueError("all control scenarios must use the configured monitored node order")
        dependency_paths={ref: paths for ref,paths in context.artifacts.items()}
        def resolve(reference): return dependency_paths[reference.job][reference.artifact]
        train_schedule=dataset.schedule_for("train")
        train_scenarios=dataset.scenarios_for("train")
        streams=spawn_schedule_sources(
            train_schedule,algo.n_envs,hydraulic_catalog=hydraulic_catalog)
        # EPANET-MSX stores its active project in process-global state, so
        # multiple live training environments require one process each.
        isolated = isinstance(self.backend, EpytFlowBackend)
        subprocess_envs=isolated and algo.n_envs>1
        factories=[]
        for index,stream in enumerate(streams):
            def factory(index=index,stream=stream):
                if subprocess_envs:signal.signal(signal.SIGINT,signal.SIG_IGN)
                provider=_make_observation_provider(job.observation,node_ids,resolve,dataset.hydraulic_timestep_seconds)
                builder=ControlObservationBuilder(job.observation,node_ids,job.max_injection,provider)
                scenarios=_training_scenario_order(train_scenarios,job.training_seed,index)
                env=ControlEnv(scenarios,self.backend,stream,builder,create_reward(job.reward),dataset.hydraulic_timestep_seconds,
                    dataset.duration_seconds//dataset.hydraulic_timestep_seconds,job.max_injection,job.max_delta,job.initial_injection,
                    dataset.network,context.output_dir/"trajectories"/"train"/f"env-{index:03d}")
                return Monitor(env)
            factories.append(factory)
        # EPANET-MSX evaluations run in disposable spawned processes below, so
        # a single training environment can remain in-process.
        if subprocess_envs:
            previous_sigint=signal.signal(signal.SIGINT,signal.SIG_IGN)
            try:
                with _native_worker_thread_limits():raw=SubprocVecEnv(factories,start_method="spawn")
            finally:
                signal.signal(signal.SIGINT,previous_sigint)
        else:raw=DummyVecEnv(factories)
        checkpoints=context.output_dir/"checkpoints";checkpoints.mkdir(parents=True,exist_ok=True)
        training_complete_on_resume=False
        legacy_evaluation_cutoff=None
        evaluation_queue=[]
        runtime={"training_seconds":0.0,"checkpoint_evaluation_seconds":0.0,
                 "final_validation_seconds":0.0,"final_test_seconds":0.0}
        if context.resume:
            train_env=VecNormalize.load(str(checkpoints/"vec_normalize.pkl"),raw); train_env.training=True
            model=_algorithm_class(algo).load(str(checkpoints/"latest.zip"),env=train_env,device=algo.device)
            state=_read(checkpoints/"latest.json")
            saved_runtime=state.get("runtime")
            if isinstance(saved_runtime,dict):
                for key in runtime:
                    if isinstance(saved_runtime.get(key),int|float):
                        runtime[key]=float(saved_runtime[key])
            completed=int(state["completed_timesteps"]); best=float(state["best_reward"])
            history=list(state.get("history",()))
            evaluation_queue=list(state.get("pending_evaluations",()))
            if not evaluation_queue and state.get("pending_evaluation") is not None:
                evaluation_queue=[state["pending_evaluation"]]
            legacy_evaluation_cutoff=state.get("legacy_evaluation_cutoff_timesteps")
            if legacy_evaluation:
                # Discard incompatible evaluation results without copying old
                # data or repeating evaluation of the archived legacy models.
                legacy_evaluation_cutoff=completed
                history=[];best=-float("inf");evaluation_queue=[]
                print(f"Resuming {job.name} at {completed} timesteps; discarding legacy evaluation history "
                      "without copying old data.",flush=True)
            initialization=state.get("pretrained_policy")
            training_complete_on_resume=completed>=total_timesteps and not legacy_evaluation
            environment_states=list(state["environment_states"])
            for index,item in enumerate(environment_states): train_env.env_method("load_runtime_state",item,indices=index)
            _restore_rng(state["rng"])
        else:
            pretrained_bundle = _resolve_pretrained_policy_bundle(algo, resolve)
            if pretrained_bundle is not None:
                _validate_pretrained_policy_contract(
                    _read_pretrained_policy_manifest(pretrained_bundle), job
                )
            train_env = _training_vec_normalize(
                raw, algo, job.observation, pretrained_bundle=pretrained_bundle
            )
            policy_name = "MlpPolicy" if isinstance(algo, DynamicsAuxiliaryPPOSpec) else "MlpLstmPolicy"
            model=_algorithm_class(algo)(policy_name,train_env,learning_rate=algo.learning_rate,n_steps=algo.n_steps,batch_size=algo.batch_size,n_epochs=algo.n_epochs,
                gamma=algo.gamma,gae_lambda=algo.gae_lambda,clip_range=algo.clip_range,ent_coef=algo.ent_coef,vf_coef=algo.vf_coef,max_grad_norm=algo.max_grad_norm,
                target_kl=algo.target_kl,
                policy_kwargs=_policy_kwargs(
                    algo, job.observation, int(np.prod(train_env.observation_space.shape)), resolve
                ),
                tensorboard_log=str(context.output_dir/"tensorboard"),seed=job.training_seed,device=algo.device,verbose=0)
            if isinstance(algo, DynamicsAuxiliaryPPOSpec):
                model.configure_auxiliary_dynamics(
                    coefficient=algo.auxiliary_loss_coefficient,
                    epochs=algo.auxiliary_epochs,
                    batch_size=algo.auxiliary_batch_size,
                    chlorine_weight=algo.chlorine_weight,
                )
            initialization=_initialize_pretrained_policy(
                model, algo, resolve, bundle=pretrained_bundle, job=job
            )
            completed=0;best=-float("inf");history=[];environment_states=[]
        runtime.update({
            "device": str(model.device),
            "training_environments": int(algo.n_envs),
            "evaluation_workers": int(context.workers),
            "checkpoint_evaluation_execution": (
                "spawned-process" if isolated else "training-process"
            ),
        })
        previous_progress_lock=tqdm.get_lock()
        shared_progress_lock=has_shared_progress_lock()
        progress_lock=(previous_progress_lock if shared_progress_lock or not isolated
                       else mp.get_context("spawn").RLock())
        if isolated and not shared_progress_lock:tqdm.set_lock(progress_lock)
        progress=tqdm(total=total_timesteps,initial=min(completed,total_timesteps),desc=f"{job.name} training",
                      unit="step",leave=True,dynamic_ncols=True,position=0)
        progress_callback=_TqdmProgressCallback(progress,completed,total_timesteps)
        checkpoint_plot_splits=(_plot_splits(job.hooks.on_checkpoint)
                                if self.render_plots else frozenset())
        finish_plot_splits=_plot_splits(job.hooks.on_finish)
        pending=None
        training_finished=False

        def save_logical_checkpoint():
            payload={"job_hash":config_hash(job),"completed_timesteps":completed,"best_reward":best,
                "environment_states":environment_states,"rng":_json_rng(),"history":history,
                "pretrained_policy":initialization,"evaluation_protocol_version":PROTOCOL_VERSION,
                "runtime":runtime}
            if job.name in self.timestep_targets:payload["target_timesteps"]=total_timesteps
            if legacy_evaluation_cutoff is not None:
                payload["legacy_evaluation_cutoff_timesteps"]=legacy_evaluation_cutoff
            if evaluation_queue:payload["pending_evaluations"]=evaluation_queue
            _write(checkpoints/"latest.json",payload)

        def start_next_evaluation():
            nonlocal pending
            if (pending is not None or not evaluation_queue
                    or (self.defer_evaluation and not training_finished)):return
            record=evaluation_queue[0]
            snapshot_dir=_pending_snapshot_path(context.output_dir,record)
            timesteps=int(record["timesteps"])
            pending=_start_checkpoint_evaluation(
                job,dataset,node_ids,dependency_paths,snapshot_dir,timesteps,
                context.output_dir/"plots"/"checkpoints"/f"{timesteps:012d}",
                checkpoint_plot_splits,context.workers,hydraulic_catalog,progress_lock)

        def enqueue_evaluation(timesteps,nominal_thresholds):
            snapshot_dir=_archive_control_checkpoint(
                context.output_dir,checkpoints,timesteps,job,nominal_thresholds)
            evaluation_queue.append({"timesteps":timesteps,"nominal_thresholds":nominal_thresholds,
                "snapshot":str(snapshot_dir.relative_to(context.output_dir))})
            save_logical_checkpoint()
            start_next_evaluation()

        def queue_archived_evaluations():
            evaluated={int(item["timesteps"]) for item in history}
            queued={int(item["timesteps"]) for item in evaluation_queue}
            for snapshot_dir in sorted((checkpoints/"history").glob("*")):
                if not snapshot_dir.is_dir() or not snapshot_dir.name.isdigit():continue
                timesteps=int(snapshot_dir.name)
                if legacy_evaluation_cutoff is not None and timesteps<=legacy_evaluation_cutoff:
                    continue
                manifest=_read(snapshot_dir/"manifest.json")
                if (not isinstance(manifest,dict) or manifest.get("job_hash")!=config_hash(job)
                        or manifest.get("timesteps")!=timesteps):
                    raise RuntimeError(f"archived control checkpoint identity mismatch: {snapshot_dir}")
                if timesteps in evaluated or timesteps in queued:continue
                evaluation_queue.append({
                    "timesteps":timesteps,
                    "nominal_thresholds":manifest.get("nominal_thresholds",[]),
                    "snapshot":str(snapshot_dir.relative_to(context.output_dir)),
                })
            evaluation_queue.sort(key=lambda item:int(item["timesteps"]))
            save_logical_checkpoint()

        def collect_evaluations(*,wait):
            nonlocal pending,best
            collected=False
            while evaluation_queue:
                start_next_evaluation()
                if not wait and not _checkpoint_evaluation_ready(pending):return collected
                result=_finish_checkpoint_evaluation(pending)
                evaluation=result["evaluation"]
                history_item={"timesteps":pending.timesteps,"nominal_thresholds":evaluation_queue[0]["nominal_thresholds"],
                              "evaluation":evaluation}
                evaluation_runtime=result.get("runtime")
                if isinstance(evaluation_runtime,dict):
                    history_item["evaluation_runtime"]=evaluation_runtime
                    elapsed=evaluation_runtime.get("elapsed_seconds")
                    if isinstance(elapsed,int|float):
                        history_item["evaluation_elapsed_seconds"]=float(elapsed)
                        runtime["checkpoint_evaluation_seconds"]+=float(elapsed)
                if "test_evaluation" in result:history_item["test_evaluation"]=result["test_evaluation"]
                history.append(history_item)
                logger=getattr(model,"_logger",None)
                if logger is not None:
                    for key,value in evaluation.items():
                        if isinstance(value,int|float):logger.record(f"evaluation/{key}",value)
                    logger.dump(pending.timesteps)
                if evaluation["mean_reward"]>best:
                    best=evaluation["mean_reward"]
                    _publish_snapshot(context.output_dir/"best",pending.snapshot_dir,job,evaluation,node_ids)
                snapshot_dir=pending.snapshot_dir
                _write(snapshot_dir/"evaluation.json",history_item)
                pending=None
                evaluation_queue.pop(0)
                save_logical_checkpoint()
                collected=True
            return collected

        try:
            if isolated and context.resume and not self.defer_evaluation:
                scheduled_timesteps=[int(item["timesteps"]) for item in evaluation_queue]
                scheduled_timesteps.extend(int(item["timesteps"]) for item in history)
                latest_scheduled=max(scheduled_timesteps,default=0)
                current_is_scheduled=completed in scheduled_timesteps
                missed_current_evaluation=(latest_scheduled//algo.eval_frequency)<(completed//algo.eval_frequency)
                if completed>0 and not current_is_scheduled and (
                        missed_current_evaluation or completed>=total_timesteps):
                    enqueue_evaluation(completed,crossed_thresholds(latest_scheduled,completed,algo.eval_frequency,total_timesteps))
                else:start_next_evaluation()
            if not context.resume:
                model.save(str(checkpoints/"latest"));train_env.save(str(checkpoints/"vec_normalize.pkl"))
                save_logical_checkpoint()
                _archive_control_checkpoint(context.output_dir,checkpoints,0,job,[0])
                if isolated and not self.defer_evaluation:
                    enqueue_evaluation(0,[0])
                elif not self.defer_evaluation:
                    evaluation_started=time.perf_counter()
                    evaluation=_run_evaluation(job,dataset,node_ids,self.backend,dependency_paths,resolve,train_env,model,
                        checkpoints/"latest.zip",checkpoints/"vec_normalize.pkl","evaluation","validation",None,
                        context.workers,hydraulic_catalog=hydraulic_catalog,
                        raw_output_dir=checkpoints/"history"/"000000000000"/"evaluation-data"/"validation")
                    evaluation_elapsed=time.perf_counter()-evaluation_started
                    runtime["checkpoint_evaluation_seconds"]+=evaluation_elapsed
                    history_item={"timesteps":0,"nominal_thresholds":[0],"evaluation":evaluation,
                                  "evaluation_elapsed_seconds":evaluation_elapsed}
                    history.append(history_item)
                    _write(checkpoints/"history"/"000000000000"/"evaluation.json",history_item)
                    best=evaluation["mean_reward"]
                    _publish(context.output_dir/"best",model,train_env,job,evaluation,node_ids)
                    save_logical_checkpoint()
            if isolated and not self.defer_evaluation and completed == 0 and evaluation_queue:
                # Validate the actual initialized policy before the first update.
                collect_evaluations(wait=True)
            while completed<total_timesteps:
                chunk=next_training_chunk(completed,total_timesteps,
                                          algo.eval_frequency,algo.checkpoint_frequency)
                before=model.num_timesteps;previous_completed=completed
                training_started=time.perf_counter()
                try:
                    model.learn(chunk,callback=progress_callback,reset_num_timesteps=False,progress_bar=False,tb_log_name=job.name)
                finally:
                    runtime["training_seconds"]+=time.perf_counter()-training_started
                completed+=model.num_timesteps-before
                model.save(str(checkpoints/"latest"));train_env.save(str(checkpoints/"vec_normalize.pkl"))
                environment_states=[train_env.env_method("runtime_state",indices=index)[0] for index in range(algo.n_envs)]
                # Publish a complete logical checkpoint before evaluation.
                # Validation is comparatively expensive and may fail for
                # reasons unrelated to the training state; the learned model
                # must remain resumable in that case.
                save_logical_checkpoint()
                checkpoint_thresholds=crossed_thresholds(
                    previous_completed,completed,algo.checkpoint_frequency,total_timesteps)
                crossed_evaluation=(previous_completed//algo.eval_frequency)<(completed//algo.eval_frequency)
                if checkpoint_thresholds or (self.defer_evaluation and crossed_evaluation) or completed>=total_timesteps:
                    archive_thresholds=checkpoint_thresholds
                    if self.defer_evaluation and crossed_evaluation:
                        archive_thresholds=sorted(set(checkpoint_thresholds+crossed_thresholds(
                            previous_completed,completed,algo.eval_frequency,total_timesteps)))
                    _archive_control_checkpoint(
                        context.output_dir,checkpoints,completed,job,archive_thresholds)
                if not self.defer_evaluation and (crossed_evaluation or completed>=total_timesteps):
                    if isolated:
                        enqueue_evaluation(completed,crossed_thresholds(previous_completed,completed,algo.eval_frequency,total_timesteps))
                    else:
                        checkpoint_plots=context.output_dir/"plots"/"checkpoints"/f"{completed:012d}"
                        evaluation_thresholds=crossed_thresholds(
                            previous_completed,completed,algo.eval_frequency,total_timesteps)
                        _archive_control_checkpoint(
                            context.output_dir,checkpoints,completed,job,evaluation_thresholds)
                        evaluation_started=time.perf_counter()
                        evaluation=_run_evaluation(job,dataset,node_ids,self.backend,dependency_paths,resolve,train_env,model,
                                                   checkpoints/"latest.zip",checkpoints/"vec_normalize.pkl","evaluation","validation",
                                                   checkpoint_plots/"validation" if "validation" in checkpoint_plot_splits else None,
                                                   context.workers,hydraulic_catalog=hydraulic_catalog,
                                                   raw_output_dir=checkpoints/"history"/f"{completed:012d}"/"evaluation-data"/"validation")
                        history_item={"timesteps":completed,"nominal_thresholds":evaluation_thresholds,
                                      "evaluation":evaluation}
                        if "test" in checkpoint_plot_splits:
                            history_item["test_evaluation"]=_run_evaluation(
                                job,dataset,node_ids,self.backend,dependency_paths,resolve,train_env,model,
                                checkpoints/"latest.zip",checkpoints/"vec_normalize.pkl","test","test",checkpoint_plots/"test",
                                context.workers,hydraulic_catalog=hydraulic_catalog,
                                raw_output_dir=checkpoints/"history"/f"{completed:012d}"/"evaluation-data"/"test")
                        evaluation_elapsed=time.perf_counter()-evaluation_started
                        runtime["checkpoint_evaluation_seconds"]+=evaluation_elapsed
                        history_item["evaluation_elapsed_seconds"]=evaluation_elapsed
                        history.append(history_item)
                        for key,value in evaluation.items():
                            if isinstance(value,int|float):model.logger.record(f"evaluation/{key}",value)
                        model.logger.dump(completed)
                        if evaluation["mean_reward"]>best:
                            best=evaluation["mean_reward"]
                            _publish(context.output_dir/"best",model,train_env,job,evaluation,node_ids)
                        _write(checkpoints/"history"/f"{completed:012d}"/"evaluation.json",history_item)
                if isolated and not self.defer_evaluation:collect_evaluations(wait=False)
                save_logical_checkpoint()
            training_finished=True
            if self.defer_evaluation:
                progress.close()
                if self.on_training_complete is not None:
                    self.on_training_complete()
                hydraulic_catalog=prepare_hydraulic_outflow_cache(
                    dataset,self.backend,workers=context.workers,
                    splits=("evaluation","test"),
                    description=f"{job.name} evaluation hydraulics")
                queue_archived_evaluations()
                collect_evaluations(wait=True)
            else:
                collect_evaluations(wait=True)
                progress.close()
            test_model=None;test_normalization=None
            evaluation_model_path=(context.output_dir/"best"/"model.zip"
                                   if self.evaluate_best_model else checkpoints/"latest.zip")
            evaluation_normalization_path=(context.output_dir/"best"/"vec_normalize.pkl"
                                           if self.evaluate_best_model else checkpoints/"vec_normalize.pkl")
            if not isolated:
                test_model=_algorithm_class(algo).load(str(evaluation_model_path),device=algo.device)
                with evaluation_normalization_path.open("rb") as handle:
                    test_normalization=pickle.load(handle)
            final_plots=context.output_dir/"plots"/"final"
            metrics=context.output_dir/"metrics.json"
            cached=_read(metrics) if training_complete_on_resume else None
            if isinstance(cached,dict) and cached.get("protocol_version") != PROTOCOL_VERSION:
                cached=None
            cached_runtime=cached.get("runtime") if isinstance(cached,dict) else None
            if isinstance(cached_runtime,dict):
                for key in runtime:
                    if isinstance(cached_runtime.get(key),int|float):
                        runtime[key]=max(runtime[key],float(cached_runtime[key]))
            cached_final_validation=cached.get("final_validation") if isinstance(cached,dict) else None
            cached_test=cached.get("test") if isinstance(cached,dict) else None
            final_validation=cached_final_validation if isinstance(cached_final_validation,dict) else None
            if "validation" in finish_plot_splits:
                if final_validation is None:
                    evaluation_started=time.perf_counter()
                    final_validation=_run_evaluation(
                        job,dataset,node_ids,self.backend,dependency_paths,resolve,test_normalization,test_model,
                        evaluation_model_path,evaluation_normalization_path,
                        "evaluation","validation",final_plots/"validation" if self.render_plots else None,context.workers,
                        hydraulic_catalog=hydraulic_catalog,
                        raw_output_dir=context.output_dir/"evaluation-data"/"final"/"validation")
                    runtime["final_validation_seconds"]+=time.perf_counter()-evaluation_started
                    _write(metrics,_control_metrics(best,history,final_validation,None,include_test=False,
                                                   runtime=runtime))
            test=cached_test if job.evaluate_test and isinstance(cached_test,dict) else None
            if job.evaluate_test and test is None:
                evaluation_started=time.perf_counter()
                test=_run_evaluation(
                    job,dataset,node_ids,self.backend,dependency_paths,resolve,test_normalization,test_model,
                    evaluation_model_path,evaluation_normalization_path,
                    "test","test",final_plots/"test" if self.render_plots and "test" in finish_plot_splits else None,
                    context.workers,hydraulic_catalog=hydraulic_catalog,
                    raw_output_dir=context.output_dir/"evaluation-data"/"final"/"test")
                runtime["final_test_seconds"]+=time.perf_counter()-evaluation_started
                _write(metrics,_control_metrics(best,history,final_validation,test,runtime=runtime))
        except Exception as exc:
            _write(context.output_dir/"evaluation-incomplete.json",{
                "status":"incomplete","protocol_version":PROTOCOL_VERSION,"error":str(exc),
                "completed_timesteps":completed})
            raise
        finally:
            try:
                if pending is not None:_cancel_checkpoint_evaluation(pending)
            finally:
                try:progress.close();train_env.close()
                finally:
                    if isolated and not shared_progress_lock:tqdm.set_lock(previous_progress_lock)
        metrics=context.output_dir/"metrics.json"
        _write(metrics,_control_metrics(best,history,final_validation,test,
                                       include_test=job.evaluate_test,runtime=runtime))
        (context.output_dir/"evaluation-incomplete.json").unlink(missing_ok=True)
        best_dir=context.output_dir/"best"
        artifacts={"best":best_dir,"latest":checkpoints,"metrics":metrics}
        checkpoint_history=checkpoints/"history"
        if checkpoint_history.is_dir():artifacts["checkpoint_history"]=checkpoint_history
        evaluation_data=context.output_dir/"evaluation-data"
        if evaluation_data.is_dir():artifacts["evaluation_data"]=evaluation_data
        plots=context.output_dir/"plots"
        if plots.is_dir():artifacts["plots"]=plots
        metadata={"best_mean_validation_reward":best,"logical_resume":context.resume,"node_ids":list(node_ids),
                  "pretrained_policy":initialization,"runtime":runtime}
        if job.name in self.timestep_targets:metadata["target_timesteps"]=total_timesteps
        if job.evaluate_test:metadata["test"]=test
        return RunResult(artifacts,metadata)


def _policy_kwargs(algo, observation, observation_size, resolve=None):
    kwargs = {}
    if not isinstance(algo, DynamicsAuxiliaryPPOSpec):
        kwargs.update({
            "lstm_hidden_size": algo.lstm_hidden_size,
            "n_lstm_layers": algo.n_lstm_layers,
            "shared_lstm": algo.shared_lstm,
            "enable_critic_lstm": algo.enable_critic_lstm,
        })
    if isinstance(observation, FixedPCALatentAssistedControlObservationSpec):
        base_size = observation_size - observation.latent_size
        if base_size <= 0:
            raise ValueError("fixed PCA latent observation has no base features")
        if resolve is None:
            def resolve(reference):
                raise ValueError(
                    f"cannot resolve fixed PCA encoder artifact {reference!r}"
                )
        latent_bundle = resolve_latent_bundle(observation.provider, resolve).resolve()
        kwargs.update(
            {
                "features_extractor_class": FixedPCALatentFeatureExtractor,
                "features_extractor_kwargs": {
                    "base_size": base_size,
                    "latent_size": observation.latent_size,
                    "projected_size": observation.projected_size,
                    "projection": observation.projection,
                    "latent_bundle": str(latent_bundle),
                    "whitening_epsilon": observation.whitening_epsilon,
                    "clip_projected": observation.clip_projected,
                },
            }
        )
    elif (
        getattr(observation, "latent_adapter", None)
        == "ungated-linear-layer-norm"
    ):
        base_size = observation_size - observation.latent_size
        if base_size <= 0:
            raise ValueError("projected latent observation has no base features")
        kwargs.update(
            {
                "features_extractor_class": UngatedProjectedLatentFeatureExtractor,
                "features_extractor_kwargs": {
                    "base_size": base_size,
                    "latent_size": observation.latent_size,
                    "projected_size": observation.projected_size,
                },
            }
        )
    elif isinstance(observation, ProjectedLatentAssistedControlObservationSpec):
        base_size = observation_size - observation.latent_size
        if base_size <= 0:
            raise ValueError("projected latent observation has no base features")
        kwargs.update(
            {
                "features_extractor_class": ProjectedLatentFeatureExtractor,
                "features_extractor_kwargs": {
                    "base_size": base_size,
                    "latent_size": observation.latent_size,
                    "projected_size": observation.projected_size,
                    "initial_gate_logit": observation.initial_gate_logit,
                },
            }
        )
    elif isinstance(observation, SharedDynamicsEncoderControlObservationSpec):
        history_size = observation.history_steps * observation.history_feature_size
        base_size = observation_size - history_size
        if base_size <= 0:
            raise ValueError("shared dynamics observation has no base features")
        if resolve is None:
            def resolve(reference):
                raise ValueError(
                    f"cannot resolve dynamics encoder artifact {reference!r}"
                )
        bundle = resolve_latent_bundle(observation.provider, resolve).resolve()
        kwargs.update(
            {
                "features_extractor_class": SharedDynamicsEncoderFeatureExtractor,
                "features_extractor_kwargs": {
                    "base_size": base_size,
                    "history_steps": observation.history_steps,
                    "history_feature_size": observation.history_feature_size,
                    "latent_size": observation.latent_size,
                    "bundle": str(bundle),
                },
                "share_features_extractor": True,
                # SB3 otherwise recursively reinitializes the just-loaded
                # pretrained encoder with orthogonal weights.
                "ortho_init": False,
            }
        )
    elif isinstance(observation, TrainableNextLatentControlObservationSpec):
        history_size = observation.history_steps * observation.history_feature_size
        base_size = observation_size - history_size
        if base_size <= 0:
            raise ValueError("trainable next-latent observation has no base features")
        kwargs.update(
            {
                "features_extractor_class": TrainableNextLatentFeatureExtractor,
                "features_extractor_kwargs": {
                    "base_size": base_size,
                    "history_steps": observation.history_steps,
                    "history_feature_size": observation.history_feature_size,
                    "latent_size": observation.latent_size,
                    "projected_size": observation.projected_size,
                    "num_layers": observation.num_layers,
                    "dropout": observation.dropout,
                },
            }
        )
    return kwargs


def _resolve_pretrained_policy_bundle(algo, resolve=None) -> Path | None:
    reference = algo.pretrained_policy
    if reference is None:
        return None
    if isinstance(reference, ArtifactRef):
        if resolve is None:
            raise ValueError(f"cannot resolve pretrained policy artifact {reference!r}")
        bundle = Path(resolve(reference))
    else:
        bundle = Path(reference)
        if not bundle.is_absolute():
            bundle = PROJECT_ROOT / bundle
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ValueError("behavior-cloned policy must be provided as a bundle directory")
    return bundle


def _read_pretrained_policy_manifest(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"pretrained policy manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pretrained policy manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != 2 or manifest.get("kind") != "recurrent-ppo-behavior-cloning":
        raise ValueError(f"incompatible pretrained policy manifest: {manifest_path}")
    return manifest


def _validate_pretrained_policy_contract(
    manifest: dict, job: ControlTrainingSpec
) -> None:
    action = manifest.get("action_contract", {})
    expected_action = {
        "raw_minimum": -1.0,
        "raw_maximum": 1.0,
        "max_delta": job.max_delta,
        "minimum_injection": 0.0,
        "maximum_injection": job.max_injection,
        "initial_injection": job.initial_injection,
        "cumulative": True,
    }
    if action != expected_action:
        raise ValueError(
            "pretrained policy action contract does not match the target control job"
        )
    observation = manifest.get("policy_observation_contract", {})
    if observation.get("max_injection") != job.max_injection:
        raise ValueError(
            "pretrained policy observation injection denominator does not match "
            "the target control job"
        )
    reward = manifest.get("reward", {})
    if (
        reward.get("id") != getattr(job.reward.factory, "reward_id", None)
        or reward.get("parameters") != job.reward.as_dict()
    ):
        raise ValueError(
            "pretrained policy reward identity does not match the target control job"
        )


def _initialize_pretrained_policy(
    model, algo, resolve=None, *, bundle=None, job: ControlTrainingSpec | None = None
):
    if algo.pretrained_policy is None:
        return None
    if isinstance(algo, DynamicsAuxiliaryPPOSpec):
        raise ValueError("behavior-cloned recurrent policies cannot initialize dynamics-auxiliary PPO")
    bundle = bundle or _resolve_pretrained_policy_bundle(algo, resolve)
    assert bundle is not None
    if job is None:
        raise ValueError(
            "target control job is required to validate a pretrained policy contract"
        )
    model_path = bundle / "model.zip"
    normalization_path = bundle / "vec_normalize.pkl"
    manifest = _read_pretrained_policy_manifest(bundle)
    _validate_pretrained_policy_contract(manifest, job)
    preprocessing = manifest.get("observation_preprocessing", {})
    if (
        preprocessing.get("kind") != "vec-normalize"
        or preprocessing.get("artifact") != "vec_normalize.pkl"
        or preprocessing.get("norm_obs") is not True
    ):
        raise ValueError("pretrained policy does not declare VecNormalize observations")
    if not model_path.is_file():
        raise ValueError(f"pretrained policy model not found: {model_path}")
    if not normalization_path.is_file():
        raise ValueError(f"pretrained VecNormalize state not found: {normalization_path}")
    components = algo.pretrained_components
    transferred_std = None
    if "actor" in components or "action_std" in components:
        source = RecurrentPPO.load(str(model_path), device=algo.device)
        if (
            source.observation_space.shape != model.observation_space.shape
            or source.action_space.shape != model.action_space.shape
            or not np.array_equal(source.action_space.low, model.action_space.low)
            or not np.array_equal(source.action_space.high, model.action_space.high)
        ):
            raise ValueError("pretrained policy observation or action space is incompatible")
        source_policy = source.policy
        target_policy = model.policy
        if "actor" in components:
            if (
                source_policy.lstm_actor.input_size != target_policy.lstm_actor.input_size
                or source_policy.lstm_actor.hidden_size != target_policy.lstm_actor.hidden_size
                or source_policy.lstm_actor.num_layers != target_policy.lstm_actor.num_layers
            ):
                raise ValueError("pretrained policy actor LSTM architecture is incompatible")
            target_policy.lstm_actor.load_state_dict(source_policy.lstm_actor.state_dict())
            target_policy.mlp_extractor.policy_net.load_state_dict(
                source_policy.mlp_extractor.policy_net.state_dict()
            )
            target_policy.action_net.load_state_dict(source_policy.action_net.state_dict())
        if "action_std" in components:
            declared_std = manifest.get("policy", {}).get("initial_action_std")
            if (not isinstance(declared_std, int | float) or isinstance(declared_std, bool)
                    or not np.isfinite(declared_std) or not 0 < declared_std <= 1
                    or not torch.allclose(
                        source_policy.log_std.detach().cpu(),
                        torch.full_like(source_policy.log_std.detach().cpu(), float(np.log(declared_std))),
                        rtol=0, atol=1e-6,
                    )):
                raise ValueError("pretrained model action standard deviation disagrees with its manifest")
            with torch.no_grad():
                target_policy.log_std.copy_(source_policy.log_std)
            transferred_std = float(declared_std)
    return {
        "bundle": str(bundle),
        "model": str(model_path),
        "vec_normalize": str(normalization_path),
        "components": list(components),
        "initial_action_std": transferred_std,
        "initial_action_std_source": (
            manifest.get("policy", {}).get("initial_action_std_source")
            if transferred_std is not None else None
        ),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "best_epoch": manifest.get("best_epoch"),
    }


def _algorithm_class(algo):
    if isinstance(algo, DynamicsAuxiliaryPPOSpec):
        return DynamicsAuxiliaryPPO
    return RecurrentPPO


class _TqdmProgressCallback(BaseCallback):
    def __init__(self, progress, current, total):
        super().__init__(verbose=0);self.progress=progress;self.current=current;self.total=total
    def _on_step(self):
        current=min(int(self.num_timesteps),self.total)
        if current>self.current:self.progress.update(current-self.current);self.current=current
        return True
    def _on_rollout_end(self):
        extractor = getattr(self.model.policy, "features_extractor", None)
        if isinstance(extractor, ProjectedLatentFeatureExtractor):
            self.logger.record(
                "train/latent_adapter_gate",
                float(extractor.gate_value.detach().cpu()),
            )


def _run_evaluation(job,dataset,node_ids,backend,dependency_paths,resolve,normalization,model,
                    model_path,normalization_path,split,phase,plot_dir,workers,*,hydraulic_catalog=None,
                    raw_output_dir=None):
    if not isinstance(backend,EpytFlowBackend):
        return _evaluate(job,dataset,node_ids,backend,resolve,normalization,model,split,phase,plot_dir,
                         hydraulic_catalog,raw_output_dir=raw_output_dir)
    # EPANET-MSX leaks native temporary-file descriptors when a project is
    # closed. Keep every complete evaluation in a disposable spawned process.
    context=mp.get_context("spawn")
    receive,send=context.Pipe(duplex=False)
    cancellation=context.Event()
    process=context.Process(target=_evaluate_saved_worker,args=(send,cancellation,job,dataset,tuple(node_ids),dependency_paths,
                            Path(model_path),Path(normalization_path),split,phase,plot_dir,workers,
                            hydraulic_catalog,raw_output_dir))
    # The spawned interpreter imports both PyTorch and EPANET, which ship
    # separate OpenMP runtimes on macOS. Limit native threading before either
    # runtime is imported in the child to avoid oversubscription and libomp
    # initialization failures.
    with _native_worker_thread_limits():process.start()
    send.close()
    try:
        try:ok,payload=receive.recv()
        except EOFError as exc:
            process.join()
            raise RuntimeError(f"evaluation worker exited without a result (exit code {process.exitcode})") from exc
    except KeyboardInterrupt:
        cancellation.set()
        process.join(timeout=5)
        if process.is_alive():process.terminate()
        process.join();raise
    except BaseException:
        cancellation.set()
        if process.is_alive():process.terminate()
        process.join();raise
    finally:receive.close()
    process.join()
    if not ok:raise RuntimeError(f"evaluation worker failed:\n{payload}")
    if process.exitcode!=0:raise RuntimeError(f"evaluation worker exited with code {process.exitcode}")
    return payload


def _archive_control_checkpoint(output_dir,checkpoints,timesteps,job,nominal_thresholds):
    snapshot=Path(output_dir)/"checkpoints"/"history"/f"{timesteps:012d}"
    required=("model.zip","vec_normalize.pkl","trainer_state.json","manifest.json")
    if snapshot.exists():
        if not all((snapshot/name).is_file() for name in required):
            raise RuntimeError(f"archived control checkpoint is incomplete: {snapshot}")
        manifest=_read(snapshot/"manifest.json")
        if (not isinstance(manifest,dict) or manifest.get("job_hash")!=config_hash(job)
                or manifest.get("timesteps")!=timesteps):
            raise RuntimeError(f"archived control checkpoint identity mismatch: {snapshot}")
        combined=sorted(set(manifest.get("nominal_thresholds",()))|set(nominal_thresholds))
        if combined!=manifest.get("nominal_thresholds"):
            manifest["nominal_thresholds"]=combined
            _write(snapshot/"manifest.json",manifest)
        return snapshot
    snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copy2(checkpoints/"latest.zip",snapshot/"model.zip")
    shutil.copy2(checkpoints/"vec_normalize.pkl",snapshot/"vec_normalize.pkl")
    shutil.copy2(checkpoints/"latest.json",snapshot/"trainer_state.json")
    _write(snapshot/"manifest.json",{
        "schema_version":1,"kind":"recurrent-ppo-training-checkpoint",
        "job_hash":config_hash(job),"timesteps":timesteps,
        "nominal_thresholds":list(nominal_thresholds),
        "artifacts":{"model":"model.zip","vec_normalize":"vec_normalize.pkl",
                     "trainer_state":"trainer_state.json"},
    })
    return snapshot


def _pending_snapshot_path(output_dir,record):
    if not isinstance(record,dict) or not isinstance(record.get("timesteps"),int):
        raise RuntimeError("invalid pending control evaluation checkpoint")
    relative=Path(record.get("snapshot", ""))
    if relative.is_absolute() or not relative.parts:
        raise RuntimeError("invalid pending control evaluation snapshot path")
    root=Path(output_dir).resolve();snapshot=(root/relative).resolve()
    if root not in snapshot.parents:
        raise RuntimeError("pending control evaluation snapshot escapes the run directory")
    if not (snapshot/"model.zip").is_file() or not (snapshot/"vec_normalize.pkl").is_file():
        raise RuntimeError(f"pending control evaluation snapshot is incomplete: {snapshot}")
    return snapshot


def _start_checkpoint_evaluation(job,dataset,node_ids,dependency_paths,snapshot_dir,timesteps,
                                 plot_root,plot_splits,workers,hydraulic_catalog,progress_lock):
    context=mp.get_context("spawn")
    receive,send=context.Pipe(duplex=False)
    cancellation=context.Event()
    process=context.Process(target=_checkpoint_evaluation_worker,args=(
        send,cancellation,job,dataset,tuple(node_ids),dependency_paths,
        Path(snapshot_dir)/"model.zip",Path(snapshot_dir)/"vec_normalize.pkl",
        Path(plot_root),tuple(plot_splits),workers,hydraulic_catalog,progress_lock))
    try:
        with _native_worker_thread_limits():process.start()
    except BaseException:
        receive.close();send.close();raise
    send.close()
    return _PendingCheckpointEvaluation(
        process,receive,cancellation,Path(snapshot_dir),timesteps)


def _checkpoint_evaluation_worker(connection,cancellation,job,dataset,node_ids,dependency_paths,
                                  model_path,normalization_path,plot_root,plot_splits,workers,
                                  hydraulic_catalog,progress_lock=None):
    started_at=datetime.now(UTC).isoformat()
    started=time.perf_counter()
    try:
        if progress_lock is not None:tqdm.set_lock(progress_lock)
        validation=_evaluate_saved(
            job,dataset,node_ids,dependency_paths,model_path,normalization_path,
            "evaluation","validation",plot_root/"validation" if "validation" in plot_splits else None,
            workers,hydraulic_catalog,cancellation=cancellation,progress_position=1,
            raw_output_dir=Path(model_path).parent/"evaluation-data"/"validation")
        result={"evaluation":validation}
        if "test" in plot_splits:
            result["test_evaluation"]=_evaluate_saved(
                job,dataset,node_ids,dependency_paths,model_path,normalization_path,
                "test","test",plot_root/"test",workers,hydraulic_catalog,cancellation=cancellation,
                progress_position=1,
                raw_output_dir=Path(model_path).parent/"evaluation-data"/"test")
        result["runtime"]={
            "started_at":started_at,
            "finished_at":datetime.now(UTC).isoformat(),
            "elapsed_seconds":time.perf_counter()-started,
            "workers":int(workers),
        }
        connection.send((True,result))
    except BaseException:
        connection.send((False,traceback.format_exc()))
    finally:
        connection.close()


def _checkpoint_evaluation_ready(pending):
    return pending.receive.poll()


def _finish_checkpoint_evaluation(pending):
    try:
        try:ok,payload=pending.receive.recv()
        except EOFError as exc:
            pending.process.join()
            raise RuntimeError(
                f"checkpoint evaluation worker exited without a result (exit code {pending.process.exitcode})"
            ) from exc
    except KeyboardInterrupt:
        _cancel_checkpoint_evaluation(pending);raise
    except BaseException:
        _cancel_checkpoint_evaluation(pending);raise
    finally:
        pending.receive.close()
    pending.process.join()
    if not ok:raise RuntimeError(f"checkpoint evaluation worker failed:\n{payload}")
    if pending.process.exitcode!=0:
        raise RuntimeError(f"checkpoint evaluation worker exited with code {pending.process.exitcode}")
    return payload


def _cancel_checkpoint_evaluation(pending):
    pending.cancellation.set()
    pending.process.join(timeout=5)
    if pending.process.is_alive():pending.process.terminate()
    pending.process.join()
    pending.receive.close()


def _evaluate_saved_worker(connection,cancellation,*args):
    try:connection.send((True,_evaluate_saved(*args,cancellation=cancellation)))
    except BaseException:connection.send((False,traceback.format_exc()))
    finally:connection.close()


def _evaluate_saved(job,dataset,node_ids,dependency_paths,model_path,normalization_path,split,phase,plot_dir,workers,
                    hydraulic_catalog,raw_output_dir=None,cancellation=None,progress_position=None):
    if workers > 1:
        schedule_spec=dataset.schedule_for(split)
        scenarios=dataset.scenarios_for(split)
        schedules=_fixed_schedules(scenarios,schedule_spec,hydraulic_catalog)
        return _evaluate_parallel(job,dataset,node_ids,dependency_paths,model_path,normalization_path,
                                  split,phase,plot_dir,scenarios,schedules,schedule_spec,workers,cancellation,
                                  progress_position,raw_output_dir)
    def resolve(reference):return dependency_paths[reference.job][reference.artifact]
    model=_algorithm_class(job.algorithm).load(str(model_path),device=job.algorithm.device)
    with Path(normalization_path).open("rb") as handle:
        normalization=pickle.load(handle)
    return _evaluate(job,dataset,node_ids,EpytFlowBackend(),resolve,normalization,model,split,phase,plot_dir,
                     hydraulic_catalog,progress_position,raw_output_dir=raw_output_dir)


def _evaluate(job,dataset,node_ids,backend,resolve,train_env,model,split="evaluation",phase="validation",plot_dir=None,
              hydraulic_catalog=None,progress_position=None,raw_output_dir=None):
    schedule_spec=dataset.schedule_for(split)
    provider=_make_observation_provider(job.observation,node_ids,resolve,dataset.hydraulic_timestep_seconds)
    builder=ControlObservationBuilder(job.observation,node_ids,job.max_injection,provider)
    scenarios=dataset.scenarios_for(split)
    schedules=_fixed_schedules(scenarios,schedule_spec,hydraulic_catalog)
    evaluation_count=len(schedules)
    def fixed(_item,episode):return schedules[episode%len(schedules)]
    env=ControlEnv(scenarios,backend,None,builder,create_reward(job.reward),dataset.hydraulic_timestep_seconds,
        dataset.duration_seconds//dataset.hydraulic_timestep_seconds,job.max_injection,job.max_delta,job.initial_injection,
        dataset.network,fixed_schedule_loader=fixed)
    vec=_evaluation_vec_normalize(
        DummyVecEnv([lambda:Monitor(env)]),
        train_env,
        job.algorithm.clip_observation,
    )
    rewards=[];episodes_info=[]
    progress=tqdm(total=evaluation_count*2,desc=f"{job.name} {phase}",unit="simulation",leave=False,
                  dynamic_ncols=True,position=0 if progress_position is None else progress_position)
    try:
        reference_masks=_zero_injection_reference_masks(scenarios,schedules,backend,dataset,progress)
        obs=vec.reset()
        for episode in range(evaluation_count):
            state=None;starts=np.ones(1,dtype=bool);total=0.0;episode_info=[]
            while True:
                action,state=model.predict(obs,state=state,episode_start=starts,deterministic=True);obs,reward,done,info=vec.step(action)
                starts=done;info[0]["evaluation_reward"]=float(vec.get_original_reward()[0]);episode_info.append(info[0])
                if info[0]["time_seconds"]>WARMUP_DAYS*86400:total+=info[0]["evaluation_reward"]
                if done[0]:break
            rewards.append(total);episodes_info.append(episode_info)
            if plot_dir is not None:
                scenario=scenarios[episode%len(scenarios)]
                repetition=episode//len(scenarios)
                trajectory=_plot_trajectory(
                    episode_info,node_ids,schedules[episode],dataset.hydraulic_timestep_seconds)
                render_control_trajectory(
                    _evaluation_plot_path(Path(plot_dir),scenario.scenario_id,repetition,schedule_spec is not None),trajectory,
                    f"{job.name} · {phase} · {scenario.scenario_id} · episode {repetition + 1}",
                )
            progress.update()
    finally:progress.close();vec.close()
    result=_operational_metrics(episodes_info,reference_masks,dataset)
    if raw_output_dir is not None:
        _write_evaluation_timeseries(
            Path(raw_output_dir),episodes_info,reference_masks,dataset,node_ids,split,phase)
    return result


_PARALLEL_EVALUATION_STATE = {}


def _evaluate_parallel(job,dataset,node_ids,dependency_paths,model_path,normalization_path,
                       split,phase,plot_dir,scenarios,schedules,schedule_spec,workers,cancellation=None,
                       progress_position=None,raw_output_dir=None):
    if dependency_paths is None or model_path is None or normalization_path is None:
        raise RuntimeError("parallel evaluation requires saved model and dependency paths")
    worker_count=min(workers,len(schedules))
    provider_states=_parallel_provider_states(job,dataset,node_ids,len(schedules))
    context=mp.get_context("spawn")
    results=[]
    progress=tqdm(total=len(schedules)*2,desc=f"{job.name} {phase}",unit="simulation",leave=False,
                  dynamic_ncols=True,position=0 if progress_position is None else progress_position)
    # Workers must not handle terminal SIGINT themselves. The evaluation
    # process owns the pool and can then close every semaphore before exiting.
    previous_sigint=signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        try:
            pool=context.Pool(worker_count,initializer=_parallel_evaluation_init,initargs=(
                job,dataset,tuple(node_ids),dependency_paths,Path(model_path),Path(normalization_path),
                split,phase,Path(plot_dir) if plot_dir is not None else None,tuple(scenarios),tuple(schedules),
                schedule_spec is not None,provider_states,
            ))
        finally:
            signal.signal(signal.SIGINT,previous_sigint)
    except BaseException:
        progress.close();raise
    try:
        pending=pool.imap_unordered(_parallel_evaluation_episode,range(len(schedules)),chunksize=1)
        while len(results)<len(schedules):
            if cancellation is not None and cancellation.is_set():raise KeyboardInterrupt
            try:result=pending.next(timeout=.1)
            except mp.TimeoutError:continue
            results.append(result);progress.update(2)
    except BaseException:
        pool.terminate();raise
    else:
        pool.close()
    finally:
        pool.join();progress.close()
    results.sort(key=lambda item:item[0])
    rewards=[item[1] for item in results]
    episodes_info=[item[2] for item in results]
    reference_masks=[item[3] for item in results]
    result=_operational_metrics(episodes_info,reference_masks,dataset)
    if raw_output_dir is not None:
        _write_evaluation_timeseries(
            Path(raw_output_dir),episodes_info,reference_masks,dataset,node_ids,split,phase)
    return result


def _parallel_evaluation_init(job,dataset,node_ids,dependency_paths,model_path,normalization_path,
                              split,phase,plot_dir,scenarios,schedules,has_arsenic_schedules,provider_states):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    model=_algorithm_class(job.algorithm).load(str(model_path),device=job.algorithm.device)
    with Path(normalization_path).open("rb") as handle:
        normalization=pickle.load(handle)
    _PARALLEL_EVALUATION_STATE.clear()
    _PARALLEL_EVALUATION_STATE.update({
        "job":job,"dataset":dataset,"node_ids":node_ids,"dependency_paths":dependency_paths,
        "split":split,"phase":phase,"plot_dir":plot_dir,"scenarios":scenarios,"schedules":schedules,
        "has_arsenic_schedules":has_arsenic_schedules,"provider_states":provider_states,
        "model":model,"normalization":normalization,
    })


@contextmanager
def _native_worker_thread_limits():
    limits=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS")
    previous={name:os.environ.get(name) for name in limits}
    try:
        for name in limits:os.environ[name]="1"
        yield
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def _parallel_evaluation_episode(index):
    state=_PARALLEL_EVALUATION_STATE;job=state["job"];dataset=state["dataset"]
    scenarios=state["scenarios"];scenario=scenarios[index%len(scenarios)];schedule=state["schedules"][index]
    backend=EpytFlowBackend()
    reference=_zero_injection_reference_masks((scenario,),(schedule,),backend,dataset)[0]
    dependency_paths=state["dependency_paths"]
    def resolve(reference):return dependency_paths[reference.job][reference.artifact]
    node_ids=state["node_ids"]
    provider=_make_observation_provider(job.observation,node_ids,resolve,dataset.hydraulic_timestep_seconds)
    provider_state=state["provider_states"][index]
    if provider is not None and provider_state is not None:provider.load_state_dict(provider_state)
    builder=ControlObservationBuilder(job.observation,node_ids,job.max_injection,provider)
    env=ControlEnv((scenario,),backend,None,builder,create_reward(job.reward),dataset.hydraulic_timestep_seconds,
        dataset.duration_seconds//dataset.hydraulic_timestep_seconds,job.max_injection,job.max_delta,job.initial_injection,
        dataset.network,fixed_schedule_loader=lambda _item,_episode:schedule)
    normalization=state["normalization"]
    vec=_evaluation_vec_normalize(
        DummyVecEnv([lambda:Monitor(env)]),
        normalization,
        job.algorithm.clip_observation,
    )
    total=0.0;episode_info=[]
    try:
        obs=vec.reset();recurrent_state=None;starts=np.ones(1,dtype=bool)
        while True:
            action,recurrent_state=state["model"].predict(obs,state=recurrent_state,episode_start=starts,deterministic=True)
            obs,reward,done,info=vec.step(action);starts=done
            info[0]["evaluation_reward"]=float(vec.get_original_reward()[0]);episode_info.append(info[0])
            if info[0]["time_seconds"]>WARMUP_DAYS*86400:total+=info[0]["evaluation_reward"]
            if done[0]:break
        plot_dir=state["plot_dir"]
        if plot_dir is not None:
            repetition=index//len(scenarios)
            render_control_trajectory(
                _evaluation_plot_path(plot_dir,scenario.scenario_id,repetition,state["has_arsenic_schedules"]),
                _plot_trajectory(episode_info,node_ids,schedule,dataset.hydraulic_timestep_seconds),
                f"{job.name} · {state['phase']} · {scenario.scenario_id} · episode {repetition + 1}",
            )
    finally:
        vec.close()
    return index,total,episode_info,reference


def _evaluation_vec_normalize(raw, normalization, clip_observation):
    """Mirror training normalization without requiring absent statistics."""
    if isinstance(normalization, BaseOnlyVecNormalize):
        vec = BaseOnlyVecNormalize(
            raw,
            base_size=normalization.base_size,
            training=False,
            norm_obs=normalization.norm_obs,
            norm_reward=False,
            clip_obs=clip_observation,
        )
    else:
        vec = VecNormalize(
            raw,
            training=False,
            norm_obs=normalization.norm_obs,
            norm_reward=False,
            clip_obs=clip_observation,
        )
    # VecNormalize deliberately omits obs_rms when norm_obs is false.
    if normalization.norm_obs:
        vec.obs_rms = normalization.obs_rms
    return vec


def _training_vec_normalize(
    raw, algorithm, observation, *, pretrained_bundle: Path | None = None
):
    kwargs = {
        "norm_obs": algorithm.normalize_observations,
        "norm_reward": algorithm.normalize_rewards,
        "clip_obs": algorithm.clip_observation,
        "clip_reward": algorithm.clip_reward,
    }
    if pretrained_bundle is not None and "observation_normalization" in algorithm.pretrained_components:
        if isinstance(observation, FixedPCALatentAssistedControlObservationSpec):
            raise ValueError(
                "behavior-cloned VecNormalize is incompatible with fixed-PCA prefix normalization"
            )
        path = pretrained_bundle / "vec_normalize.pkl"
        if not path.is_file():
            raise ValueError(f"pretrained VecNormalize state not found: {path}")
        vec = VecNormalize.load(str(path), raw)
        if not vec.norm_obs or not algorithm.normalize_observations:
            raise ValueError(
                "pretrained VecNormalize and PPO must both normalize observations"
            )
        if not np.isclose(vec.clip_obs, algorithm.clip_observation):
            raise ValueError(
                "pretrained VecNormalize observation clipping disagrees with PPO"
            )
        vec.training = True
        vec.norm_reward = algorithm.normalize_rewards
        vec.clip_reward = algorithm.clip_reward
        vec.gamma = algorithm.gamma
        return vec
    if isinstance(
        observation,
        (
            FixedPCALatentAssistedControlObservationSpec,
            TrainableNextLatentControlObservationSpec,
        ),
    ):
        suffix_size = (
            observation.latent_size
            if isinstance(observation, FixedPCALatentAssistedControlObservationSpec)
            else observation.history_steps * observation.history_feature_size
        )
        base_size = int(np.prod(raw.observation_space.shape)) - suffix_size
        if base_size <= 0:
            raise ValueError("prefix-normalized observation has no base features")
        return BaseOnlyVecNormalize(raw, base_size=base_size, **kwargs)
    return VecNormalize(raw, **kwargs)


def _parallel_provider_states(job,dataset,node_ids,evaluation_count):
    if not isinstance(job.observation,ForecastAssistedControlObservationSpec) or not isinstance(job.observation.provider,NoiseForecastProviderSpec):
        return (None,)*evaluation_count
    spec=job.observation.provider
    rng=np.random.default_rng(spec.seed);shape=(job.observation.horizon_steps,len(node_ids))
    predictions_per_episode=dataset.duration_seconds//dataset.hydraulic_timestep_seconds+1
    states=[]
    for _episode in range(evaluation_count):
        states.append(pickle.loads(pickle.dumps({"bit_generator_state":rng.bit_generator.state})))
        for _prediction in range(predictions_per_episode):
            if spec.distribution=="normal":rng.normal(0,spec.standard_deviation,shape)
            else:rng.uniform(-spec.standard_deviation,spec.standard_deviation,shape)
    return tuple(states)


def _make_observation_provider(observation,node_ids,resolve,timestep_seconds):
    if isinstance(observation,ForecastAssistedControlObservationSpec):
        return make_provider(
            observation.provider, observation.horizon_steps, node_ids, resolve,
            timestep_seconds,
        )
    if isinstance(observation,(
        LatentAssistedControlObservationSpec,
        ProjectedLatentAssistedControlObservationSpec,
        FixedPCALatentAssistedControlObservationSpec,
    )):
        return make_latent_provider(observation.provider,node_ids,resolve)
    if isinstance(
        observation,
        (
            SharedDynamicsEncoderControlObservationSpec,
            TrainableNextLatentControlObservationSpec,
        ),
    ):
        return make_latent_history_provider(observation.provider,node_ids,resolve)
    return None


def _plot_trajectory(rows,node_ids,schedule,timestep_seconds):
    times=np.asarray([row["time_seconds"] for row in rows],dtype=float)
    source_local=np.zeros((len(rows),len(node_ids)),dtype=bool)
    interval_starts=times-float(timestep_seconds)
    node_index={node_id:index for index,node_id in enumerate(node_ids)}
    for event in schedule.events:
        index=node_index.get(event.node_id)
        if index is not None:
            source_local[:,index]|=(interval_starts>=event.start_time_seconds)&(interval_starts<event.end_time_seconds)
    return ControlEvaluationTrajectory(
        time_seconds=times,
        node_ids=tuple(node_ids),
        chlorine=np.asarray([row["chlorine"] for row in rows],dtype=float),
        arsenic=np.asarray([row["arsenic"] for row in rows],dtype=float),
        injection=np.asarray([row["injection"] for row in rows],dtype=float),
        actions=np.asarray([row["action"] for row in rows],dtype=float),
        event_active=np.asarray([row["event_active"] for row in rows],dtype=bool),
        source_local_arsenic=source_local,
    )


def _plot_splits(actions):
    for action in actions:
        if isinstance(action,ControlPlotsActionSpec):return frozenset(action.splits)
    return frozenset()


def _evaluation_plot_path(root,scenario_id,repetition,has_arsenic_schedules):
    if has_arsenic_schedules:return root/scenario_id/f"arsenic-{repetition}.png"
    return root/f"{scenario_id}.png"


def _fixed_schedules(scenarios,spec,hydraulic_catalog=None):
    if spec is None:return tuple(ArsenicSchedule(()) for _scenario in scenarios)
    if isinstance(spec, ArsenicScheduleStreamSpec):
        sources = spawn_schedule_sources(spec, 1, hydraulic_catalog=hydraulic_catalog)
        return tuple(sources[0].next_schedule(scenario.eligible_arsenic_nodes, scenario_id=scenario.scenario_id)
                     for scenario in scenarios)
    return tuple(
        fixed_schedule(
            spec,scenario.scenario_id,schedule_index,scenario.eligible_arsenic_nodes,
            hydraulic_profile=(hydraulic_catalog.profile(scenario.scenario_id)
                if hydraulic_catalog is not None else None))
        for schedule_index in range(spec.schedule_count)
        for scenario in scenarios
    )


def _zero_injection_reference_masks(scenarios,schedules,backend,dataset,progress=None):
    references=[]
    episode_steps=dataset.duration_seconds//dataset.hydraulic_timestep_seconds
    for episode,schedule in enumerate(schedules):
        scenario=scenarios[episode%len(scenarios)]
        session=backend.create_session(str(scenario.path),hydraulic_timestep_seconds=dataset.hydraulic_timestep_seconds,
                                       calendar_start_day=scenario.calendar_start_day,
                                       chlorine_source_ids=scenario.chlorine_source_ids,
                                       monitored_node_ids=scenario.monitored_node_ids)
        try:
            samples=[session.reset(schedule.events)]
            for _step in range(episode_steps):
                sample,done=session.step(0.0,0.0);samples.append(sample)
                if done:break
            times=np.asarray([sample.time_seconds for sample in samples],dtype=float)
            raw=np.asarray([sample.arsenic for sample in samples],dtype=float)
            try:
                cleared=clearance_time(times,raw,episode_end=dataset.duration_seconds)
            except ClearanceError as exc:
                raise ClearanceError(f"Incomplete evaluation: scenario={scenario.scenario_id}, schedule={schedule.hash}: {exc}") from exc
            strengths=np.asarray([sample.event_strength_by_node if sample.event_strength_by_node is not None
                                  else np.zeros(raw.shape[1]) for sample in samples])
            if schedule.events and any(sample.hydraulic_outflow is None for sample in samples):
                raise ValueError("Reference source correction requires hydraulic outflow")
            outflow=np.asarray([sample.hydraulic_outflow if sample.hydraulic_outflow is not None
                                else np.zeros(raw.shape[1]) for sample in samples])
            flow_units=getattr(session,"flow_units",None)
            if schedule.events and flow_units is None:
                raise ValueError("Reference source correction requires declared flow units")
            transported,_=derive_transported_arsenic(raw,strengths,outflow,times,flow_units=flow_units or "LPS")
            event,post=reference_windows(times,transported,schedule.events)
            references.append(ReferenceEvaluation(times,event,post,scenario.scenario_id,schedule.hash,
                                                  cleared,transported,strengths,outflow,flow_units or "LPS",raw))
        finally:session.close()
        if progress is not None:progress.update()
    return references


def _numeric_evaluation_info(rows):
    excluded={"time_seconds","node_ids","chlorine","arsenic","injection","action",
              "event_active","evaluation_reward"}
    arrays={}
    for key in sorted(set.intersection(*(set(row) for row in rows)) if rows else ()):
        if key in excluded:continue
        try:values=np.asarray([row[key] for row in rows])
        except (TypeError,ValueError):continue
        if values.dtype.kind not in "biuf" or values.shape[0]!=len(rows):continue
        arrays[f"info__{key}"]=values
    return arrays


def _write_evaluation_timeseries(output_dir,episodes,references,dataset,node_ids,split,phase):
    """Persist lossless numeric policy and zero-injection evaluation trajectories."""
    output_dir.mkdir(parents=True,exist_ok=True)
    cases=[]
    for index,(episode,reference) in enumerate(zip(episodes,references,strict=True)):
        times=np.asarray([row["time_seconds"] for row in episode],dtype=np.float64)
        chlorine=np.asarray([row["chlorine"] for row in episode],dtype=np.float64)
        arsenic=np.asarray([row["arsenic"] for row in episode],dtype=np.float64)
        indices=np.searchsorted(reference.times,times)
        raw=np.zeros_like(reference.transported)
        raw[indices]=arsenic
        transported,_=derive_transported_arsenic(
            raw,reference.strengths,reference.outflow,reference.times,
            flow_units=reference.flow_units)
        arrays={
            "time_seconds":times,
            "chlorine":chlorine,
            "arsenic":arsenic,
            "transported_arsenic":transported[indices],
            "injection":np.asarray([row["injection"] for row in episode],dtype=np.float64),
            "action":np.asarray([row["action"] for row in episode],dtype=np.float64),
            "event_active":np.asarray([row["event_active"] for row in episode],dtype=bool),
            "evaluation_reward":np.asarray([row["evaluation_reward"] for row in episode],dtype=np.float64),
            "reference_time_seconds":np.asarray(reference.times,dtype=np.float64),
            "reference_event":np.asarray(reference.event,dtype=bool),
            "reference_post_source":np.asarray(reference.post_source,dtype=bool),
            "reference_transported_arsenic":np.asarray(reference.transported,dtype=np.float64),
            "reference_event_strength_by_node":np.asarray(reference.strengths,dtype=np.float64),
            "reference_hydraulic_outflow":np.asarray(reference.outflow,dtype=np.float64),
        }
        if reference.raw_arsenic is not None:
            arrays["reference_arsenic"]=np.asarray(reference.raw_arsenic,dtype=np.float64)
        arrays.update(_numeric_evaluation_info(episode))
        filename=f"case-{index:05d}.npz"
        np.savez_compressed(output_dir/filename,**arrays)
        cases.append({
            "index":index,"file":filename,"scenario_id":reference.scenario_id,
            "schedule_hash":reference.schedule_hash,
            "clearance_time_seconds":reference.clearance_time,
            "flow_units":reference.flow_units,"sample_count":len(episode),
            "reference_sample_count":len(reference.times),
            "arrays":{name:{"shape":list(value.shape),"dtype":str(value.dtype)}
                      for name,value in arrays.items()},
        })
    _write(output_dir/"manifest.json",{
        "schema_version":1,"kind":"control-evaluation-timeseries",
        "split":split,"phase":phase,"node_ids":list(node_ids),
        "hydraulic_timestep_seconds":dataset.hydraulic_timestep_seconds,
        "duration_seconds":dataset.duration_seconds,"case_count":len(cases),"cases":cases,
    })


def _operational_metrics(episodes,reference_masks,dataset):
    records=[]
    for episode,reference in zip(episodes,reference_masks,strict=True):
        times=np.asarray([row["time_seconds"] for row in episode],dtype=float)
        indices=np.searchsorted(reference.times,times)
        if (not len(times) or np.any(indices>=len(reference.times))
                or not np.array_equal(reference.times[indices],times)
                or times[0]>WARMUP_DAYS*86400 or times[-1]!=dataset.duration_seconds
                or not np.allclose(np.diff(times),dataset.hydraulic_timestep_seconds)):
            raise ValueError("Control evaluation must cover all scored reporting intervals")
        # The full post-warm-up trajectory is present even with history pre-roll.
        # Arsenic inputs begin after warm-up, so missing initial history is zero.
        raw=np.zeros_like(reference.transported)
        raw[indices]=np.asarray([row["arsenic"] for row in episode])
        transported,_=derive_transported_arsenic(raw,reference.strengths,reference.outflow,
                                               reference.times,flow_units=reference.flow_units)
        record=physical_metrics(times,np.asarray([r["chlorine"] for r in episode]),transported[indices],
            np.asarray([r["injection"] for r in episode]),reference.event[indices],reference.post_source[indices],
            dataset.hydraulic_timestep_seconds,duration_seconds=dataset.duration_seconds)
        main,tail=scoring_windows(times,duration_seconds=dataset.duration_seconds)
        record["diagnostics"]=control_metrics(
            np.asarray([r["chlorine"] for r in episode])[main],transported[indices][main],
            np.asarray([r["injection"] for r in episode])[main],reference.event[indices][main],
            dataset.hydraulic_timestep_seconds)
        record["values"]["evaluation_return"]=float(sum(row["evaluation_reward"] for row,score in zip(episode,main|tail,strict=True) if score))
        ref_main,ref_tail=scoring_windows(reference.times,duration_seconds=dataset.duration_seconds)
        record.update(scenario_id=reference.scenario_id,schedule_hash=reference.schedule_hash,
            clearance_time_seconds=reference.clearance_time,
            zero_injection_arsenic_burden=float(reference.transported[(ref_main|ref_tail)&reference.event].sum()*dataset.hydraulic_timestep_seconds/3600))
        records.append(record)
    result=summarize_cases(records)
    for key in records[0]["diagnostics"]:
        result[key]=scenario_average(records,lambda record:record["diagnostics"][key])
    result["mean_reward"]=result["evaluation_return"]
    result["protocol_version"]=PROTOCOL_VERSION
    result["status"]="complete"
    return result


def _publish(directory,model,env,job,evaluation,node_ids):
    directory.mkdir(parents=True,exist_ok=True);model.save(str(directory/"model"));env.save(str(directory/"vec_normalize.pkl"))
    _write(directory/"manifest.json",{"schema_version":1,"kind":"recurrent-ppo","job":canonical_data(job),"job_hash":config_hash(job),"node_ids":list(node_ids),"evaluation":evaluation})


def _publish_snapshot(directory,snapshot,job,evaluation,node_ids):
    directory.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(snapshot)/"model.zip",directory/"model.zip")
    shutil.copy2(Path(snapshot)/"vec_normalize.pkl",directory/"vec_normalize.pkl")
    _write(directory/"manifest.json",{"schema_version":1,"kind":"recurrent-ppo","job":canonical_data(job),"job_hash":config_hash(job),"node_ids":list(node_ids),"evaluation":evaluation})


def _seed(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def _training_scenario_order(scenarios, training_seed: int, environment_index: int):
    """Return a stable environment-specific permutation of training scenarios."""
    def key(indexed_scenario):
        original_index,scenario=indexed_scenario
        identity=("control-training-scenario-order",training_seed,environment_index,
                  original_index,scenario.scenario_id)
        payload=json.dumps(identity,separators=(",",":"),ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).digest()
    return tuple(scenario for _,scenario in sorted(enumerate(scenarios),key=key))
def _json_rng():
    return {"python_numpy":base64.b64encode(pickle.dumps((random.getstate(),np.random.get_state()))).decode(),"torch":torch.get_rng_state().tolist()}
def _restore_rng(state):
    py_state,np_state=pickle.loads(base64.b64decode(state["python_numpy"]));random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch.tensor(state["torch"],dtype=torch.uint8))
def _read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def _write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def _control_metrics(best,history,final_validation,test,include_test=True,runtime=None):
    result={"protocol_version":PROTOCOL_VERSION,"status":"complete","best_mean_validation_reward":best,"history":history,"final_validation":final_validation,
            "final_timesteps":max((row["timesteps"] for row in history),default=0)}
    if runtime is not None:
        result["runtime"]={
            key:float(value) if isinstance(value,int|float) else value
            for key,value in runtime.items()
        }
    if include_test:result["test"]=test
    return result
