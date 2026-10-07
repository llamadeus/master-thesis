#!/usr/bin/env python3
"""Export five full, matched control rollouts from a completed principal run.

Run on the simulation machine. See README.md, section 8, for the reproduction
commands. This script does not train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

# Match the principal pipeline's single-threaded native worker configuration.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE = "hanoi-principal-v89"
LABELS = {
    "ppo_initial": "PPO initial",
    "ppo_best": "PPO validation best",
    "mpc": "Closed-loop MPC/CEM",
    "bc_initial": "BC initial (0 PPO steps)",
    "finetuned_best": "BC + PPO validation best",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                    + "\n", encoding="utf-8")


def require_files(directory: Path, *names: str) -> None:
    for name in names:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Required completed-run artifact is missing: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control_artifacts(data_root: Path, plan: dict, key: str) -> dict:
    kind = "baseline" if key == "baseline" else "mpc-teacher-bc"
    job_hash = plan["control"][key]["job_hash"]
    run = (data_root / "weights/control/principal/v89" / kind
           / PIPELINE / kind / job_hash)
    require_files(run, "status.json", "metrics.json", "best/manifest.json",
                  "best/model.zip", "best/vec_normalize.pkl")
    if read_json(run / "status.json").get("status") != "completed":
        raise ValueError(f"Control training is not completed: {run}")
    metrics = read_json(run / "metrics.json")
    published = read_json(run / "best/manifest.json")
    job = published["job"]
    if (metrics.get("status") != "complete"
            or metrics.get("final_timesteps", 0) < job["total_timesteps"]
            or published["job_hash"] != job_hash
            or job["training_seed"] != plan["seed"]):
        raise ValueError(f"Run identity or completed training budget mismatch: {run}")
    # Earliest checkpoint wins ties, matching publication/selection behavior.
    history = sorted(metrics["history"], key=lambda row: row["timesteps"])
    selected = max(history, key=lambda row: row["evaluation"]["evaluation_return"])
    if (published["evaluation"] != selected["evaluation"]
            or metrics["final_validation"] != selected["evaluation"]):
        raise ValueError(f"best/ is not the validation-selected policy: {run}")
    initial = run / "checkpoints/history/000000000000"
    require_files(initial, "manifest.json", "model.zip", "vec_normalize.pkl")
    initial_manifest = read_json(initial / "manifest.json")
    if (initial_manifest.get("timesteps") != 0
            or initial_manifest.get("job_hash") != job_hash):
        raise ValueError(f"Invalid archived initial checkpoint: {initial}")
    return {"run": run, "initial": initial, "best": run / "best", "job": job,
            "node_ids": published["node_ids"],
            "selected_timesteps": selected["timesteps"],
            "completed_timesteps": metrics["final_timesteps"]}


def preflight(args) -> dict:
    data_root = args.data_root.resolve()
    state = data_root / "pipelines" / PIPELINE / f"seed-{args.seed}"
    plan = read_json(state / "plan.json")
    status = read_json(state / "status.json")
    if (plan.get("pipeline") != PIPELINE or plan.get("seed") != args.seed
            or status.get("status") != "completed"):
        raise ValueError(f"A completed principal pipeline seed is required: {state}")
    baseline = control_artifacts(data_root, plan, "baseline")
    pretrained = control_artifacts(data_root, plan, "mpc_teacher_bc")
    for key in ("dataset", "observation", "reward", "max_injection", "max_delta",
                "initial_injection", "total_timesteps"):
        if baseline["job"][key] != pretrained["job"][key]:
            raise ValueError(f"Policy contracts disagree: {key}")
    algorithms = [dict(item["job"]["algorithm"]) for item in (baseline, pretrained)]
    for algorithm in algorithms:
        algorithm.pop("pretrained_policy", None)
    if algorithms[0] != algorithms[1]:
        raise ValueError("The two policy algorithm contracts differ beyond BC initialization")
    job = baseline["job"]
    if (job["reward"]["factory"] != "training.rewards.v89:V89Reward"
            or job["observation"]["factory"] != "training.rewards.v89:V89Observation"
            or job["initial_injection"] != 0):
        raise ValueError("This comparison requires plain V89 observations and zero initial injection")
    split = job["dataset"][args.split]
    if args.scenario not in split["scenarios"]:
        raise ValueError(f"{args.scenario} is absent from the saved {args.split} split")
    if not 0 <= args.schedule_index < split["arsenic"]["schedule_count"]:
        raise ValueError("--schedule-index is outside the saved fixed schedule set")
    teacher_dir = (data_root / "datasets/mpc-teacher-cem"
                   / plan["validation_teacher"]["configuration_hash"])
    teacher = read_json(teacher_dir / "manifest.json")
    if (teacher.get("teacher_configuration_hash") != plan["validation_teacher"]["configuration_hash"]
            or teacher["reward"]["parameters"] != job["reward"]["parameters"]
            or teacher["policy_observation_contract"]["spec"] != job["observation"]):
        raise ValueError("MPC and policy reward/observation contracts disagree")
    action = teacher["action_contract"]
    if (action["max_delta"] != job["max_delta"]
            or action["maximum_injection"] != job["max_injection"]
            or action["minimum_injection"] != 0
            or action["initial_injection"] != job["initial_injection"]):
        raise ValueError("MPC and policy action bounds disagree")
    bc = (data_root / "weights/control/mpc-teacher-bc"
          / plan["behavior_cloning"]["configuration_hash"])
    bc_manifest = read_json(bc / "manifest.json")
    bc_reference = pretrained["job"]["algorithm"].get("pretrained_policy")
    if (bc_manifest["reward"] != teacher["reward"]
            or bc_manifest["validation_teacher_configuration_hash"] != plan["validation_teacher"]["configuration_hash"]
            or not isinstance(bc_reference, str)
            or bc_reference.replace("\\", "/").rstrip("/").split("/")[-1] != bc.name):
        raise ValueError("The pretrained run is missing its BC initialization contract")
    bundles = []
    members = {member["job_hash"]: member for member in teacher["ensemble"]["members"]}
    for forecast in plan["forecasting"]["jobs"]:
        run = (data_root / "weights/forecasting/encoder-decoder-lstm-cross-fit"
               / plan["forecasting"]["experiment"] / forecast["name"]
               / forecast["job_hash"])
        if read_json(run / "status.json").get("status") != "completed":
            raise ValueError(f"Forecaster did not complete: {run}")
        bundle = run / "best"
        require_files(bundle, "manifest.json", "model.pt", "normalization.json")
        for name, expected in members[forecast["job_hash"]]["bundle_hashes"].items():
            if sha256(bundle / name) != expected:
                raise ValueError(f"Forecaster differs from the validation teacher: {bundle / name}")
        # Only calendar and proposed injections may enter the decoder as future inputs.
        manifest = read_json(bundle / "manifest.json")
        if any(feature["name"] not in {"cyclic-time", "chlorine-injection"}
               for feature in manifest["job"]["observation"]["future_known"]):
            raise ValueError("MPC requires causally available future-known features")
        bundles.append(bundle)
    if len(bundles) != 5:
        raise ValueError("The principal held-out MPC comparison requires all five fold forecasters")
    # Locate by the fixed schedule enumeration, before seeing any policy outcomes.
    reference_dir = baseline["run"] / "evaluation-data/final" / args.split
    reference_manifest = read_json(reference_dir / "manifest.json")
    if (baseline["node_ids"] != reference_manifest["node_ids"]
            or pretrained["node_ids"] != reference_manifest["node_ids"]):
        raise ValueError("Archived policy and reference node orders disagree")
    case_index = (args.schedule_index * len(split["scenarios"])
                  + split["scenarios"].index(args.scenario))
    case = next(case for case in reference_manifest["cases"] if case["index"] == case_index)
    if case["scenario_id"] != args.scenario:
        raise ValueError("Saved evaluation case ordering disagrees with the run contract")
    require_files(reference_dir, case["file"])
    return dict(plan=plan, baseline=baseline, pretrained=pretrained, job=job,
                teacher=teacher, bundles=tuple(bundles), split=split,
                reference_dir=reference_dir, reference_manifest=reference_manifest, case=case)


def trajectory_from_samples(env, dataset, schedule, schedule_seed: int,
                            extra_future: int = 0):
    import numpy as np
    from packages.trainer.simulation import Trajectory

    samples = list(env.samples)
    if extra_future:
        # Padding supplies calendar positions ONLY. Candidate injection replaces
        # the future injection feature. No future simulator outputs are queried.
        current = samples[-1]
        samples.extend(replace(current, time_seconds=current.time_seconds + offset * env.timestep)
                       for offset in range(1, extra_future + 1))
    first = samples[0]
    def stack(name):
        return np.asarray([getattr(sample, name) for sample in samples])
    return Trajectory(
        network=dataset.network, scenario_id=env.current_scenario.scenario_id,
        calendar_start_day=first.calendar_start_day, chlorine_profile_hash="policy",
        arsenic_schedule_hash=schedule.hash, arsenic_schedule_seed=schedule_seed,
        node_ids=first.node_ids, link_ids=first.link_ids,
        source_ids=env.current_scenario.chlorine_source_ids,
        time_seconds=stack("time_seconds"), pressure=stack("pressure"), demand=stack("demand"),
        flow=stack("flow"), chlorine=stack("chlorine"), arsenic=stack("arsenic"),
        injection=stack("injection"), actions=stack("action"), event_active=stack("event_active"),
        event_strength_by_node=stack("event_strength_by_node"),
        hydraulic_outflow=stack("hydraulic_outflow"), simulator_backend="epyt-flow",
        flow_units=getattr(env.session, "flow_units", None),
    )


def load_reference(inputs):
    import numpy as np
    from packages.trainer.evaluation.protocol import ReferenceEvaluation

    case = inputs["case"]
    with np.load(inputs["reference_dir"] / case["file"], allow_pickle=False) as arrays:
        reference = ReferenceEvaluation(
            times=arrays["reference_time_seconds"].copy(),
            event=arrays["reference_event"].copy(), post_source=arrays["reference_post_source"].copy(),
            scenario_id=case["scenario_id"], schedule_hash=case["schedule_hash"],
            clearance_time=case["clearance_time_seconds"],
            transported=arrays["reference_transported_arsenic"].copy(),
            strengths=arrays["reference_event_strength_by_node"].copy(),
            outflow=arrays["reference_hydraulic_outflow"].copy(), flow_units=case["flow_units"],
            raw_arsenic=arrays["reference_arsenic"].copy(),
        )
    return reference


def export_rollout(directory, trajectory, rows, reference, dataset, metadata, planning):
    import numpy as np
    from packages.trainer.control.runner import _operational_metrics, _write_evaluation_timeseries
    from packages.trainer.forecasting.features import derive_transported_arsenic
    from packages.trainer.simulation import TrajectoryStore

    if (not np.array_equal(trajectory.time_seconds, reference.times)
            or trajectory.flow_units != reference.flow_units
            or not np.allclose(trajectory.event_strength_by_node, reference.strengths)
            or not np.allclose(trajectory.hydraulic_outflow, reference.outflow)):
        raise ValueError("New simulator trajectory differs from the saved reference contract")
    TrajectoryStore().save(directory, trajectory)
    metrics = _operational_metrics([rows], [reference], dataset)
    write_json(directory / "metrics.json", metrics)
    _write_evaluation_timeseries(directory / "evaluation-data", [rows], [reference], dataset,
                               trajectory.node_ids, dataset.split, "five-controller-case")
    transported, _ = derive_transported_arsenic(
        trajectory.arsenic, reference.strengths, reference.outflow,
        trajectory.time_seconds, flow_units=reference.flow_units)
    with (directory / "timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_seconds", "time_hours", "injection_mg_per_min", "raw_action",
                         "reward", "controller_phase"] +
                        [f"{variable}_{node}_mg_per_l" for variable in
                         ("chlorine", "raw_arsenic", "transported_arsenic") for node in trajectory.node_ids])
        for index, seconds in enumerate(trajectory.time_seconds):
            phase = ("initial_observation" if index == 0 else
                     "zero_injection_history" if metadata["approach"] == "mpc" and index < 48 else
                     "mpc_replanning" if metadata["approach"] == "mpc" else "policy")
            writer.writerow([seconds, seconds / 3600, trajectory.injection[index],
                             trajectory.actions[index], "" if index == 0 else rows[index - 1]["evaluation_reward"],
                             phase, *trajectory.chlorine[index], *trajectory.arsenic[index], *transported[index]])
    if planning:
        np.savez_compressed(directory / "mpc-plans.npz",
                            **{key: np.asarray([row[key] for row in planning]) for key in planning[0]})
    write_json(directory / "run.json", dict(metadata, status="complete",
               evaluation_return=metrics["evaluation_return"], metrics=metrics["cases"][0]["values"]))
    return metrics


def run_controller(approach, checkpoint, inputs, args, dataset, schedule, backend,
                   reference, forecast, output):
    import numpy as np
    from sb3_contrib import RecurrentPPO
    from tqdm.auto import tqdm
    from packages.trainer.control.environment import ControlEnv
    from packages.trainer.observations import ControlObservationBuilder
    from packages.trainer.specs import ControlObservationSpec
    from training.rewards.v89 import V89Observation, V89Reward
    from tools.mpc.generate_mpc_teacher_dataset_cem import (
        CEMBatchEvaluation, HISTORY_STEPS, HORIZON_STEPS, actions_to_injections,
        optimize_cem_action_sequence, score_forecast_predictions,
    )

    job = inputs["job"]
    builder = ControlObservationBuilder(
        ControlObservationSpec(V89Observation, job["observation"]["parameters"]),
        dataset.scenarios[0].monitored_node_ids, job["max_injection"])
    env = ControlEnv(dataset.scenarios, backend, None, builder,
                     V89Reward(job["reward"]["parameters"]), dataset.hydraulic_timestep_seconds,
                     dataset.duration_seconds // dataset.hydraulic_timestep_seconds,
                     job["max_injection"], job["max_delta"], job["initial_injection"],
                     dataset.network, fixed_schedule_loader=lambda _scenario, _episode: schedule)
    model = normalization = None
    if checkpoint is not None:
        model = RecurrentPPO.load(checkpoint / "model.zip", device=args.device)
        with (checkpoint / "vec_normalize.pkl").open("rb") as stream:
            normalization = pickle.load(stream)
        normalization.training = False
        normalization.norm_reward = False
        normalization.clip_obs = job["algorithm"]["clip_observation"]
        model.policy.set_training_mode(False)
    directory = output / approach
    directory.mkdir()
    metadata = {"approach": approach, "label": LABELS[approach], "device": args.device,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "deterministic_policy": checkpoint is not None}
    if checkpoint is not None:
        metadata["checkpoint_hashes"] = {name: sha256(checkpoint / name)
                                         for name in ("model.zip", "vec_normalize.pkl", "manifest.json")}
    started = time.perf_counter()
    rows, planning = [], []
    try:
        obs, info = env.reset(seed=args.seed)
        if info["pre_roll_steps"] != 0:
            raise ValueError("Plain V89 policies must begin at timestep zero")
        if env.sample.node_ids != tuple(inputs["reference_manifest"]["node_ids"]):
            raise ValueError("Simulator and archived policy/reference node orders disagree")
        state, starts = None, np.ones(1, dtype=bool)
        for _ in tqdm(range(env.episode_steps), desc=LABELS[approach], unit="step"):
            if model is not None:
                action, state = model.predict(normalization.normalize_obs(obs[None, :]),
                                              state=state, episode_start=starts, deterministic=True)
                starts[:] = False
                raw = float(np.asarray(action).reshape(-1)[0])
            elif len(env.samples) < HISTORY_STEPS:
                raw = 0.0  # Zero injection for the first 47 actions, by user choice.
            else:
                trajectory = trajectory_from_samples(env, dataset, schedule,
                    inputs["split"]["arsenic"]["seed"], extra_future=HORIZON_STEPS)
                index = env.step_count
                query = forecast.query_from_prepared(forecast.prepare_trajectory(trajectory), index)
                reward_state = env.reward.state_dict()
                def evaluator(actions):
                    injections = actions_to_injections(actions, env.injection,
                        max_delta=job["max_delta"], maximum=job["max_injection"])
                    predictions = forecast.predict_candidates(query, injections, env.sample.chlorine)
                    returns = score_forecast_predictions(predictions, actions, injections,
                        reward_state=reward_state, trajectory=trajectory, query_index=index,
                        timestep_seconds=env.timestep,
                        discount_factor=inputs["teacher"]["reward"]["discount_factor"],
                        reward_parameters=job["reward"]["parameters"])
                    return CEMBatchEvaluation(returns, predictions)
                settings = inputs["teacher"]["cem"]
                payload = f"{args.cem_seed}:{args.seed}:{args.scenario}:{schedule.hash}:{index}".encode()
                query_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
                result = optimize_cem_action_sequence(evaluator,
                    candidates=settings["candidates"], iterations=settings["iterations"],
                    elite_fraction=settings["elite_fraction"], seed=query_seed,
                    initial_std=settings["initial_action_std"], min_std=settings["minimum_action_std"])
                raw = float(result.best_actions[0])
                planning.append(dict(query_index=index, time_seconds=env.sample.time_seconds,
                    seed=query_seed, raw_actions=result.best_actions,
                    injections=actions_to_injections(result.best_actions, env.injection,
                        max_delta=job["max_delta"], maximum=job["max_injection"]),
                    predicted_return=float(result.best_return),
                    predicted_chlorine=result.best_predicted_chlorine,
                    best_so_far=result.diagnostics.best_so_far))
            obs, reward, done, truncated, info = env.step(np.asarray([raw], dtype=np.float32))
            rows.append(dict(info, evaluation_reward=float(reward)))
            if done or truncated:
                break
        if env.step_count != env.episode_steps or env.sample.time_seconds != dataset.duration_seconds:
            raise ValueError("Simulator did not complete the entire episode")
        trajectory = trajectory_from_samples(env, dataset, schedule, inputs["split"]["arsenic"]["seed"])
        metadata["elapsed_seconds"] = time.perf_counter() - started
        return export_rollout(directory, trajectory, rows, reference, dataset, metadata, planning)
    finally:
        env.close()


def plot_injections(output, schedule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    for approach, label in LABELS.items():
        with np.load(output / approach / "trajectory.npz", allow_pickle=False) as arrays:
            # Row i records the injection applied over (t[i-1], t[i]].
            axis.step(arrays["time_seconds"] / 3600, arrays["injection"] / 1000,
                      where="pre", label=label, linewidth=1.15)
    for event in schedule.events:
        axis.axvspan(event.start_time_seconds / 3600, event.end_time_seconds / 3600,
                     color="grey", alpha=0.12)
    axis.set(xlabel="Episode time (h)", ylabel="Chlorine injection (10³ mg/min)",
             title="Matched full-episode controller comparison")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.savefig(output / "injection-comparison.png", dpi=180)
    figure.savefig(output / "injection-comparison.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=17, help="Completed principal pipeline seed")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--scenario", default="scenario-90")
    parser.add_argument("--schedule-index", type=int, default=8, help="Zero-based fixed arsenic schedule index")
    parser.add_argument("--cem-seed", type=int, default=102, help="Base seed for deterministic per-step CEM seeds")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--check-inputs", action="store_true", help="Validate saved artifacts only, without simulation")
    args = parser.parse_args()
    if args.cem_seed < 0:
        parser.error("--cem-seed must be non-negative")
    inputs = preflight(args)
    print(f"Artifacts verified for seed {args.seed}, {args.scenario}, {args.split} schedule {args.schedule_index}")
    if args.check_inputs:
        return

    import torch
    from packages.trainer.data import resolve_dataset_split
    from packages.trainer.simulation import prepare_hydraulic_outflow_cache
    from packages.trainer.simulation.epyt import EpytFlowBackend
    from packages.trainer.schedules import fixed_schedule
    from packages.trainer.specs import (DatasetSplitSpec, FixedArsenicScheduleSetSpec,
                                       HydraulicPeakArsenicScheduleGeneratorSpec)
    from tools.mpc.generate_mpc_teacher_dataset_cem import (
        EnsembleForecastRuntime, HISTORY_STEPS, HORIZON_STEPS,
        cem_optimizer_settings, reward_from_manifest, seed_everything,
    )

    torch.set_num_threads(1)
    seed_everything(args.seed)
    reward_from_manifest(inputs["teacher"])  # Reject incomplete reward contracts.
    settings = inputs["teacher"]["cem"]
    expected = cem_optimizer_settings(settings["candidates"], settings["iterations"], settings["elite_fraction"])
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved CEM settings differ from the supported principal parameterization")
    generator = dict(inputs["split"]["arsenic"]["generator"])
    if generator.pop("mode") != "hydraulic-local-peak":
        raise ValueError("Expected the principal hydraulic-local-peak schedule generator")
    schedule_spec = FixedArsenicScheduleSetSpec(
        seed=inputs["split"]["arsenic"]["seed"],
        schedule_count=inputs["split"]["arsenic"]["schedule_count"],
        generator=HydraulicPeakArsenicScheduleGeneratorSpec(**generator))
    spec = inputs["job"]["dataset"]
    internal_split = "evaluation" if args.split == "validation" else "test"
    dataset = resolve_dataset_split(spec["network"],
        DatasetSplitSpec(scenarios=(args.scenario,), arsenic=schedule_spec), split=internal_split,
        scenarios_dir=spec["scenarios_dir"], uncertainty_seed=spec["uncertainty_seed"])
    backend = EpytFlowBackend()
    catalog = prepare_hydraulic_outflow_cache(dataset, backend, workers=1, splits=(internal_split,))
    scenario = dataset.scenarios[0]
    schedule = fixed_schedule(schedule_spec, args.scenario, args.schedule_index,
        scenario.eligible_arsenic_nodes, hydraulic_profile=catalog.profile(args.scenario))
    if schedule.hash != inputs["case"]["schedule_hash"]:
        raise ValueError("Regenerated schedule differs from the completed run. Check scenario/cache/source versions.")
    forecast = EnsembleForecastRuntime(inputs["bundles"], torch.device(args.device))
    if (forecast.node_ids != scenario.monitored_node_ids
            or forecast.source_ids != scenario.chlorine_source_ids
            or forecast.timestep_seconds != dataset.hydraulic_timestep_seconds):
        raise ValueError("Forecaster and scenario contracts disagree")
    reference = load_reference(inputs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or (args.data_root / "analysis/principal-v89-rollouts"
        / f"seed-{args.seed}" / f"{args.scenario}-schedule-{args.schedule_index}" / stamp)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, status="running", pipeline=PIPELINE, seed=args.seed,
        scenario_id=args.scenario, scenario_content_hash=scenario.content_hash,
        split=args.split, schedule_index=args.schedule_index, schedule_hash=schedule.hash,
        schedule_events=[asdict(event) for event in schedule.events],
        selection="CLI selection; the default scenario-90/schedule-8 is a selected illustration of a larger pretrained arsenic-burden reduction, with four source-injection events and two overlapping in time",
        duration_seconds=dataset.duration_seconds, timestep_seconds=dataset.hydraulic_timestep_seconds,
        policy_selection={key: {field: value[field] for field in ("selected_timesteps", "completed_timesteps")}
                          for key, value in (("baseline", inputs["baseline"]), ("pretrained", inputs["pretrained"]))},
        saved_job_contract=inputs["job"], mpc_forecast_bundles=[str(path) for path in inputs["bundles"]],
        mpc=dict(cem=settings, cem_seed=args.cem_seed, history_steps=HISTORY_STEPS,
            horizon_steps=HORIZON_STEPS, initial_zero_injection_actions=HISTORY_STEPS - 1,
            first_replanning_time_seconds=(HISTORY_STEPS - 1) * dataset.hydraulic_timestep_seconds,
            ensemble_aggregation="mean of separately scored member returns",
            terminal_planning="Keep the full 16-step forecast horizon, including future calendar positions beyond the episode end",
            execution="Replan every timestep and apply only the first action"),
        reference_source=str(inputs["reference_dir"] / inputs["case"]["file"]))
    write_json(output / "comparison.json", manifest)
    controllers = {
        "ppo_initial": inputs["baseline"]["initial"], "ppo_best": inputs["baseline"]["best"],
        "mpc": None, "bc_initial": inputs["pretrained"]["initial"],
        "finetuned_best": inputs["pretrained"]["best"],
    }
    for approach, checkpoint in controllers.items():
        metrics = run_controller(approach, checkpoint, inputs, args, dataset, schedule,
                                 backend, reference, forecast, output)
        manifest.setdefault("completed_approaches", {})[approach] = metrics["cases"][0]["values"]
        write_json(output / "comparison.json", manifest)
    plot_injections(output, schedule)
    manifest["status"] = "complete"
    write_json(output / "comparison.json", manifest)
    print(f"Five complete rollouts exported to {output}")


if __name__ == "__main__":
    main()
