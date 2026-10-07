#!/usr/bin/env python3
"""Run four V89 HPO rounds with the current V89 parameter schema.

Completed runs are skipped by the trainer. Use --round 1, 2, 3 or 4 to select
one round; the default runs all. --job selects matching names across rounds.
A changed timestep budget creates a separate run, not a continuation of 500k.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer import (V70_MAX_CHLORINE_DELTA,
                              V70_MAX_CHLORINE_INJECTION, ControlHooksSpec,
                              ControlObservationSpec, ControlPlotsActionSpec,
                              ControlTrainingBaseSpec, Experiment, OutputSpec,
                              RecurrentPPOSpec, RewardSpec)
from packages.trainer import main as trainer_main  # noqa: E402
from training.datasets.hanoi import control_full  # noqa: E402
from training.rewards.v89 import V89Observation, V89Reward  # noqa: E402

TRAINING_SEED = 42

# Specify the remaining reward parameters explicitly for this screening experiment.
REWARD_PARAMETERS = {
    "target": 0.30,
    "tracking_reference": 0.01,
    "alpha_one_threshold": 0.15,
    "alpha_zero_threshold": 0.20,
    "blend_grace_steps": 4.0,
    "blend_decay_steps": 12.0,
    "warmup_steps": 12.0,
    "blend_steps": 12.0,
    "upper_penalty_min_factor": 0.0001,
    "wasted_movement_reference": 0.01,
    "wasted_movement_weight": 0.1,
    "wasted_movement_gate_full_threshold": 0.35,
    "wasted_movement_gate_zero_threshold": 0.40,
    "action_jerk_reference": 0.05,
    "action_jerk_weight": 0.1,
}

REFERENCE_ALGORITHM = RecurrentPPOSpec(
    n_envs=4,
    n_steps=1024,
    learning_rate=1e-4,
    batch_size=512,
    ent_coef=0.0,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    target_kl=0.02,
    eval_frequency=50_000,
    checkpoint_frequency=50_000,
)

# Entries are (round, name, total training timesteps, algorithm overrides).
# Budgets count transitions across all four environments, not per environment.
ALGORITHM_VARIANTS: tuple[tuple[int, str, int, dict[str, int | float]], ...] = (
    # Round 1: single-parameter screening, 500k timesteps per candidate.
    (1, "reference", 500_000, {}),
    (1, "learning-rate-5e-5", 500_000, {"learning_rate": 5e-5}),
    (1, "learning-rate-3e-4", 500_000, {"learning_rate": 3e-4}),
    (1, "n-steps-512", 500_000, {"n_steps": 512}),
    (1, "n-steps-2048", 500_000, {"n_steps": 2048}),
    (1, "batch-size-256", 500_000, {"batch_size": 256}),
    (1, "batch-size-1024", 500_000, {"batch_size": 1024}),
    (1, "ent-coef-1e-3", 500_000, {"ent_coef": 1e-3}),
    (1, "ent-coef-1e-2", 500_000, {"ent_coef": 1e-2}),

    # Round 2: longer finalist runs and targeted parameter combinations.
    (2, "n-steps-2048", 1_000_000, {"n_steps": 2048}),
    (2, "batch-size-256", 1_000_000, {"batch_size": 256}),
    (
        2,
        "n-steps-2048_batch-size-256",
        1_000_000,
        {"n_steps": 2048, "batch_size": 256},
    ),
    # Keep a same-budget reference and give the late-improving higher learning
    # rate a full run before deciding whether its weaker arsenic control persists.
    (2, "reference", 1_000_000, {}),
    (2, "learning-rate-3e-4", 1_000_000, {"learning_rate": 3e-4}),
    # Exploratory interaction: test the higher learning rate with the rollout
    # length that performed best on arsenic burden in the first screening.
    (
        2,
        "n-steps-2048_learning-rate-3e-4",
        500_000,
        {"n_steps": 2048, "learning_rate": 3e-4},
    ),

    # Round 3: gamma/GAE screening on the selected 2048-step, 512-batch setup.
    # Reuse the completed 500k reference with gamma=0.99 and gae_lambda=0.95.
    # The five new combinations get 500k first; this is not a final ranking.
    (3, "n-steps-2048", 500_000, {"n_steps": 2048}),
    (
        3,
        "n-steps-2048_gamma-0p98_gae-0p95",
        1_000_000,
        {"n_steps": 2048, "gamma": 0.98, "gae_lambda": 0.95},
    ),
    (
        3,
        "n-steps-2048_gamma-0p98_gae-0p98",
        1_000_000,
        {"n_steps": 2048, "gamma": 0.98, "gae_lambda": 0.98},
    ),
    (
        3,
        "n-steps-2048_gamma-0p99_gae-0p98",
        1_000_000,
        {"n_steps": 2048, "gamma": 0.99, "gae_lambda": 0.98},
    ),
    (
        3,
        "n-steps-2048_gamma-0p995_gae-0p95",
        1_000_000,
        {"n_steps": 2048, "gamma": 0.995, "gae_lambda": 0.95},
    ),
    (
        3,
        "n-steps-2048_gamma-0p995_gae-0p98",
        1_000_000,
        {"n_steps": 2048, "gamma": 0.995, "gae_lambda": 0.98},
    ),
)


# Round 4: vary one reward parameter at a time with the selected PPO setup.
# Entries are (name, total training timesteps, reward overrides).
# The matching 1M reference is round 2's completed n-steps-2048 run.
REWARD_SEARCH_ALGORITHM = replace(REFERENCE_ALGORITHM, n_steps=2048)
REWARD_VARIANTS: tuple[tuple[str, int, dict[str, float]], ...] = (
    ("n-steps-2048_decay-6", 1_000_000, {"blend_decay_steps": 6.0}),
    ("n-steps-2048_grace-8", 1_000_000, {"blend_grace_steps": 8.0}),
    (
        "n-steps-2048_upper-factor-1e-3",
        1_000_000,
        {"upper_penalty_min_factor": 1e-3},
    ),
    (
        "n-steps-2048_upper-factor-1e-5",
        1_000_000,
        {"upper_penalty_min_factor": 1e-5},
    ),
)


def build_experiment(round_number: int = 4) -> Experiment:
    base = ControlTrainingBaseSpec(
        dataset=control_full,
        observation=ControlObservationSpec(V89Observation, REWARD_PARAMETERS),
        reward=RewardSpec(V89Reward, REWARD_PARAMETERS),
        training_seed=TRAINING_SEED,
        initial_injection=0.0,
        max_injection=V70_MAX_CHLORINE_INJECTION,
        max_delta=V70_MAX_CHLORINE_DELTA,
        # Keep validation metrics at every checkpoint, but render plots only
        # at the end. The held-out test split is not used for HPO.
        hooks=ControlHooksSpec(
            on_finish=(ControlPlotsActionSpec(("validation",)),),
        ),
        evaluate_test=False,
        output=OutputSpec(group="control"),
    )
    jobs = tuple(
        base.build(
            name=name,
            total_timesteps=total_timesteps,
            algorithm=replace(REFERENCE_ALGORITHM, **overrides),
        )
        for variant_round, name, total_timesteps, overrides in ALGORITHM_VARIANTS
        if variant_round == round_number
    )
    if round_number == 4:
        # Observation and reward share the Alpha state contract, so apply each
        # override to both factories, including their Grace/Decay parameters.
        jobs = tuple(
            base.build(
                name=name,
                total_timesteps=total_timesteps,
                algorithm=REWARD_SEARCH_ALGORITHM,
                reward=RewardSpec(V89Reward, {**REWARD_PARAMETERS, **overrides}),
                observation=ControlObservationSpec(
                    V89Observation, {**REWARD_PARAMETERS, **overrides}
                ),
            )
            for name, total_timesteps, overrides in REWARD_VARIANTS
        )
    return Experiment(
        "hanoi-control-v89-hpo",
        jobs,
        f"V89 HPO round {round_number} on control_full with seed 42. "
        "Rounds 1-3 vary PPO parameters; round 4 varies reward parameters. "
        "Completed runs are skipped. Script options: --round {1,2,3,4} "
        "(default: all), --job NAME (repeatable).",
    )


def main(argv: list[str] | None = None) -> int:
    # Separate experiments permit repeated job names across rounds.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--round", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--job", action="append")
    args, trainer_args = parser.parse_known_args(argv)
    rounds = (args.round,) if args.round is not None else (1, 2, 3, 4)
    experiments = [build_experiment(number) for number in rounds]
    if args.job:
        selected = set(args.job)
        known = {job.name for experiment in experiments for job in experiment.jobs}
        if unknown := selected - known:
            parser.error(f"unknown selected jobs: {sorted(unknown)}")
        experiments = [
            replace(experiment, jobs=jobs)
            for experiment in experiments
            if (jobs := tuple(job for job in experiment.jobs if job.name in selected))
        ]
    if not trainer_args or any(arg in {"-h", "--help", "help"} for arg in trainer_args):
        return trainer_main(experiments[0], trainer_args)
    if trainer_args == ["describe", "--json"]:
        print(json.dumps({"rounds": [
            {"experiment": experiment.to_dict(),
             "selected_jobs": [job.name for job in experiment.jobs]}
            for experiment in experiments
        ]}, indent=2, sort_keys=True))
        return 0
    for experiment in experiments:
        result = trainer_main(experiment, trainer_args)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
