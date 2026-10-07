#!/usr/bin/env python3
"""Recurrent-PPO v89 baseline on control-full."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer import (ControlHooksSpec,
                              ControlObservationSpec, ControlPlotsActionSpec,
                              ControlTrainingBaseSpec, Experiment, OutputSpec,
                              RecurrentPPOSpec, RewardSpec, main)
from packages.trainer.constants import MAX_CHLORINE_INJECTION
from training.datasets.hanoi import control_full
from training.rewards.v89 import DEFAULT_PARAMETERS, V89Observation, V89Reward

REWARD_VARIANTS: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "baseline",
        {},
    ),
)
MAX_CHLORINE_DELTA = 50_000.0


def build_experiment() -> Experiment:
    base = ControlTrainingBaseSpec(
        dataset=control_full,
        observation=ControlObservationSpec(V89Observation),
        algorithm=RecurrentPPOSpec(
            n_envs=4,
            n_steps=2048,
            learning_rate=1e-4,
            batch_size=512,
            ent_coef=0.0,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.02,
        ),
        training_seed=42,
        total_timesteps=1_000_000,
        initial_injection=0.0,
        max_injection=MAX_CHLORINE_INJECTION,
        max_delta=MAX_CHLORINE_DELTA,
        reward=RewardSpec(V89Reward, DEFAULT_PARAMETERS),
        hooks=ControlHooksSpec(
            on_checkpoint=(ControlPlotsActionSpec(("validation",)),),
            on_finish=(ControlPlotsActionSpec(("validation", "test")),),
        ),
        output=OutputSpec(group="control"),
    )

    jobs = tuple(
        base.build(
            name=name,
            reward=RewardSpec(
                V89Reward,
                {**DEFAULT_PARAMETERS, **reward_overrides},
            ),
            observation=ControlObservationSpec(
                V89Observation,
                {**DEFAULT_PARAMETERS, **reward_overrides},
            ),
        )
        for name, reward_overrides in REWARD_VARIANTS
    )
    return Experiment(
        "v89",
        jobs,
        "Train the recurrent-PPO v89 baseline on control-full.",
    )


if __name__ == "__main__":
    raise SystemExit(main(build_experiment()))
