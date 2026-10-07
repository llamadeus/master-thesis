"""Observation-normalization variants for control policies."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

class _PrefixRunningMeanStd(RunningMeanStd):
    """Track a prefix while exposing identity statistics for the suffix."""

    def __init__(self, shape: tuple[int, ...], prefix_size: int):
        if len(shape) != 1 or not 0 < prefix_size < shape[0]:
            raise ValueError("prefix normalization requires a proper 1-D prefix")
        super().__init__(shape=shape)
        self.prefix_size = int(prefix_size)

    def update(self, arr: np.ndarray) -> None:
        values = np.asarray(arr)
        if values.ndim < 2 or values.shape[-1] != self.mean.shape[0]:
            raise ValueError("observation shape disagrees with prefix normalization")
        prefix = values[..., : self.prefix_size].reshape(-1, self.prefix_size)
        batch_mean = np.mean(prefix, axis=0)
        batch_var = np.var(prefix, axis=0)
        batch_count = len(prefix)
        total_count = self.count + batch_count
        old_mean = self.mean[: self.prefix_size]
        old_var = self.var[: self.prefix_size]
        delta = batch_mean - old_mean
        mean = old_mean + delta * batch_count / total_count
        old_sum = old_var * self.count
        batch_sum = batch_var * batch_count
        combined_sum = (
            old_sum
            + batch_sum
            + np.square(delta) * self.count * batch_count / total_count
        )
        var = combined_sum / total_count
        count = total_count
        self.mean[: self.prefix_size] = mean
        self.var[: self.prefix_size] = var
        self.mean[self.prefix_size :] = 0.0
        self.var[self.prefix_size :] = 1.0
        self.count = count


class BaseOnlyVecNormalize(VecNormalize):
    """Normalize a base-observation prefix and leave a latent suffix untouched."""

    def __init__(
        self,
        venv: VecEnv,
        *,
        base_size: int,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            venv,
            training=training,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            epsilon=epsilon,
        )
        self.base_size = int(base_size)
        if norm_obs:
            self.obs_rms = _PrefixRunningMeanStd(
                tuple(self.observation_space.shape), self.base_size
            )
