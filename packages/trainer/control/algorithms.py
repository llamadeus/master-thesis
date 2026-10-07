"""Control algorithms that extend the upstream PPO implementation."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO

from .features import SharedDynamicsEncoderFeatureExtractor


class DynamicsAuxiliaryPPO(PPO):
    """PPO followed by supervised updates of its shared dynamics encoder."""

    auxiliary_loss_coefficient: float = 0.1
    auxiliary_epochs: int = 1
    auxiliary_batch_size: int = 512
    auxiliary_chlorine_weight: float = 0.5

    def configure_auxiliary_dynamics(
        self,
        *,
        coefficient: float,
        epochs: int,
        batch_size: int,
        chlorine_weight: float,
    ) -> None:
        self.auxiliary_loss_coefficient = float(coefficient)
        self.auxiliary_epochs = int(epochs)
        self.auxiliary_batch_size = int(batch_size)
        self.auxiliary_chlorine_weight = float(chlorine_weight)

    def train(self) -> None:
        current, following = self._chronological_auxiliary_pairs()
        super().train()
        if len(current) == 0:
            return

        extractor = self.policy.features_extractor
        if not isinstance(extractor, SharedDynamicsEncoderFeatureExtractor):
            raise TypeError(
                "dynamics-auxiliary PPO requires a shared dynamics encoder"
            )
        device = self.device
        observations = torch.as_tensor(current, device=device)
        next_observations = torch.as_tensor(following, device=device)
        losses: list[float] = []
        chlorine_losses: list[float] = []
        latent_losses: list[float] = []

        for _epoch in range(self.auxiliary_epochs):
            permutation = torch.randperm(len(observations), device=device)
            for start in range(0, len(observations), self.auxiliary_batch_size):
                indexes = permutation[start : start + self.auxiliary_batch_size]
                auxiliary, components = extractor.auxiliary_loss(
                    observations[indexes],
                    next_observations[indexes],
                    chlorine_weight=self.auxiliary_chlorine_weight,
                )
                weighted = self.auxiliary_loss_coefficient * auxiliary
                self.policy.optimizer.zero_grad()
                weighted.backward()
                torch.nn.utils.clip_grad_norm_(
                    extractor.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
                losses.append(float(auxiliary.detach()))
                chlorine_losses.append(
                    float(components["chlorine_loss"].detach())
                )
                latent_losses.append(float(components["latent_loss"].detach()))

        self.logger.record("train/auxiliary_dynamics_loss", np.mean(losses))
        self.logger.record(
            "train/auxiliary_chlorine_loss", np.mean(chlorine_losses)
        )
        self.logger.record("train/auxiliary_latent_loss", np.mean(latent_losses))

    def _chronological_auxiliary_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        observations = self.rollout_buffer.observations
        episode_starts = self.rollout_buffer.episode_starts
        if not isinstance(observations, np.ndarray) or observations.ndim != 3:
            raise TypeError(
                "dynamics auxiliary training requires flat vector observations"
            )
        valid = episode_starts[1:] < 0.5
        return observations[:-1][valid].copy(), observations[1:][valid].copy()
