"""Trainable policy-side feature adapters for recurrent control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from ..forecasting.bundle import load_bundle_manifest
from ..forecasting.model import LatentDynamicsForecaster


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ProjectedLatentFeatureExtractor(BaseFeaturesExtractor):
    """Replace a trailing raw latent with a gated, projected representation."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        base_size: int,
        latent_size: int,
        projected_size: int,
        initial_gate_logit: float = -4.0,
    ):
        observation_size = int(np.prod(observation_space.shape))
        if min(base_size, latent_size, projected_size) <= 0:
            raise ValueError("latent adapter dimensions must be positive")
        if observation_size != base_size + latent_size:
            raise ValueError(
                "latent adapter dimensions do not match the observation space"
            )
        super().__init__(observation_space, features_dim=base_size + projected_size)
        self.base_size = base_size
        self.latent_size = latent_size
        self.projected_size = projected_size
        self.latent_norm = nn.LayerNorm(latent_size)
        self.projection = nn.Linear(latent_size, projected_size)
        self.gate_logit = nn.Parameter(
            torch.tensor(float(initial_gate_logit), dtype=torch.float32)
        )

    @property
    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = observations[..., : self.base_size]
        latent = observations[
            ..., self.base_size : self.base_size + self.latent_size
        ]
        projected = torch.tanh(self.projection(self.latent_norm(latent)))
        return torch.cat((base, self.gate_value * projected), dim=-1)


class UngatedProjectedLatentFeatureExtractor(BaseFeaturesExtractor):
    """Replace a trailing raw latent with a normalized linear projection."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        base_size: int,
        latent_size: int,
        projected_size: int,
    ):
        observation_size = int(np.prod(observation_space.shape))
        if min(base_size, latent_size, projected_size) <= 0:
            raise ValueError("latent adapter dimensions must be positive")
        if observation_size != base_size + latent_size:
            raise ValueError(
                "latent adapter dimensions do not match the observation space"
            )
        super().__init__(observation_space, features_dim=base_size + projected_size)
        self.base_size = base_size
        self.latent_size = latent_size
        self.projected_size = projected_size
        self.projection = nn.Linear(latent_size, projected_size)
        self.projected_norm = nn.LayerNorm(projected_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = observations[..., : self.base_size]
        latent = observations[
            ..., self.base_size : self.base_size + self.latent_size
        ]
        projected = torch.tanh(self.projected_norm(self.projection(latent)))
        return torch.cat((base, projected), dim=-1)


class TrainableNextLatentFeatureExtractor(BaseFeaturesExtractor):
    """Encode a causal history and project it inside the recurrent PPO policy."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        base_size: int,
        history_steps: int,
        history_feature_size: int,
        latent_size: int,
        projected_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        observation_size = int(np.prod(observation_space.shape))
        history_size = history_steps * history_feature_size
        if min(
            base_size,
            history_steps,
            history_feature_size,
            latent_size,
            projected_size,
            num_layers,
        ) <= 0:
            raise ValueError("trainable next-latent dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("trainable next-latent dropout must be in [0, 1)")
        if observation_size != base_size + history_size:
            raise ValueError(
                "trainable next-latent dimensions do not match the observation space"
            )
        super().__init__(observation_space, features_dim=base_size + projected_size)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.base_size = base_size
        self.history_steps = history_steps
        self.history_feature_size = history_feature_size
        self.history_size = history_size
        self.latent_size = latent_size
        self.projected_size = projected_size
        self.encoder = nn.LSTM(
            history_feature_size,
            latent_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.latent_norm = nn.LayerNorm(latent_size)
        self.projection = nn.Linear(latent_size, projected_size)
        self.projected_norm = nn.LayerNorm(projected_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = observations[..., : self.base_size]
        history = observations[..., self.base_size :].reshape(
            -1, self.history_steps, self.history_feature_size
        )
        _encoded, (hidden, _cell) = self.encoder(history)
        latent = self.latent_norm(hidden[-1])
        projected = torch.tanh(self.projected_norm(self.projection(latent)))
        return torch.cat((base, projected), dim=-1)


class FixedPCALatentFeatureExtractor(BaseFeaturesExtractor):
    """Append a fixed PCA-whitened view of a trailing frozen latent.

    The vector environment normalizes only ``base``.  The latent suffix stays
    in the encoder's native coordinates and is centered and whitened here with
    statistics fitted once on training trajectories.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        base_size: int,
        latent_size: int,
        projected_size: int,
        projection: str,
        latent_bundle: str,
        whitening_epsilon: float = 1e-6,
        clip_projected: float = 10.0,
    ):
        observation_size = int(np.prod(observation_space.shape))
        if min(base_size, latent_size, projected_size) <= 0:
            raise ValueError("fixed PCA dimensions must be positive")
        if observation_size != base_size + latent_size:
            raise ValueError("fixed PCA dimensions do not match observation space")
        if projected_size > latent_size:
            raise ValueError("fixed PCA projected size exceeds latent size")
        if whitening_epsilon <= 0 or clip_projected <= 0:
            raise ValueError("fixed PCA numerical parameters must be positive")

        projection_path = Path(projection)
        if not projection_path.is_absolute():
            projection_path = PROJECT_ROOT / projection_path
        manifest_path = projection_path / "manifest.json"
        arrays_path = projection_path / "projection.npz"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                f"fixed PCA manifest not found: {manifest_path}"
            ) from exc
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "fixed-latent-pca"
        ):
            raise ValueError(f"invalid fixed PCA manifest: {manifest_path}")
        latent_bundle_path = Path(latent_bundle)
        if not latent_bundle_path.is_absolute():
            latent_bundle_path = PROJECT_ROOT / latent_bundle_path
        source_model = latent_bundle_path / "model.pt"
        try:
            source_model_sha256 = hashlib.sha256(source_model.read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            raise ValueError(
                f"fixed PCA source encoder not found: {source_model}"
            ) from exc
        if manifest.get("source_model_sha256") != source_model_sha256:
            raise ValueError(
                "fixed PCA artifact was fitted with a different encoder checkpoint"
            )
        try:
            with np.load(arrays_path, allow_pickle=False) as raw:
                mean = np.asarray(raw["mean"], dtype=np.float32)
                components = np.asarray(raw["components"], dtype=np.float32)
                explained_variance = np.asarray(
                    raw["explained_variance"], dtype=np.float32
                )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"invalid fixed PCA arrays: {arrays_path}") from exc
        expected_components = int(manifest.get("component_count", -1))
        if (
            int(manifest.get("latent_size", -1)) != latent_size
            or mean.shape != (latent_size,)
            or components.shape != (expected_components, latent_size)
            or explained_variance.shape != (expected_components,)
            or projected_size > expected_components
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(components))
            or not np.all(np.isfinite(explained_variance))
            or np.any(explained_variance[:projected_size] <= 0)
        ):
            raise ValueError("fixed PCA artifact dimensions or values are incompatible")

        super().__init__(
            observation_space, features_dim=base_size + projected_size
        )
        self.base_size = base_size
        self.latent_size = latent_size
        self.projected_size = projected_size
        self.whitening_epsilon = float(whitening_epsilon)
        self.clip_projected = float(clip_projected)
        self.register_buffer("latent_mean", torch.from_numpy(mean))
        self.register_buffer(
            "pca_components", torch.from_numpy(components[:projected_size].copy())
        )
        self.register_buffer(
            "whitening_scale",
            torch.sqrt(
                torch.from_numpy(explained_variance[:projected_size].copy())
                + self.whitening_epsilon
            ),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = observations[..., : self.base_size]
        latent = observations[
            ..., self.base_size : self.base_size + self.latent_size
        ]
        centered = latent - self.latent_mean
        projected = torch.nn.functional.linear(centered, self.pca_components)
        whitened = torch.clamp(
            projected / self.whitening_scale,
            -self.clip_projected,
            self.clip_projected,
        )
        return torch.cat((base, whitened), dim=-1)


class SharedDynamicsEncoderFeatureExtractor(BaseFeaturesExtractor):
    """Fine-tunable pretrained encoder shared by actor, critic, and dynamics loss."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        base_size: int,
        history_steps: int,
        history_feature_size: int,
        latent_size: int,
        bundle: str,
    ):
        bundle_path = Path(bundle)
        info = load_bundle_manifest(bundle_path)
        if info.get("kind") != "latent-dynamics-forecast":
            raise ValueError("shared dynamics encoder requires a latent-dynamics bundle")
        latent = info["latent_contract"]
        model_kwargs = info["model_kwargs"]
        bundle_latent_size = int(latent["size"])
        observation_size = int(np.prod(observation_space.shape))
        history_size = history_steps * history_feature_size
        if observation_size != base_size + history_size:
            raise ValueError(
                "shared dynamics encoder dimensions do not match the observation space"
            )
        if (
            int(latent["history_steps"]) != history_steps
            or int(model_kwargs["input_size"]) != history_feature_size
            or bundle_latent_size != latent_size
        ):
            raise ValueError("shared dynamics encoder dimensions disagree with its bundle")

        history_contract = tuple(info["history_feature_contract"]["names"])
        future_contract = tuple(info["future_known_feature_contract"]["names"])
        try:
            future_history_indices = tuple(
                history_contract.index(name) for name in future_contract
            )
        except ValueError as exc:
            raise ValueError(
                "future-known dynamics features are absent from the history contract"
            ) from exc
        chlorine_indices = tuple(
            index
            for index, name in enumerate(history_contract)
            if name.startswith("chlorine:")
        )
        if len(chlorine_indices) != int(model_kwargs["output_size"]):
            raise ValueError("dynamics chlorine contract is inconsistent")

        super().__init__(
            observation_space, features_dim=base_size + bundle_latent_size
        )
        self.base_size = base_size
        self.history_steps = history_steps
        self.history_feature_size = history_feature_size
        self.history_size = history_size
        self.latent_size = bundle_latent_size
        self.dynamics = LatentDynamicsForecaster(**model_kwargs)
        self.dynamics.load_state_dict(
            torch.load(bundle_path / "model.pt", map_location="cpu", weights_only=True)
        )

        normalization = json.loads(
            (bundle_path / "normalization.json").read_text(encoding="utf-8")
        )
        target = normalization["targets"]["chlorine"]
        self.register_buffer(
            "history_mean",
            torch.as_tensor(normalization["history_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "history_std",
            torch.as_tensor(normalization["history_std"], dtype=torch.float32),
        )
        self.register_buffer(
            "future_mean",
            torch.as_tensor(normalization["future_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "future_std",
            torch.as_tensor(normalization["future_std"], dtype=torch.float32),
        )
        self.register_buffer(
            "target_mean",
            torch.as_tensor(target["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "target_std",
            torch.as_tensor(target["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "future_history_indices",
            torch.as_tensor(future_history_indices, dtype=torch.long),
        )
        self.register_buffer(
            "chlorine_indices",
            torch.as_tensor(chlorine_indices, dtype=torch.long),
        )

    def _history(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[..., self.base_size :].reshape(
            -1, self.history_steps, self.history_feature_size
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = observations[..., : self.base_size]
        latent = self.dynamics.encode(self._history(observations))
        return torch.cat((base, latent), dim=-1)

    def auxiliary_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        *,
        chlorine_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One-step version of the pretrained chlorine/latent objective."""
        history = self._history(observations)
        next_history = self._history(next_observations)
        state = self.dynamics.encode(history)
        with torch.no_grad():
            target_state = self.dynamics.encode(next_history)

        next_last_normalized = next_history[:, -1]
        next_last_physical = (
            next_last_normalized * self.history_std + self.history_mean
        )
        future_physical = next_last_physical.index_select(
            -1, self.future_history_indices
        )
        future = (future_physical - self.future_mean) / self.future_std
        predicted_state = state + self.dynamics.transition(
            torch.cat((state, future), dim=-1)
        )
        predicted_delta = self.dynamics.chlorine_head(predicted_state)

        current_last_physical = history[:, -1] * self.history_std + self.history_mean
        current_chlorine = current_last_physical.index_select(
            -1, self.chlorine_indices
        )
        next_chlorine = next_last_physical.index_select(
            -1, self.chlorine_indices
        )
        target_delta = (
            next_chlorine - current_chlorine - self.target_mean
        ) / self.target_std

        chlorine_loss = torch.nn.functional.mse_loss(
            predicted_delta, target_delta
        )
        latent_loss = torch.nn.functional.mse_loss(
            predicted_state, target_state
        )
        combined = (
            chlorine_weight * chlorine_loss
            + (1.0 - chlorine_weight) * latent_loss
        )
        return combined, {
            "chlorine_loss": chlorine_loss,
            "latent_loss": latent_loss,
        }
