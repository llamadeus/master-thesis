from __future__ import annotations

import copy

import torch
from torch import nn


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size: int, decoder_input_size: int, hidden_size: int, num_layers: int,
                 head_sizes: dict[str, int] | tuple[tuple[str, int], ...],
                 decoder_mode: str, dropout: float = 0.0, horizon_steps: int = 1):
        super().__init__()
        if input_size <= 0 or decoder_input_size < 0 or hidden_size <= 0 or num_layers <= 0 or horizon_steps <= 0:
            raise ValueError("forecast model dimensions must be positive (decoder input may be zero)")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=recurrent_dropout)
        configured = dict(head_sizes)
        if len(configured) != len(tuple(head_sizes.items()) if isinstance(head_sizes, dict) else tuple(head_sizes)):
            raise ValueError("forecast head names must be unique")
        if not configured or any(not name or size <= 0 for name, size in configured.items()):
            raise ValueError("forecast heads require names and positive output sizes")
        if decoder_mode not in ("shared", "separate-per-head"):
            raise ValueError("unsupported decoder sharing mode")
        self.decoder_mode = decoder_mode
        if decoder_mode == "shared":
            self.decoder = nn.LSTM(decoder_input_size, hidden_size, num_layers, batch_first=True, dropout=recurrent_dropout)
            self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, size) for name, size in configured.items()})
        else:
            self.decoders = nn.ModuleDict({name: nn.LSTM(decoder_input_size, hidden_size, num_layers,
                                          batch_first=True, dropout=recurrent_dropout) for name in configured})
            self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, size) for name, size in configured.items()})
        self.horizon_steps = horizon_steps

    def forward(self, history: torch.Tensor, future_known: torch.Tensor):
        if history.ndim != 3 or future_known.ndim != 3 or history.shape[0] != future_known.shape[0]:
            raise ValueError("forecast inputs must be batch-first rank-three tensors")
        if future_known.shape[1] != self.horizon_steps:
            raise ValueError("future-known horizon does not match model construction")
        _encoded, state = self.encoder(history)
        if self.decoder_mode == "shared":
            decoded, _state = self.decoder(future_known, state)
            return {name: head(decoded) for name, head in self.heads.items()}
        return {name: self.heads[name](self.decoders[name](future_known, state)[0]) for name in self.heads}


class LatentDynamicsForecaster(nn.Module):
    """Causal history encoder with action-conditioned latent rollouts."""

    def __init__(self, input_size: int, transition_input_size: int, hidden_size: int,
                 transition_hidden_size: int, output_size: int, horizon_steps: int):
        super().__init__()
        if min(input_size, hidden_size, transition_hidden_size, output_size, horizon_steps) <= 0:
            raise ValueError("latent-dynamics model dimensions must be positive")
        if transition_input_size < 0:
            raise ValueError("latent-dynamics transition input size must be non-negative")
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.transition = nn.Sequential(
            nn.Linear(hidden_size + transition_input_size, transition_hidden_size),
            nn.ReLU(),
            nn.Linear(transition_hidden_size, hidden_size),
        )
        self.chlorine_head = nn.Linear(hidden_size, output_size)
        self.horizon_steps = horizon_steps
        self.hidden_size = hidden_size

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("latent-dynamics history must be a batch-first rank-three tensor")
        _encoded, (hidden, _cell) = self.encoder(history)
        return hidden[-1]

    def rollout(self, history: torch.Tensor, future_known: torch.Tensor):
        if future_known.ndim != 3 or history.shape[0] != future_known.shape[0]:
            raise ValueError("latent-dynamics inputs must be batch-first rank-three tensors")
        if future_known.shape[1] != self.horizon_steps:
            raise ValueError("future-known horizon does not match model construction")
        state = self.encode(history)
        states, deltas = [], []
        for step in range(self.horizon_steps):
            residual = self.transition(torch.cat((state, future_known[:, step]), dim=-1))
            state = state + residual
            states.append(state)
            deltas.append(self.chlorine_head(state))
        return {"chlorine": torch.stack(deltas, dim=1)}, torch.stack(states, dim=1)

    def forward(self, history: torch.Tensor, future_known: torch.Tensor):
        predictions, _states = self.rollout(history, future_known)
        return predictions


class NextLatentLSTMEncoder(nn.Module):
    """Encode a causal history and optionally forecast chlorine from it."""

    def __init__(self, input_size: int, latent_size: int,
                 predictor_hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, ema_decay: float = 0.99,
                 decoder_input_size: int = 0, output_size: int = 0,
                 horizon_steps: int = 1):
        super().__init__()
        if min(input_size, latent_size, predictor_hidden_size, num_layers) <= 0:
            raise ValueError("next-latent model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("next-latent dropout must be in [0, 1)")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("next-latent EMA decay must be in [0, 1)")
        if decoder_input_size < 0 or output_size < 0 or horizon_steps <= 0:
            raise ValueError("next-latent decoder dimensions are invalid")
        if (decoder_input_size == 0) != (output_size == 0):
            raise ValueError(
                "next-latent chlorine decoder input and output must be configured together"
            )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size,
            latent_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.latent_norm = nn.LayerNorm(latent_size)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_latent_norm = copy.deepcopy(self.latent_norm)
        self.target_encoder.requires_grad_(False)
        self.target_latent_norm.requires_grad_(False)
        self.target_encoder.eval()
        self.target_latent_norm.eval()
        self.predictor = nn.Sequential(
            nn.Linear(latent_size, predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(predictor_hidden_size, latent_size),
        )
        if output_size:
            self.chlorine_decoder = nn.LSTM(
                decoder_input_size,
                latent_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.chlorine_head = nn.Linear(latent_size, output_size)
        self.latent_size = latent_size
        self.ema_decay = ema_decay
        self.horizon_steps = horizon_steps
        self.has_chlorine_head = bool(output_size)

    def _encode_state(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if history.ndim != 3:
            raise ValueError("next-latent history must be a batch-first rank-three tensor")
        _encoded, state = self.encoder(history)
        return self.latent_norm(state[0][-1]), state

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        latent, _state = self._encode_state(history)
        return latent

    @torch.no_grad()
    def encode_target(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("next-latent target history must be a batch-first rank-three tensor")
        _encoded, (hidden, _cell) = self.target_encoder(history)
        return self.target_latent_norm(hidden[-1])

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        online_modules = (self.encoder, self.latent_norm)
        target_modules = (self.target_encoder, self.target_latent_norm)
        for online, target in zip(online_modules, target_modules, strict=True):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(self.ema_decay).add_(
                    online_parameter, alpha=1.0 - self.ema_decay
                )
            for online_buffer, target_buffer in zip(
                online.buffers(), target.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_latent_norm.eval()
        return self

    def predict_next(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(history)
        return latent, latent + self.predictor(latent)

    def predict_next_and_forecast(
        self, history: torch.Tensor, future_known: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self.has_chlorine_head:
            raise ValueError("next-latent model has no chlorine forecast head")
        if (
            future_known.ndim != 3
            or history.shape[0] != future_known.shape[0]
            or future_known.shape[1] != self.horizon_steps
        ):
            raise ValueError("next-latent forecast inputs have an incompatible horizon")
        latent, state = self._encode_state(history)
        predicted = latent + self.predictor(latent)
        decoded, _decoder_state = self.chlorine_decoder(future_known, state)
        return latent, predicted, {"chlorine": self.chlorine_head(decoded)}

    def forecast(
        self, history: torch.Tensor, future_known: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.predict_next_and_forecast(history, future_known)[2]

    def forward(
        self, history: torch.Tensor, future_known: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.has_chlorine_head:
            if future_known is None:
                raise ValueError("chlorine forecasting requires future-known features")
            return self.forecast(history, future_known)
        if future_known is not None:
            raise ValueError("headless next-latent encoding does not use future-known features")
        return self.predict_next(history)[1]
