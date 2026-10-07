"""Fixed-contract forecast providers used by assisted control and ablations."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..forecasting.bundle import load_bundle_manifest
from ..forecasting.features import inverse_transform
from ..forecasting.model import EncoderDecoderLSTM
from ..forecasting.model import LatentDynamicsForecaster
from ..forecasting.model import NextLatentLSTMEncoder
from ..simulation import SimulationSample
from ..time import cyclic_time_features
from ..specs import (
    ArtifactRef,
    ForecastProviderSpec, NoiseForecastProviderSpec, PersistenceForecastProviderSpec,
    PretrainedForecastProviderSpec, RandomFrozenForecastProviderSpec, ShuffledForecastProviderSpec,
    PretrainedLatentProviderSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CausalHistoryBuffer:
    """A bounded history containing only causal, observed feature rows."""

    def __init__(self, history_steps: int, feature_size: int, *, label: str):
        if history_steps < 1 or feature_size < 1:
            raise ValueError("causal history dimensions must be positive")
        self.history_steps = int(history_steps)
        self.feature_size = int(feature_size)
        self.label = label
        self.rows: deque[np.ndarray] = deque(maxlen=self.history_steps)

    @property
    def ready(self) -> bool:
        return len(self.rows) == self.history_steps

    def reset(self) -> None:
        self.rows.clear()

    def observe(self, row) -> None:
        value = np.asarray(row, dtype=np.float32)
        if value.shape != (self.feature_size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{self.label} history row is incompatible")
        self.rows.append(value.copy())

    def values(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError(
                f"{self.label} requires {self.history_steps} observed history "
                f"samples; {len(self.rows)} are available"
            )
        return np.asarray(self.rows, dtype=np.float32)

    def state_dict(self) -> dict[str, object]:
        return {"rows": [row.tolist() for row in self.rows]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        rows = np.asarray(state.get("rows", ()), dtype=np.float32)
        if rows.size == 0:
            rows = np.empty((0, self.feature_size), dtype=np.float32)
        if (
            rows.ndim != 2
            or rows.shape[1] != self.feature_size
            or rows.shape[0] > self.history_steps
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError(f"{self.label} history state is incompatible")
        self.rows = deque((row.copy() for row in rows), maxlen=self.history_steps)


class ForecastProvider:
    def __init__(self, horizon: int, node_ids: tuple[str, ...]):
        self.output_shape = (horizon, len(node_ids)); self.node_ids = node_ids
    @property
    def required_history_samples(self) -> int: return 1
    @property
    def ready(self) -> bool: return True
    def reset(self) -> None: pass
    def observe(self, sample: SimulationSample) -> None: pass
    def predict(self, sample: SimulationSample) -> np.ndarray: raise NotImplementedError
    def state_dict(self): return {}
    def load_state_dict(self,state): pass


class PersistenceProvider(ForecastProvider):
    def predict(self, sample): return np.repeat(np.asarray(sample.chlorine, dtype=np.float32)[None, :], self.output_shape[0], axis=0)


class NoiseProvider(ForecastProvider):
    def __init__(self, spec, horizon, node_ids):
        super().__init__(horizon, node_ids); self.spec = spec; self.rng = np.random.default_rng(spec.seed)
    def predict(self, sample):
        shape = self.output_shape
        values = self.rng.normal(0, self.spec.standard_deviation, shape) if self.spec.distribution == "normal" else self.rng.uniform(-self.spec.standard_deviation, self.spec.standard_deviation, shape)
        return values.astype(np.float32)
    def state_dict(self): return {"bit_generator_state":self.rng.bit_generator.state}
    def load_state_dict(self,state): self.rng.bit_generator.state=state["bit_generator_state"]


class BundleProvider(ForecastProvider):
    def __init__(self, bundle: Path, horizon: int, node_ids: tuple[str, ...]):
        info = load_bundle_manifest(bundle)
        if info.get("kind") != "encoder-decoder-lstm": raise ValueError("invalid forecast bundle")
        observation = info["job"]["observation"]
        allowed = {"chlorine", "chlorine-injection", "cyclic-time"}
        if any(feature["name"] not in allowed for feature in (*observation["history"], *observation["future_known"])):
            raise ValueError("forecast bundle contains non-deployable feature providers")
        if tuple(info["node_ids"]) != node_ids or info["model_kwargs"]["horizon_steps"] != horizon:
            raise ValueError("forecast bundle node order or horizon mismatch")
        super().__init__(horizon, node_ids)
        self.info = info; self.device = torch.device("cpu")
        self.head = next((head for head in info["heads"] if head["name"] == "chlorine"), None)
        if self.head is None: raise ValueError("forecast bundle has no deployable chlorine head")
        self.timestep_seconds=int(info["hydraulic_timestep_seconds"])
        self.model = EncoderDecoderLSTM(**info["model_kwargs"]); self.model.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)); self.model.eval()
        raw = json.loads((bundle / "normalization.json").read_text())
        self.norm = raw
        self.history_steps = int(info["job"]["model"]["history_steps"])
        self.history_names=tuple(info["history_feature_contract"]["names"]);self.future_names=tuple(info["future_known_feature_contract"]["names"])
        self.history_transforms=tuple(info["history_feature_contract"].get("transforms",()))
        self.future_transforms=tuple(info["future_known_feature_contract"].get("transforms",()))
        self.history = CausalHistoryBuffer(
            self.history_steps, len(self.history_names), label="forecast provider"
        )
    @property
    def required_history_samples(self) -> int: return self.history_steps
    @property
    def ready(self) -> bool: return self.history.ready
    def reset(self): self.history.reset()
    def observe(self, sample):
        row = _apply_contract_transforms(
            _named_row(sample,self.history_names,sample.time_seconds),
            self.history_transforms,
        )
        self.history.observe(row)
    def state_dict(self): return self.history.state_dict()
    def load_state_dict(self,state): self.history.load_state_dict(state)
    def predict(self, sample):
        self.observe(sample)
        future = np.asarray([_named_row(sample,self.future_names,sample.time_seconds+(index+1)*self.timestep_seconds)
                             for index in range(self.output_shape[0])],dtype=np.float32)
        future = _apply_contract_transforms(future, self.future_transforms)
        history = (self.history.values() - np.asarray(self.norm["history_mean"])) / np.asarray(self.norm["history_std"])
        future = (future - np.asarray(self.norm["future_mean"])) / np.asarray(self.norm["future_std"])
        with torch.no_grad(): outputs = self.model(torch.from_numpy(history[None].astype(np.float32)), torch.from_numpy(future[None].astype(np.float32)))
        normalized = outputs["chlorine"].numpy()[0]
        target = self.norm["targets"]["chlorine"]
        represented = normalized * np.asarray(target["std"]) + np.asarray(target["mean"])
        physical = inverse_transform(represented, self.head["transform"], self.head.get("scale"))
        if self.head["prediction"] == "delta":
            physical = sample.chlorine[None, :] + np.cumsum(physical, axis=0)
        return np.maximum(0, physical).astype(np.float32)


class LatentBundleProvider:
    """Frozen causal encoder exposing the bundle's policy-facing ``z_t``."""

    def __init__(
        self,
        bundle: Path,
        node_ids: tuple[str, ...],
        *,
        load_model: bool = True,
    ):
        info = load_bundle_manifest(bundle)
        bundle_kind = info.get("kind")
        if bundle_kind not in {
            "latent-dynamics-forecast",
            "next-latent-lstm-encoder",
        }:
            raise ValueError("invalid latent encoder bundle")
        if tuple(info["node_ids"]) != node_ids:
            raise ValueError("latent encoder bundle node order mismatch")
        observation = info["job"]["observation"]
        allowed = {"chlorine", "chlorine-injection", "cyclic-time"}
        if any(
            feature["name"] not in allowed
            for feature in (*observation["history"], *observation["future_known"])
        ):
            raise ValueError(
                "latent encoder bundle contains non-deployable feature providers"
            )
        if bundle_kind == "next-latent-lstm-encoder" and observation["future_known"]:
            raise ValueError("next-latent encoder bundle must not use future-known features")

        latent = info["latent_contract"]
        self.bundle_kind = bundle_kind
        self.output_shape = (int(latent["size"]),)
        self.node_ids = node_ids
        self.history_steps = int(latent["history_steps"])
        self.history_names = tuple(info["history_feature_contract"]["names"])
        self.history_transforms = tuple(
            info["history_feature_contract"].get("transforms", ())
        )
        model_kwargs = info["model_kwargs"]
        model_latent_size = int(
            model_kwargs[
                "hidden_size"
                if bundle_kind == "latent-dynamics-forecast"
                else "latent_size"
            ]
        )
        if (
            model_kwargs["input_size"] != len(self.history_names)
            or model_latent_size != self.output_shape[0]
        ):
            raise ValueError("latent encoder bundle dimensions are inconsistent")
        if load_model:
            if bundle_kind == "latent-dynamics-forecast":
                self.model = LatentDynamicsForecaster(**model_kwargs)
            else:
                self.model = NextLatentLSTMEncoder(**model_kwargs)
            self.model.load_state_dict(
                torch.load(
                    bundle / "model.pt", map_location="cpu", weights_only=True
                )
            )
            self.model.eval()
            self.model.requires_grad_(False)
        raw = json.loads((bundle / "normalization.json").read_text())
        self.history_mean = np.asarray(raw["history_mean"], dtype=np.float32)
        self.history_std = np.asarray(raw["history_std"], dtype=np.float32)
        expected_normalization = (len(self.history_names),)
        if (
            self.history_mean.shape != expected_normalization
            or self.history_std.shape != expected_normalization
            or not np.all(np.isfinite(self.history_mean))
            or not np.all(np.isfinite(self.history_std))
            or np.any(self.history_std <= 0)
        ):
            raise ValueError("latent encoder bundle normalization is incompatible")
        self.history = CausalHistoryBuffer(
            self.history_steps, len(self.history_names), label="latent provider"
        )

    @property
    def required_history_samples(self) -> int:
        return self.history_steps

    @property
    def ready(self) -> bool:
        return self.history.ready

    def reset(self) -> None:
        self.history.reset()

    def observe(self, sample: SimulationSample) -> None:
        row = _apply_contract_transforms(
            _named_row(sample, self.history_names, sample.time_seconds),
            self.history_transforms,
        )
        self.history.observe(row)

    def predict(self, sample: SimulationSample) -> np.ndarray:
        history = self.normalized_history(sample)
        with torch.no_grad():
            latent = self.model.encode(torch.from_numpy(history[None])).numpy()[0]
        return latent.astype(np.float32)

    def normalized_history(self, sample: SimulationSample) -> np.ndarray:
        """Append one causal row and return the forecast-normalized window."""
        self.observe(sample)
        history = (
            self.history.values() - self.history_mean
        ) / self.history_std
        return history.astype(np.float32)

    def state_dict(self):
        return self.history.state_dict()

    def load_state_dict(self, state):
        self.history.load_state_dict(state)


class LatentHistoryBundleProvider(LatentBundleProvider):
    """Expose the normalized causal window for a policy-owned encoder."""

    def __init__(self, bundle: Path, node_ids: tuple[str, ...]):
        super().__init__(bundle, node_ids, load_model=False)
        self.output_shape = (self.history_steps * len(self.history_names),)

    def predict(self, sample: SimulationSample) -> np.ndarray:
        return self.normalized_history(sample).reshape(-1)


class ShuffledProvider(BundleProvider):
    def __init__(self, bundle, horizon, node_ids, seed):
        super().__init__(bundle, horizon, node_ids); self.rng = np.random.default_rng(seed); self.permutation = self.rng.permutation(np.prod(self.output_shape))
    def predict(self, sample): return super().predict(sample).reshape(-1)[self.permutation].reshape(self.output_shape)


class RandomFrozenProvider(ForecastProvider):
    def __init__(self, spec, horizon, node_ids,timestep_seconds):
        super().__init__(horizon, node_ids); torch.manual_seed(spec.seed); model_spec = spec.model
        if model_spec.horizon_steps != horizon: raise ValueError("random-frozen horizon mismatch")
        if "chlorine" not in {head.name for head in model_spec.heads}: raise ValueError("random-frozen model requires a chlorine head")
        self.history_steps = model_spec.history_steps; self.timestep_seconds=timestep_seconds
        self.model = EncoderDecoderLSTM(len(node_ids) + 7, 7, model_spec.hidden_size, model_spec.num_layers,
            tuple((head.name, len(node_ids)) for head in model_spec.heads), model_spec.decoder_mode,
            model_spec.dropout, horizon); self.model.eval()
        self.history = CausalHistoryBuffer(
            self.history_steps, len(self.node_ids) + 7, label="random-frozen provider"
        )
    @property
    def required_history_samples(self) -> int: return self.history_steps
    @property
    def ready(self) -> bool: return self.history.ready
    def reset(self): self.history.reset()
    def observe(self, sample): self.history.observe(_causal_row(sample))
    def state_dict(self): return self.history.state_dict()
    def load_state_dict(self,state): self.history.load_state_dict(state)
    def predict(self, sample):
        self.observe(sample); future = _future_rows(sample, self.output_shape[0],self.timestep_seconds)
        with torch.no_grad(): values = self.model(torch.from_numpy(self.history.values()[None]), torch.from_numpy(future[None]))["chlorine"].numpy()[0]
        return values.astype(np.float32)


def make_provider(spec: ForecastProviderSpec, horizon: int, node_ids: tuple[str, ...], resolve,timestep_seconds: int=1800):
    if isinstance(spec, PersistenceForecastProviderSpec): return PersistenceProvider(horizon, node_ids)
    if isinstance(spec, NoiseForecastProviderSpec): return NoiseProvider(spec, horizon, node_ids)
    if isinstance(spec, PretrainedForecastProviderSpec): return BundleProvider(resolve(spec.model), horizon, node_ids)
    if isinstance(spec, ShuffledForecastProviderSpec): return ShuffledProvider(resolve(spec.model), horizon, node_ids, spec.seed)
    if isinstance(spec, RandomFrozenForecastProviderSpec): return RandomFrozenProvider(spec, horizon, node_ids,timestep_seconds)
    raise TypeError(f"unsupported provider spec {type(spec).__name__}")


def make_latent_provider(spec: PretrainedLatentProviderSpec, node_ids: tuple[str, ...], resolve):
    if isinstance(spec.model, ArtifactRef):
        bundle = resolve(spec.model)
    else:
        bundle = Path(spec.model)
        if not bundle.is_absolute():
            bundle = PROJECT_ROOT / bundle
    return LatentBundleProvider(bundle, node_ids)


def make_latent_history_provider(spec: PretrainedLatentProviderSpec, node_ids: tuple[str, ...], resolve):
    return LatentHistoryBundleProvider(resolve_latent_bundle(spec, resolve), node_ids)


def resolve_latent_bundle(spec: PretrainedLatentProviderSpec, resolve) -> Path:
    """Resolve an artifact reference or repository-relative latent bundle."""
    if isinstance(spec.model, ArtifactRef):
        return Path(resolve(spec.model))
    bundle = Path(spec.model)
    return bundle if bundle.is_absolute() else PROJECT_ROOT / bundle


def _causal_row(sample):
    time = cyclic_time_features(sample.time_seconds, sample.calendar_start_day)
    return np.concatenate([sample.chlorine, [sample.injection], time], dtype=np.float32)


def _future_rows(sample, horizon,step=1800):
    rows=[]
    for index in range(1, horizon+1):
        time=cyclic_time_features(sample.time_seconds+index*step, sample.calendar_start_day)
        rows.append([sample.injection,*time])
    return np.asarray(rows, dtype=np.float32)


def _named_row(sample,names,time_seconds):
    node_index={node:index for index,node in enumerate(sample.node_ids)};values=[]
    time=dict(zip(("day_sin","day_cos","week_sin","week_cos","year_sin","year_cos"),
                  cyclic_time_features(time_seconds,sample.calendar_start_day),strict=True))
    for name in names:
        if name.startswith("chlorine:"):values.append(float(sample.chlorine[node_index[name.split(":",1)[1]]]))
        elif name in {"injection","planned_injection","chlorine-injection",f"chlorine-injection:{sample.node_ids[0]}"} or name.startswith("chlorine-injection:"):values.append(float(sample.injection))
        elif name in time:values.append(time[name])
        else:raise ValueError(f"deployable forecast has unsupported causal feature {name!r}")
    return np.asarray(values,dtype=np.float32)


def _apply_contract_transforms(values, transforms):
    result = np.asarray(values, dtype=np.float32).copy()
    if not transforms: return result
    if len(transforms) != result.shape[-1]: raise ValueError("bundle transform contract width mismatch")
    for index, contract in enumerate(transforms):
        if contract["transform"] == "log1p":
            if np.any(result[..., index] < 0): raise ValueError("bundle log1p input is negative")
            result[..., index] = np.log1p(result[..., index])
        elif contract["transform"] == "divide-by": result[..., index] /= float(contract["scale"])
        elif contract["transform"] != "identity": raise ValueError("unsupported bundle feature transform")
    return result
