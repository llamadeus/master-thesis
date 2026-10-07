"""Small frozen presets and matched-ablation validation."""

from __future__ import annotations

from dataclasses import replace

from .observations.control import default_control_observation
from .serialization import canonical_data
from .specs import (
    ControlTrainingBaseSpec, ControlTrainingSpec, EncoderDecoderLSTMSpec, ForecastAssistedControlObservationSpec,
    ForecastProviderSpec, NoiseForecastProviderSpec, PersistenceForecastProviderSpec,
    PretrainedForecastProviderSpec, RandomFrozenForecastProviderSpec,
    ControlObservationSpec, RecurrentPPOSpec, ShuffledForecastProviderSpec, ArtifactRef,
)


def recurrent_ppo(*, n_envs: int = 4) -> RecurrentPPOSpec:
    return RecurrentPPOSpec(n_envs=n_envs, batch_size=512, n_steps=2048, lstm_hidden_size=128)


def assisted_observation(
    provider: ForecastProviderSpec,
    *,
    node_ids: tuple[str, ...],
    horizon_steps: int,
    base: ControlObservationSpec,
) -> ForecastAssistedControlObservationSpec:
    return ForecastAssistedControlObservationSpec(
        provider=provider,
        node_ids=node_ids,
        horizon_steps=horizon_steps,
        base=base,
    )


def control_variant(base: ControlTrainingBaseSpec | ControlTrainingSpec, *, name: str, observation) -> ControlTrainingSpec:
    if isinstance(base, ControlTrainingBaseSpec):
        return base.build(name=name, observation=observation)
    return replace(base, name=name, observation=observation)


def validate_matched_control_ablations(jobs: tuple[ControlTrainingSpec, ...]) -> None:
    if len(jobs) < 2: raise ValueError("at least two control jobs are required for an ablation comparison")
    signatures=[]
    for job in jobs:
        data=canonical_data(job);data.pop("name");data.pop("observation");data.pop("output");data.pop("dependencies")
        signatures.append(data)
    if any(item != signatures[0] for item in signatures[1:]):
        raise ValueError("control ablations differ outside name, observation/provider, output, or dependencies")
    assisted=[job.observation for job in jobs if isinstance(job.observation,ForecastAssistedControlObservationSpec)]
    if assisted:
        contract={(item.horizon_steps,item.node_ids) for item in assisted}
        if len(contract)!=1: raise ValueError("forecast-assisted ablations must have identical forecast shape and node order")


def complete_control_ablation_matrix(
    base: ControlTrainingBaseSpec | ControlTrainingSpec,
    *,
    forecast: ArtifactRef,
    node_ids: tuple[str, ...],
    model: EncoderDecoderLSTMSpec,
    shuffled_seed: int,
    noise_seed: int,
    random_frozen_seed: int,
) -> tuple[ControlTrainingSpec, ...]:
    horizon=model.horizon_steps
    observation_base=default_control_observation()
    jobs=(
        control_variant(base,name="baseline",observation=observation_base),
        control_variant(base,name="pretrained-forecast",observation=assisted_observation(PretrainedForecastProviderSpec(forecast),node_ids=node_ids,horizon_steps=horizon,base=observation_base)),
        control_variant(base,name="persistence-forecast",observation=assisted_observation(PersistenceForecastProviderSpec(),node_ids=node_ids,horizon_steps=horizon,base=observation_base)),
        control_variant(base,name="shuffled-forecast",observation=assisted_observation(ShuffledForecastProviderSpec(forecast,shuffled_seed),node_ids=node_ids,horizon_steps=horizon,base=observation_base)),
        control_variant(base,name="noise-features",observation=assisted_observation(NoiseForecastProviderSpec(noise_seed),node_ids=node_ids,horizon_steps=horizon,base=observation_base)),
        control_variant(base,name="random-frozen-forecast",observation=assisted_observation(RandomFrozenForecastProviderSpec(random_frozen_seed,model),node_ids=node_ids,horizon_steps=horizon,base=observation_base)),
    )
    validate_matched_control_ablations(jobs)
    return jobs
