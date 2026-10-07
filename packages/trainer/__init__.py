"""Public, lightweight configuration API for trainer2."""

from .cli import main
from .constants import (MAX_CHLORINE_INJECTION, V70_MAX_CHLORINE_DELTA,
                        V70_MAX_CHLORINE_INJECTION)
from .experiment import Experiment
from .cross_fit import CrossFitFold, cross_fit_folds
from .execution import RunContext, RunResult, Runner, RunnerRegistry
from .control import Reward, RewardInput
from .observations import (
    ControlObservation,
    DefaultControlObservation,
    default_control_observation,
)
from .presets import assisted_observation, complete_control_ablation_matrix, control_variant, recurrent_ppo, validate_matched_control_ablations
from .specs import (
    ArtifactRef, ControlHooksSpec, ControlPlotsActionSpec,
    ControlTrainingBaseSpec, ControlTrainingSpec,
    DistillationObjectiveSpec, EarlyStoppingSpec, EncoderDecoderLSTMSpec,
    LatentDynamicsForecastSpec, LatentDynamicsObjectiveSpec,
    NextLatentLSTMEncoderSpec, NextLatentObjectiveSpec,
    AuxiliaryNextLatentLSTMEncoderSpec, AuxiliaryNextLatentObjectiveSpec,
    ForecastAssistedControlObservationSpec, LatentAssistedControlObservationSpec,
    ProjectedLatentAssistedControlObservationSpec,
    FixedPCALatentAssistedControlObservationSpec,
    SharedDynamicsEncoderControlObservationSpec,
    TrainableNextLatentControlObservationSpec,
    ForecastLoopSpec, ForecastTrainingSpec,
    NoiseForecastProviderSpec, OptimizerSpec, OutcomeObjectiveSpec, OutputSpec,
    PersistenceForecastProviderSpec, PretrainedForecastProviderSpec,
    PretrainedLatentProviderSpec,
    RandomFrozenForecastProviderSpec,
    ControlObservationSpec, RecurrentPPOSpec,
    DynamicsAuxiliaryPPOSpec, ReduceOnPlateauSpec,
    ArsenicScheduleGeneratorSpec, HydraulicPeakArsenicScheduleGeneratorSpec,
    ArsenicScheduleStreamSpec, DatasetSpec, DatasetSplitSpec,
    FixedArsenicScheduleSetSpec, RewardSpec, ShuffledForecastProviderSpec,
    ChlorineInjectionScheduleGeneratorSpec, FixedChlorineInjectionScheduleSetSpec,
    ForecastTrajectoryDataSpec, ForecastFeatureSpec, ForecastInputSpec,
    ForecastHeadSpec, ForecastEvaluationSpec, LatentEvaluationSpec,
    ForecastNormalizationSpec, ForecastPlotsSpec,
    ForecastDiagnosticPlotsSpec,
)

__all__ = [name for name in globals() if not name.startswith("_")]
