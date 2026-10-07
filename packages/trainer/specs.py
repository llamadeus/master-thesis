"""Typed, immutable configuration for trainer2 experiments."""

from __future__ import annotations

import importlib
import inspect
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

from .constants import MAX_CHLORINE_INJECTION
from .serialization import canonical_data, importable_type_path
from .control_observation import ControlObservation

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
Device = Literal["auto", "cpu", "cuda", "mps"]
ControlEvaluationSplit = Literal["validation", "test"]


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} may contain only letters, digits, '.', '_' and '-'")


def _positive(value: int | float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{label} must be positive")


def _probability(value: float, label: str, *, include_one: bool = True) -> None:
    upper_ok = value <= 1 if include_one else value < 1
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0 or not upper_ok:
        bracket = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{label} must be in {bracket}")


def validate_identifier(value: str, name: str) -> None:
    _identifier(value, name)


@dataclass(frozen=True)
class ArtifactRef:
    job: str
    artifact: str = "best"

    def __post_init__(self) -> None:
        _identifier(self.job, "artifact job")
        _identifier(self.artifact, "artifact name")


@dataclass(frozen=True)
class ArsenicScheduleGeneratorSpec:
    """Bounded strength prior with Gaussian parameters derived from its interval."""

    strength_distribution: Literal["uniform", "log-uniform", "gaussian"] = "log-uniform"
    minimum_strength_mg_per_minute: float = 1_000.0
    maximum_strength_mg_per_minute: float = 10_000.0

    def __post_init__(self) -> None:
        if self.strength_distribution not in ("uniform", "log-uniform", "gaussian"):
            raise ValueError(
                "arsenic strength_distribution must be 'uniform', 'log-uniform', or 'gaussian'"
            )
        bounds = (
            self.minimum_strength_mg_per_minute,
            self.maximum_strength_mg_per_minute,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
            for value in bounds
        ):
            raise ValueError("arsenic strength bounds must be finite and positive")
        if self.maximum_strength_mg_per_minute <= self.minimum_strength_mg_per_minute:
            raise ValueError("arsenic strength bounds must be increasing")
        object.__setattr__(
            self, "minimum_strength_mg_per_minute",
            float(self.minimum_strength_mg_per_minute),
        )
        object.__setattr__(
            self, "maximum_strength_mg_per_minute",
            float(self.maximum_strength_mg_per_minute),
        )
        if self.strength_distribution == "gaussian" and (
            not math.isfinite(self.strength_variance_mg2_per_minute2)
            or self.strength_variance_mg2_per_minute2 <= 0.0
        ):
            raise ValueError("arsenic strength bounds must yield a finite positive gaussian variance")

    @property
    def strength_mean_mg_per_minute(self) -> float:
        return self.minimum_strength_mg_per_minute + (
            self.maximum_strength_mg_per_minute
            - self.minimum_strength_mg_per_minute
        ) / 2.0

    @property
    def strength_variance_mg2_per_minute2(self) -> float:
        standard_deviation = (
            self.maximum_strength_mg_per_minute
            - self.minimum_strength_mg_per_minute
        ) / 4.0
        return standard_deviation * standard_deviation


@dataclass(frozen=True)
class HydraulicPeakArsenicScheduleGeneratorSpec:
    """Bounded local concentration prior converted to a MASS rate hydraulically."""

    mode: Literal["hydraulic-local-peak"] = field(
        default="hydraulic-local-peak", init=False
    )
    concentration_distribution: Literal["log-gaussian", "log-uniform"] = (
        "log-gaussian"
    )
    minimum_local_concentration_mg_per_liter: float = 0.1
    maximum_local_concentration_mg_per_liter: float = 10.0
    log_gaussian_midpoint_mg_per_liter: float | None = field(
        default=None, metadata={"canonical_omit_default": True}
    )
    maximum_mass_rate_mg_per_minute: float = 50_000.0

    def __post_init__(self) -> None:
        if self.concentration_distribution not in ("log-gaussian", "log-uniform"):
            raise ValueError(
                "hydraulic arsenic concentration_distribution must be "
                "'log-gaussian' or 'log-uniform'"
            )
        bounds = (
            self.minimum_local_concentration_mg_per_liter,
            self.maximum_local_concentration_mg_per_liter,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0.0
            for value in bounds
        ):
            raise ValueError(
                "hydraulic arsenic local-concentration bounds must be finite and positive"
            )
        if (
            self.maximum_local_concentration_mg_per_liter
            <= self.minimum_local_concentration_mg_per_liter
        ):
            raise ValueError(
                "hydraulic arsenic local-concentration bounds must be increasing"
            )
        object.__setattr__(
            self,
            "minimum_local_concentration_mg_per_liter",
            float(self.minimum_local_concentration_mg_per_liter),
        )
        object.__setattr__(
            self,
            "maximum_local_concentration_mg_per_liter",
            float(self.maximum_local_concentration_mg_per_liter),
        )
        if (
            isinstance(self.maximum_mass_rate_mg_per_minute, bool)
            or not isinstance(self.maximum_mass_rate_mg_per_minute, int | float)
            or not math.isfinite(self.maximum_mass_rate_mg_per_minute)
            or self.maximum_mass_rate_mg_per_minute <= 0.0
        ):
            raise ValueError(
                "hydraulic arsenic maximum MASS rate must be finite and positive"
            )
        object.__setattr__(
            self,
            "maximum_mass_rate_mg_per_minute",
            float(self.maximum_mass_rate_mg_per_minute),
        )
        midpoint = self.log_gaussian_midpoint_mg_per_liter
        if midpoint is not None:
            if self.concentration_distribution != "log-gaussian":
                raise ValueError(
                    "hydraulic arsenic log-Gaussian midpoint is only valid for "
                    "the 'log-gaussian' distribution"
                )
            if (
                isinstance(midpoint, bool)
                or not isinstance(midpoint, int | float)
                or not math.isfinite(midpoint)
                or not self.minimum_local_concentration_mg_per_liter
                < midpoint
                < self.maximum_local_concentration_mg_per_liter
            ):
                raise ValueError(
                    "hydraulic arsenic log-Gaussian midpoint must be finite and "
                    "strictly between the local-concentration bounds"
                )
            object.__setattr__(
                self, "log_gaussian_midpoint_mg_per_liter", float(midpoint)
            )

    @property
    def log_peak_mean(self) -> float:
        if self.log_gaussian_midpoint_mg_per_liter is not None:
            return math.log(self.log_gaussian_midpoint_mg_per_liter)
        lower = math.log(self.minimum_local_concentration_mg_per_liter)
        upper = math.log(self.maximum_local_concentration_mg_per_liter)
        return lower + (upper - lower) / 2.0

    @property
    def log_peak_variance(self) -> float:
        lower = math.log(self.minimum_local_concentration_mg_per_liter)
        upper = math.log(self.maximum_local_concentration_mg_per_liter)
        standard_deviation = (upper - lower) / 4.0
        return standard_deviation * standard_deviation

    @property
    def median_peak_concentration_mg_per_liter(self) -> float:
        if self.log_gaussian_midpoint_mg_per_liter is not None:
            return self.log_gaussian_midpoint_mg_per_liter
        return math.exp(self.log_peak_mean)


ArsenicScheduleGenerator: TypeAlias = (
    ArsenicScheduleGeneratorSpec | HydraulicPeakArsenicScheduleGeneratorSpec
)


@dataclass(frozen=True)
class FixedArsenicScheduleSetSpec:
    seed: int
    schedule_count: int
    generator: ArsenicScheduleGenerator = field(default_factory=ArsenicScheduleGeneratorSpec)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("arsenic schedule seed must be an integer")
        if isinstance(self.schedule_count, bool) or not isinstance(self.schedule_count, int) or self.schedule_count <= 0:
            raise ValueError("arsenic schedule_count must be a positive integer")
        if not isinstance(
            self.generator,
            ArsenicScheduleGeneratorSpec | HydraulicPeakArsenicScheduleGeneratorSpec,
        ):
            raise ValueError("unsupported arsenic schedule generator")


@dataclass(frozen=True)
class ArsenicScheduleStreamSpec:
    seed: int
    seed_path: tuple[int, ...] = ()
    generator: ArsenicScheduleGenerator = field(default_factory=ArsenicScheduleGeneratorSpec)
    minimum_event_count: int = field(
        default=1, metadata={"canonical_omit_default": True}
    )
    maximum_event_count: int = field(
        default=4, metadata={"canonical_omit_default": True}
    )
    minimum_event_duration_steps: int = field(
        default=4, metadata={"canonical_omit_default": True}
    )
    maximum_event_duration_steps: int = field(
        default=144, metadata={"canonical_omit_default": True}
    )
    episode_steps: int | None = field(
        default=None, metadata={"canonical_omit_default": True}
    )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("arsenic schedule seed must be an integer")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.seed_path):
            raise ValueError("arsenic schedule seed_path values must be non-negative integers")
        if not isinstance(
            self.generator,
            ArsenicScheduleGeneratorSpec | HydraulicPeakArsenicScheduleGeneratorSpec,
        ):
            raise ValueError("unsupported arsenic schedule generator")
        for minimum, maximum, label in (
            (
                self.minimum_event_count,
                self.maximum_event_count,
                "arsenic event count",
            ),
            (
                self.minimum_event_duration_steps,
                self.maximum_event_duration_steps,
                "arsenic event duration",
            ),
        ):
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or minimum <= 0
                or maximum < minimum
            ):
                raise ValueError(
                    f"{label} bounds must be positive, ordered integers"
                )
        if (
            self.episode_steps is not None
            and (
                isinstance(self.episode_steps, bool)
                or not isinstance(self.episode_steps, int)
                or self.episode_steps <= 0
            )
        ):
            raise ValueError("arsenic schedule episode_steps must be positive")

    def for_environment(self, environment_index: int) -> ArsenicScheduleStreamSpec:
        if isinstance(environment_index, bool) or not isinstance(environment_index, int) or environment_index < 0:
            raise ValueError("environment index must be a non-negative integer")
        return ArsenicScheduleStreamSpec(
            seed=self.seed,
            seed_path=(*self.seed_path, environment_index),
            generator=self.generator,
            minimum_event_count=self.minimum_event_count,
            maximum_event_count=self.maximum_event_count,
            minimum_event_duration_steps=self.minimum_event_duration_steps,
            maximum_event_duration_steps=self.maximum_event_duration_steps,
            episode_steps=self.episode_steps,
        )


ArsenicScheduleSourceSpec: TypeAlias = FixedArsenicScheduleSetSpec | ArsenicScheduleStreamSpec


@dataclass(frozen=True)
class ChlorineInjectionScheduleGeneratorSpec:
    families: tuple[str, ...]
    mix_families_within_schedule: bool = False
    set_coverage: Literal["independent", "balanced"] = "independent"
    transition_style: str = "abrupt"
    minimum_injection: float = 0.0
    maximum_injection: float = MAX_CHLORINE_INJECTION
    sampling_bands: tuple[tuple[float, float, float], ...] = field(
        default=(), metadata={"canonical_omit_default": True}
    )

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("chlorine generator families must not be empty")
        for family in self.families:
            _identifier(family, "chlorine pattern family")
        if len(set(self.families)) != len(self.families):
            raise ValueError("chlorine generator families must be unique")
        _identifier(self.transition_style, "chlorine transition style")
        if self.minimum_injection < 0 or self.maximum_injection <= self.minimum_injection:
            raise ValueError("chlorine injection bounds must be non-negative and increasing")
        if self.sampling_bands:
            previous_upper = self.minimum_injection
            for lower, upper, weight in self.sampling_bands:
                if not all(math.isfinite(value) for value in (lower, upper, weight)):
                    raise ValueError("chlorine sampling bands must be finite")
                if lower != previous_upper or upper <= lower or weight <= 0.0:
                    raise ValueError(
                        "chlorine sampling bands must be contiguous, increasing, and positive-weighted"
                    )
                previous_upper = upper
            if previous_upper != self.maximum_injection:
                raise ValueError(
                    "chlorine sampling bands must cover the complete injection interval"
                )


@dataclass(frozen=True)
class FixedChlorineInjectionScheduleSetSpec:
    seed: int
    schedule_count: int
    generator: ChlorineInjectionScheduleGeneratorSpec

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("chlorine schedule seed must be an integer")
        if isinstance(self.schedule_count, bool) or not isinstance(self.schedule_count, int) or self.schedule_count <= 0:
            raise ValueError("chlorine schedule_count must be a positive integer")


ChlorineInjectionScheduleSourceSpec: TypeAlias = FixedChlorineInjectionScheduleSetSpec


@dataclass(frozen=True)
class DatasetSplitSpec:
    scenarios: tuple[str, ...]
    arsenic: ArsenicScheduleSourceSpec | None = None
    chlorine: ChlorineInjectionScheduleSourceSpec | None = None

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("dataset split scenarios must not be empty")
        for value in self.scenarios:
            _identifier(value, "dataset split scenario")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("dataset split scenarios contain duplicates")


@dataclass(frozen=True)
class DatasetSpec:
    network: str
    train: DatasetSplitSpec
    validation: DatasetSplitSpec
    test: DatasetSplitSpec
    scenarios_dir: str = "hanoi"
    uncertainty_seed: int = 42

    def __post_init__(self) -> None:
        _identifier(self.network, "dataset network")
        _identifier(self.scenarios_dir, "scenarios_dir (a direct child of data/scenarios/)")
        if isinstance(self.uncertainty_seed, bool) or not isinstance(self.uncertainty_seed, int) or self.uncertainty_seed < 0:
            raise ValueError("uncertainty_seed must be a non-negative integer")
        combined = (*self.train.scenarios, *self.validation.scenarios, *self.test.scenarios)
        if len(set(combined)) != len(combined):
            raise ValueError("dataset train, validation, and test scenarios must be disjoint")
        # Schedule sources own their iteration semantics.  DatasetSpec deliberately
        # does not assign source types to particular splits or runner roles.


@dataclass(frozen=True)
class OutputSpec:
    root: str | Path = "data/weights"
    group: str = ""

    def __post_init__(self) -> None:
        if not str(self.root):
            raise ValueError("output root must be non-empty")
        group = Path(self.group)
        if group.is_absolute() or ".." in group.parts:
            raise ValueError("output group must be relative and may not contain '..'")


@dataclass(frozen=True)
class ReduceOnPlateauSpec:
    factor: float = 0.5
    patience: int = 4
    min_learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.factor < 1:
            raise ValueError("scheduler factor must be in (0, 1)")
        if self.patience < 0:
            raise ValueError("scheduler patience must be non-negative")
        if self.min_learning_rate < 0:
            raise ValueError("minimum learning rate must be non-negative")


@dataclass(frozen=True)
class OptimizerSpec:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: ReduceOnPlateauSpec | None = field(default_factory=ReduceOnPlateauSpec)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning rate")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be non-negative")


@dataclass(frozen=True)
class EarlyStoppingSpec:
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("early-stopping patience and min_delta must be non-negative")


def _canonical_parameters(value: Mapping[str, Any] | tuple[tuple[str, Any], ...], label: str) -> tuple[tuple[str, Any], ...]:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if any(not isinstance(key, str) or _IDENTIFIER.fullmatch(key) is None for key, _ in items):
        raise ValueError(f"{label} names must be valid identifiers")
    if len({key for key, _ in items}) != len(items):
        raise ValueError(f"{label} names must be unique")
    # Validate serializability here, while retaining immutable canonical order.
    from .serialization import canonical_data
    for _key, item in items:
        canonical_data(item)
    return tuple(sorted(items))


@dataclass(frozen=True)
class ForecastFeatureSpec:
    name: str
    scope: Literal["global", "node", "link", "source"]
    parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(default=(), metadata={"canonical_mapping": True})
    transform: Literal["identity", "log1p", "divide-by"] = "identity"
    transform_parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(default=(), metadata={"canonical_mapping": True})
    scale: float | None = None
    cycles: tuple[Literal["day", "week", "year"], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "forecast feature provider")
        object.__setattr__(self, "parameters", _canonical_parameters(self.parameters, "feature parameter"))
        object.__setattr__(self, "transform_parameters", _canonical_parameters(self.transform_parameters, "transform parameter"))
        if self.name == "cyclic-time":
            if self.scope != "global" or not self.cycles or len(set(self.cycles)) != len(self.cycles):
                raise ValueError("cyclic-time requires unique configured cycles and global scope")
        elif self.cycles:
            raise ValueError("cycles are only supported by cyclic-time")
        configured_scale = self.scale if self.scale is not None else dict(self.transform_parameters).get("scale")
        if self.transform == "divide-by":
            _positive(configured_scale, "feature transform scale")
            object.__setattr__(self, "scale", float(configured_scale))
        elif self.scale is not None:
            raise ValueError("feature scale is only valid for divide-by")

    def parameters_dict(self) -> dict[str, Any]:
        return dict(self.parameters)


@dataclass(frozen=True)
class ForecastInputSpec:
    history: tuple[ForecastFeatureSpec, ...]
    future_known: tuple[ForecastFeatureSpec, ...]

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("forecast history features must not be empty")
        # Repetition would make bundle contracts and normalization ambiguous.
        for label, values in (("history", self.history), ("future-known", self.future_known)):
            identities = [(item.name, item.scope) for item in values]
            if len(set(identities)) != len(identities):
                raise ValueError(f"forecast {label} features must be unique")


@dataclass(frozen=True)
class ForecastHeadSpec:
    name: str
    target: str
    prediction: Literal["level", "delta"] = "level"
    transform: Literal["identity", "log1p", "divide-by"] = "identity"
    scale: float | None = None
    parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(default=(), metadata={"canonical_mapping": True})

    def __post_init__(self) -> None:
        _identifier(self.name, "forecast head")
        _identifier(self.target, "forecast target")
        object.__setattr__(self, "parameters", _canonical_parameters(self.parameters, "head parameter"))
        if self.transform == "divide-by":
            _positive(self.scale, "head transform scale")
        elif self.scale is not None:
            raise ValueError("head scale is only valid for divide-by")
        if self.prediction == "delta" and self.transform == "log1p":
            raise ValueError("log1p is incompatible with signed delta prediction")


@dataclass(frozen=True)
class ForecastPlotsSpec:
    splits: tuple[Literal["train", "validation", "test"], ...] = ("validation", "test")
    one_plot_per_scenario: bool = True
    scenario_trajectory_selection: str = "first"
    separate_target_files: bool = True
    node_selection: str | tuple[str, ...] = "all"
    separate_node_files: bool = False
    rolling_prediction_steps: int = 1
    actual_line_style: str = "solid"
    rolling_prediction_line_style: str = "dashed"
    include_random_horizon_plot: bool = False
    random_horizon_cutoff_day_range: tuple[float, float] | None = None
    random_horizon_include_history: bool = True
    random_horizon_include_past_predictions: bool = False
    random_horizon_line_style: str = "dotted"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("forecast plot splits must be non-empty and unique")
        _positive(self.rolling_prediction_steps, "rolling prediction steps")
        if isinstance(self.node_selection, tuple):
            if not self.node_selection or len(set(self.node_selection)) != len(self.node_selection):
                raise ValueError("plot node selection must be non-empty and unique")
        else:
            _identifier(self.node_selection, "plot node selection")
        if self.random_horizon_cutoff_day_range is not None:
            if len(self.random_horizon_cutoff_day_range) != 2:
                raise ValueError("random horizon cutoff day range requires two values")
            start, end = self.random_horizon_cutoff_day_range
            if (any(isinstance(value, bool) or not isinstance(value, int | float)
                    for value in (start, end)) or start < 0 or end < start):
                raise ValueError(
                    "random horizon cutoff day range must be non-negative and ordered"
                )
        for value, label in ((self.scenario_trajectory_selection, "trajectory selection"),
                             (self.actual_line_style, "actual line style"),
                             (self.rolling_prediction_line_style, "prediction line style"),
                             (self.random_horizon_line_style, "random horizon line style")):
            _identifier(value, label)


@dataclass(frozen=True)
class ForecastDiagnosticPlotsSpec(ForecastPlotsSpec):
    include_training_history_plot: bool = True
    metric_breakdown_plots: tuple[Literal["node", "horizon"], ...] = (
        "node",
        "horizon",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(set(self.metric_breakdown_plots)) != len(self.metric_breakdown_plots):
            raise ValueError("metric plot breakdowns must be unique")
        unsupported = set(self.metric_breakdown_plots) - {"node", "horizon"}
        if unsupported:
            raise ValueError(
                f"unsupported metric plot breakdowns: {sorted(unsupported)}"
            )


@dataclass(frozen=True)
class ForecastEvaluationSpec:
    targets: tuple[str, ...]
    metrics: tuple[Literal["mae", "mse", "rmse"] , ...] = ("mae", "rmse")
    breakdowns: tuple[Literal["overall", "node", "horizon", "scenario"], ...] = ("overall",)
    report_in_physical_units: bool = True
    scenario_weighting: Literal["pooled", "equal"] = "pooled"
    plots: ForecastPlotsSpec | None = None
    checkpoint_metric: str = "outcome_loss"
    runtime_plots: ForecastPlotsSpec | None = field(
        default=None,
        metadata={"canonical_exclude": True},
    )

    @property
    def effective_plots(self) -> ForecastPlotsSpec | None:
        """Plot settings that do not invalidate an already trained model."""
        return self.runtime_plots if self.runtime_plots is not None else self.plots

    def __post_init__(self) -> None:
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("forecast evaluation targets must be non-empty and unique")
        for target in self.targets:
            _identifier(target, "evaluation target")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("forecast evaluation metrics must be non-empty and unique")
        if not self.breakdowns or len(set(self.breakdowns)) != len(self.breakdowns):
            raise ValueError("forecast evaluation breakdowns must be non-empty and unique")
        if self.scenario_weighting not in ("pooled", "equal"):
            raise ValueError("invalid scenario weighting")
        _identifier(self.checkpoint_metric, "checkpoint metric")


@dataclass(frozen=True)
class LatentEvaluationSpec:
    """Loss-only evaluation for headless latent-representation training."""

    checkpoint_metric: str = "combined_loss"
    plots: None = field(default=None, init=False)

    @property
    def effective_plots(self) -> None:
        return None

    def __post_init__(self) -> None:
        _identifier(self.checkpoint_metric, "latent checkpoint metric")


@dataclass(frozen=True)
class ForecastNormalizationSpec:
    """Choose whether feature transforms are followed by fitted z-scoring."""

    inputs: Literal["fitted", "fixed"] = "fitted"

    def __post_init__(self) -> None:
        if self.inputs not in ("fitted", "fixed"):
            raise ValueError("forecast input normalization must be 'fitted' or 'fixed'")


@dataclass(frozen=True)
class ForecastTrajectoryDataSpec:
    backend: str
    cache_root: str | Path
    materialize_missing: bool = False
    schedule_pairing: Literal["cartesian", "zip", "broadcast"] = "cartesian"
    simulator_parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(default=(), metadata={"canonical_mapping": True})

    def __post_init__(self) -> None:
        _identifier(self.backend, "forecast simulator backend")
        if not str(self.cache_root):
            raise ValueError("forecast cache root must be non-empty")
        object.__setattr__(self, "simulator_parameters", _canonical_parameters(self.simulator_parameters, "simulator parameter"))


@dataclass(frozen=True)
class EncoderDecoderLSTMSpec:
    history_steps: int
    horizon_steps: int
    heads: tuple[ForecastHeadSpec, ...]
    decoder_mode: Literal["shared", "separate-per-head"]
    hidden_size: int = 192
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("history steps", self.history_steps),
            ("horizon steps", self.horizon_steps),
            ("hidden size", self.hidden_size),
            ("number of layers", self.num_layers),
        ):
            _positive(value, label)
        _probability(self.dropout, "dropout", include_one=False)
        if not self.heads:
            raise ValueError("forecast model heads must not be empty")
        names = [head.name for head in self.heads]
        if len(set(names)) != len(names):
            raise ValueError("forecast model head names must be unique")


@dataclass(frozen=True)
class LatentDynamicsForecastSpec:
    history_steps: int
    horizon_steps: int
    heads: tuple[ForecastHeadSpec, ...]
    hidden_size: int = 192
    transition_hidden_size: int = 192

    def __post_init__(self) -> None:
        for label, value in (
            ("history steps", self.history_steps),
            ("horizon steps", self.horizon_steps),
            ("hidden size", self.hidden_size),
            ("transition hidden size", self.transition_hidden_size),
        ):
            _positive(value, label)
        if len(self.heads) != 1:
            raise ValueError("latent-dynamics forecasting requires exactly one chlorine head")
        head = self.heads[0]
        if head.name != "chlorine" or head.target != "chlorine" or head.prediction != "delta":
            raise ValueError("latent-dynamics forecasting requires a delta-prediction chlorine head")


@dataclass(frozen=True)
class NextLatentLSTMEncoderSpec:
    """Headless LSTM encoder trained to predict its next shifted-window latent."""

    history_steps: int
    latent_size: int = 192
    predictor_hidden_size: int = 192
    num_layers: int = 1
    dropout: float = 0.0
    ema_decay: float = 0.99
    horizon_steps: Literal[1] = field(default=1, init=False)
    heads: tuple[ForecastHeadSpec, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("history steps", self.history_steps),
            ("latent size", self.latent_size),
            ("predictor hidden size", self.predictor_hidden_size),
            ("number of layers", self.num_layers),
        ):
            _positive(value, label)
        _probability(self.dropout, "dropout", include_one=False)
        _probability(self.ema_decay, "EMA decay", include_one=False)


@dataclass(frozen=True)
class AuxiliaryNextLatentLSTMEncoderSpec:
    """Next-latent encoder with an action-conditioned chlorine decoder."""

    history_steps: int
    horizon_steps: int
    heads: tuple[ForecastHeadSpec, ...]
    latent_size: int = 192
    predictor_hidden_size: int = 192
    num_layers: int = 1
    dropout: float = 0.0
    ema_decay: float = 0.99

    def __post_init__(self) -> None:
        for label, value in (
            ("history steps", self.history_steps),
            ("horizon steps", self.horizon_steps),
            ("latent size", self.latent_size),
            ("predictor hidden size", self.predictor_hidden_size),
            ("number of layers", self.num_layers),
        ):
            _positive(value, label)
        _probability(self.dropout, "dropout", include_one=False)
        _probability(self.ema_decay, "EMA decay", include_one=False)
        if len(self.heads) != 1:
            raise ValueError(
                "auxiliary next-latent forecasting requires exactly one chlorine head"
            )
        head = self.heads[0]
        if head.name != "chlorine" or head.target != "chlorine":
            raise ValueError(
                "auxiliary next-latent forecasting requires a chlorine target head"
            )


ForecastModelSpec: TypeAlias = (
    EncoderDecoderLSTMSpec | LatentDynamicsForecastSpec | NextLatentLSTMEncoderSpec
    | AuxiliaryNextLatentLSTMEncoderSpec
)


@dataclass(frozen=True)
class ForecastLoopSpec:
    epochs: int = 100
    batch_size: int = 256
    num_workers: int = 0
    device: Device = "auto"
    early_stopping: EarlyStoppingSpec = field(default_factory=EarlyStoppingSpec)

    def __post_init__(self) -> None:
        _positive(self.epochs, "epochs")
        _positive(self.batch_size, "batch size")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class OutcomeObjectiveSpec:
    head_weights: tuple[tuple[str, float], ...]
    loss: Literal["mse", "mae"] = "mse"
    normalize_targets_independently: bool = True
    checkpoint_metric: str = "outcome_loss"
    name: Literal["outcome"] = field(default="outcome", init=False)

    def __post_init__(self) -> None:
        names = []
        for name, weight in self.head_weights:
            _identifier(name, "objective head")
            if weight < 0:
                raise ValueError("objective head weights must be non-negative")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("objective head weights must be unique")
        if not self.head_weights:
            raise ValueError("objective head weights must not be empty")
        if not any(weight > 0 for _, weight in self.head_weights):
            raise ValueError("at least one objective head weight must be positive")
        _identifier(self.checkpoint_metric, "objective checkpoint metric")


@dataclass(frozen=True)
class DistillationObjectiveSpec:
    teacher: ArtifactRef
    teacher_weight: float
    outcome_weight: float
    loss: Literal["mse", "mae"] = "mse"
    teacher_head: str = "chlorine"
    student_head: str = "chlorine"
    checkpoint_metric: str = "outcome_loss"
    name: Literal["distillation"] = field(default="distillation", init=False)

    def __post_init__(self) -> None:
        if self.teacher_weight < 0 or self.outcome_weight < 0:
            raise ValueError("distillation weights must be non-negative")
        if self.teacher_weight + self.outcome_weight <= 0:
            raise ValueError("at least one distillation weight must be positive")
        _identifier(self.teacher_head, "distillation source head")
        _identifier(self.student_head, "distillation destination head")
        _identifier(self.checkpoint_metric, "distillation checkpoint metric")


@dataclass(frozen=True)
class LatentDynamicsObjectiveSpec:
    chlorine_weight: float
    checkpoint_metric: str = "chlorine.rmse"
    name: Literal["latent-dynamics"] = field(default="latent-dynamics", init=False)

    def __post_init__(self) -> None:
        _probability(self.chlorine_weight, "latent-dynamics chlorine weight")
        _identifier(self.checkpoint_metric, "latent-dynamics checkpoint metric")

    @property
    def latent_weight(self) -> float:
        return 1.0 - self.chlorine_weight


@dataclass(frozen=True)
class NextLatentObjectiveSpec:
    variance_weight: float = 1.0
    variance_target_std: float = 1.0
    variance_epsilon: float = 1e-4
    variance_regularizer: Literal["floor", "two-sided"] = field(
        default="floor", metadata={"canonical_omit_default": True}
    )
    name: Literal["next-latent"] = field(default="next-latent", init=False)

    def __post_init__(self) -> None:
        if self.variance_weight < 0:
            raise ValueError("next-latent variance weight must be non-negative")
        _positive(self.variance_target_std, "next-latent variance target standard deviation")
        _positive(self.variance_epsilon, "next-latent variance epsilon")
        if self.variance_regularizer not in ("floor", "two-sided"):
            raise ValueError("unsupported next-latent variance regularizer")


@dataclass(frozen=True)
class AuxiliaryNextLatentObjectiveSpec:
    """Joint next-latent and auxiliary chlorine forecast objective."""

    chlorine_weight: float = 1.0
    loss: Literal["mse", "mae"] = "mse"
    variance_weight: float = 1.0
    variance_target_std: float = 1.0
    variance_epsilon: float = 1e-4
    variance_regularizer: Literal["floor", "two-sided"] = field(
        default="floor", metadata={"canonical_omit_default": True}
    )
    name: Literal["auxiliary-next-latent"] = field(
        default="auxiliary-next-latent", init=False
    )

    def __post_init__(self) -> None:
        if self.chlorine_weight < 0:
            raise ValueError("auxiliary chlorine weight must be non-negative")
        if self.variance_weight < 0:
            raise ValueError("next-latent variance weight must be non-negative")
        _positive(
            self.variance_target_std,
            "next-latent variance target standard deviation",
        )
        _positive(self.variance_epsilon, "next-latent variance epsilon")
        if self.variance_regularizer not in ("floor", "two-sided"):
            raise ValueError("unsupported next-latent variance regularizer")


ForecastObjectiveSpec: TypeAlias = (
    OutcomeObjectiveSpec | DistillationObjectiveSpec | LatentDynamicsObjectiveSpec
    | NextLatentObjectiveSpec | AuxiliaryNextLatentObjectiveSpec
)


@dataclass(frozen=True)
class RewardSpec:
    factory: type[Any]
    parameters: Mapping[str, float | bool] | tuple[
        tuple[str, float | bool], ...
    ] = field(
        default=(),metadata={"canonical_mapping":True},
    )

    def __post_init__(self) -> None:
        importable_type_path(self.factory)
        items=tuple(self.parameters.items()) if isinstance(self.parameters,Mapping) else tuple(self.parameters)
        if any(not isinstance(key,str) or not key for key,_value in items):
            raise ValueError("reward parameter names must be non-empty strings")
        if any(not isinstance(value,int|float) for _key,value in items):
            raise ValueError("reward parameter values must be numeric or boolean")
        if len({key for key,_value in items}) != len(items):
            raise ValueError("reward parameter names must be unique")
        object.__setattr__(self,"parameters",tuple(sorted(items)))

    def as_dict(self) -> dict[str, float | bool]:
        return dict(self.parameters)


@dataclass(frozen=True)
class RecurrentPPOSpec:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.05
    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True
    n_envs: int = 1
    eval_frequency: int = 50_000
    checkpoint_frequency: int = 50_000
    normalize_observations: bool = True
    normalize_rewards: bool = True
    clip_observation: float = 10.0
    clip_reward: float = 10.0
    pretrained_policy: ArtifactRef | str | Path | None = None
    pretrained_components: tuple[Literal["actor", "action_std", "observation_normalization"], ...] = field(
        default=("actor", "action_std", "observation_normalization"),
        metadata={"canonical_omit_default": True},
    )
    device: Device = "auto"

    def __post_init__(self) -> None:
        for label, value in (
            ("learning rate", self.learning_rate), ("n_steps", self.n_steps),
            ("batch size", self.batch_size), ("n_epochs", self.n_epochs),
            ("gamma", self.gamma), ("gae_lambda", self.gae_lambda),
            ("clip range", self.clip_range), ("max grad norm", self.max_grad_norm),
            ("LSTM hidden size", self.lstm_hidden_size), ("LSTM layers", self.n_lstm_layers),
            ("environment count", self.n_envs), ("evaluation frequency", self.eval_frequency),
            ("checkpoint frequency", self.checkpoint_frequency),
            ("observation clip", self.clip_observation), ("reward clip", self.clip_reward),
        ):
            _positive(value, label)
        _probability(self.gamma, "gamma")
        _probability(self.gae_lambda, "gae_lambda")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("PPO batch size cannot exceed rollout size")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("PPO loss coefficients must be non-negative")
        if self.target_kl is not None:
            _positive(self.target_kl, "target KL divergence")
        if isinstance(self.pretrained_policy, str | Path) and not str(self.pretrained_policy):
            raise ValueError("pretrained policy path must be non-empty")
        components = self.pretrained_components
        allowed = {"actor", "action_std", "observation_normalization"}
        if (not isinstance(components, tuple) or not components
                or any(component not in allowed for component in components)
                or len(set(components)) != len(components)):
            raise ValueError("pretrained components must be a nonempty tuple without duplicates")
        if self.pretrained_policy is None and set(components) != allowed:
            raise ValueError("selected pretrained components require a pretrained policy bundle")


@dataclass(frozen=True)
class DynamicsAuxiliaryPPOSpec(RecurrentPPOSpec):
    """PPO whose temporal encoder also receives a latent-dynamics update.

    The dynamics encoder replaces the recurrent policy LSTM. The inherited
    rollout and optimization fields intentionally keep this experiment matched
    to the recurrent-PPO control baseline.
    """

    auxiliary_loss_coefficient: float = 0.1
    auxiliary_epochs: int = 1
    auxiliary_batch_size: int = 512
    chlorine_weight: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        _positive(self.auxiliary_loss_coefficient, "auxiliary loss coefficient")
        _positive(self.auxiliary_epochs, "auxiliary epochs")
        _positive(self.auxiliary_batch_size, "auxiliary batch size")
        _probability(self.chlorine_weight, "auxiliary chlorine weight")

    @property
    def latent_weight(self) -> float:
        return 1.0 - self.chlorine_weight


@dataclass(frozen=True)
class ControlPlotsActionSpec:
    """Render trajectories for the selected control evaluation splits."""

    splits: tuple[ControlEvaluationSplit, ...]
    name: Literal["plots"] = field(default="plots", init=False)

    def __post_init__(self) -> None:
        if not self.splits:
            raise ValueError("control plot action requires at least one split")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("control plot action contains duplicate splits")
        invalid = set(self.splits) - {"validation", "test"}
        if invalid:
            raise ValueError(f"control plot action contains invalid splits: {sorted(invalid)}")


ControlHookActionSpec: TypeAlias = ControlPlotsActionSpec


@dataclass(frozen=True)
class ControlHooksSpec:
    """Actions invoked at stable points in the control training lifecycle."""

    on_checkpoint: tuple[ControlHookActionSpec, ...] = ()
    on_finish: tuple[ControlHookActionSpec, ...] = ()

    def __post_init__(self) -> None:
        for label, actions in (("on_checkpoint", self.on_checkpoint), ("on_finish", self.on_finish)):
            names = [action.name for action in actions]
            if len(set(names)) != len(names):
                raise ValueError(f"control hooks {label} contains duplicate actions")


@dataclass(frozen=True)
class ControlObservationSpec:
    """Factory for a fresh experiment-owned control observation runtime."""

    factory: type[ControlObservation]
    parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(
        default=(), metadata={"canonical_mapping": True}
    )

    def __post_init__(self) -> None:
        factory_path = importable_type_path(self.factory)
        if not issubclass(self.factory, ControlObservation):
            raise TypeError(
                "control observation factory must inherit ControlObservation"
            )
        if inspect.isabstract(self.factory):
            raise TypeError(
                "control observation factory must implement the runtime contract"
            )
        module_name, qualname = factory_path.split(":", 1)
        if module_name == "__main__":
            raise ValueError(
                "control observation factory must be a top-level importable class"
            )
        try:
            resolved: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError) as error:
            raise ValueError(
                "control observation factory must be a top-level importable class"
            ) from error
        if resolved is not self.factory:
            raise ValueError(
                "control observation factory must be a top-level importable class"
            )
        items = (
            tuple(self.parameters.items())
            if isinstance(self.parameters, Mapping)
            else tuple(self.parameters)
        )
        if any(
            not isinstance(key, str) or not key.isidentifier()
            for key, _value in items
        ):
            raise ValueError(
                "control observation parameter names must be valid identifiers"
            )
        if len({key for key, _value in items}) != len(items):
            raise ValueError("control observation parameter names must be unique")
        canonical_data(dict(items))
        object.__setattr__(self, "parameters", tuple(sorted(items)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.parameters)


@dataclass(frozen=True)
class PretrainedForecastProviderSpec:
    model: ArtifactRef
    name: Literal["pretrained"] = field(default="pretrained", init=False)
    startup_contract: Literal["simulator-pre-roll-v1"] = field(
        default="simulator-pre-roll-v1", init=False
    )


@dataclass(frozen=True)
class PersistenceForecastProviderSpec:
    name: Literal["persistence"] = field(default="persistence", init=False)


@dataclass(frozen=True)
class ShuffledForecastProviderSpec:
    model: ArtifactRef
    seed: int
    name: Literal["shuffled"] = field(default="shuffled", init=False)
    startup_contract: Literal["simulator-pre-roll-v1"] = field(
        default="simulator-pre-roll-v1", init=False
    )


@dataclass(frozen=True)
class NoiseForecastProviderSpec:
    seed: int
    standard_deviation: float = 1.0
    distribution: Literal["normal", "uniform"] = "normal"
    name: Literal["noise"] = field(default="noise", init=False)

    def __post_init__(self) -> None:
        _positive(self.standard_deviation, "noise standard deviation")


@dataclass(frozen=True)
class RandomFrozenForecastProviderSpec:
    seed: int
    model: EncoderDecoderLSTMSpec
    name: Literal["random-frozen"] = field(default="random-frozen", init=False)
    startup_contract: Literal["simulator-pre-roll-v1"] = field(
        default="simulator-pre-roll-v1", init=False
    )


ForecastProviderSpec: TypeAlias = (
    PretrainedForecastProviderSpec
    | PersistenceForecastProviderSpec
    | ShuffledForecastProviderSpec
    | NoiseForecastProviderSpec
    | RandomFrozenForecastProviderSpec
)


@dataclass(frozen=True)
class PretrainedLatentProviderSpec:
    """A frozen latent-dynamics bundle from a job artifact or disk path."""

    model: ArtifactRef | str | Path
    name: Literal["pretrained-latent"] = field(
        default="pretrained-latent", init=False
    )
    startup_contract: Literal["simulator-pre-roll-v1"] = field(
        default="simulator-pre-roll-v1", init=False
    )

    def __post_init__(self) -> None:
        if not str(self.model):
            raise ValueError("pretrained latent model path must be non-empty")


@dataclass(frozen=True)
class ForecastAssistedControlObservationSpec:
    provider: ForecastProviderSpec
    horizon_steps: int
    node_ids: tuple[str, ...]
    base: ControlObservationSpec
    name: Literal["forecast-assisted"] = field(default="forecast-assisted", init=False)

    def __post_init__(self) -> None:
        _positive(self.horizon_steps, "forecast horizon")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("forecast-assisted node_ids must be non-empty and unique")


@dataclass(frozen=True)
class LatentAssistedControlObservationSpec:
    """Recurrent observations augmented with a frozen dynamics latent state."""

    provider: PretrainedLatentProviderSpec
    node_ids: tuple[str, ...]
    latent_size: int
    base: ControlObservationSpec
    name: Literal["latent-assisted"] = field(default="latent-assisted", init=False)

    def __post_init__(self) -> None:
        _positive(self.latent_size, "latent state size")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(
                "latent-assisted node_ids must be non-empty and unique"
            )


@dataclass(frozen=True)
class ProjectedLatentAssistedControlObservationSpec:
    """Frozen dynamics latent adapted before the recurrent PPO policy."""

    provider: PretrainedLatentProviderSpec
    node_ids: tuple[str, ...]
    latent_size: int
    projected_size: int
    base: ControlObservationSpec
    initial_gate_logit: float = -4.0
    name: Literal["projected-latent-assisted"] = field(
        default="projected-latent-assisted", init=False
    )

    def __post_init__(self) -> None:
        _positive(self.latent_size, "latent state size")
        _positive(self.projected_size, "projected latent size")
        if (
            isinstance(self.initial_gate_logit, bool)
            or not isinstance(self.initial_gate_logit, int | float)
            or not math.isfinite(self.initial_gate_logit)
        ):
            raise ValueError("initial latent gate logit must be finite")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(
                "projected-latent-assisted node_ids must be non-empty and unique"
            )


@dataclass(frozen=True)
class FixedPCALatentAssistedControlObservationSpec:
    """Frozen dynamics latent mapped through a fixed, whitened PCA basis."""

    provider: PretrainedLatentProviderSpec
    node_ids: tuple[str, ...]
    latent_size: int
    projected_size: int
    projection: str
    base: ControlObservationSpec
    whitening_epsilon: float = 1e-6
    clip_projected: float = 10.0
    name: Literal["fixed-pca-latent-assisted"] = field(
        default="fixed-pca-latent-assisted", init=False
    )

    def __post_init__(self) -> None:
        _positive(self.latent_size, "latent state size")
        _positive(self.projected_size, "PCA projected size")
        _positive(self.whitening_epsilon, "PCA whitening epsilon")
        _positive(self.clip_projected, "PCA projected-feature clip")
        if self.projected_size > self.latent_size:
            raise ValueError("PCA projected size cannot exceed latent state size")
        if not isinstance(self.projection, str) or not self.projection:
            raise ValueError("PCA projection path must be a non-empty string")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(
                "fixed-PCA latent-assisted node_ids must be non-empty and unique"
            )


@dataclass(frozen=True)
class SharedDynamicsEncoderControlObservationSpec:
    """A normalized history window encoded inside the trainable PPO policy."""

    provider: PretrainedLatentProviderSpec
    node_ids: tuple[str, ...]
    history_steps: int
    history_feature_size: int
    latent_size: int
    base: ControlObservationSpec
    name: Literal["shared-dynamics-encoder"] = field(
        default="shared-dynamics-encoder", init=False
    )

    def __post_init__(self) -> None:
        _positive(self.history_steps, "dynamics history steps")
        _positive(self.history_feature_size, "dynamics history feature size")
        _positive(self.latent_size, "dynamics latent size")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(
                "shared-dynamics-encoder node_ids must be non-empty and unique"
            )


@dataclass(frozen=True)
class TrainableNextLatentControlObservationSpec:
    """A causal history encoded by a randomly initialized PPO feature extractor."""

    provider: PretrainedLatentProviderSpec
    node_ids: tuple[str, ...]
    history_steps: int
    history_feature_size: int
    latent_size: int
    projected_size: int
    base: ControlObservationSpec
    num_layers: int = 1
    dropout: float = 0.0
    name: Literal["trainable-next-latent"] = field(
        default="trainable-next-latent", init=False
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("next-latent history steps", self.history_steps),
            ("next-latent history feature size", self.history_feature_size),
            ("next-latent latent size", self.latent_size),
            ("next-latent projected size", self.projected_size),
            ("next-latent encoder layers", self.num_layers),
        ):
            _positive(value, label)
        _probability(self.dropout, "next-latent encoder dropout", include_one=False)
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(
                "trainable-next-latent node_ids must be non-empty and unique"
            )


ControlPolicyObservationSpec: TypeAlias = (
    ControlObservationSpec
    | ForecastAssistedControlObservationSpec
    | LatentAssistedControlObservationSpec
    | ProjectedLatentAssistedControlObservationSpec
    | FixedPCALatentAssistedControlObservationSpec
    | SharedDynamicsEncoderControlObservationSpec
    | TrainableNextLatentControlObservationSpec
)


@dataclass(frozen=True)
class ForecastTrainingSpec:
    name: str
    dataset: DatasetSpec
    observation: ForecastInputSpec
    model: ForecastModelSpec
    objective: ForecastObjectiveSpec
    trajectory_data: ForecastTrajectoryDataSpec
    evaluation: ForecastEvaluationSpec | LatentEvaluationSpec
    optimizer: OptimizerSpec
    loop: ForecastLoopSpec
    training_seed: int
    normalization: ForecastNormalizationSpec = field(
        default_factory=ForecastNormalizationSpec,
        metadata={"canonical_omit_default": True},
    )
    output: OutputSpec = field(default_factory=OutputSpec)
    dependencies: tuple[str, ...] = ()
    kind: Literal["forecast"] = field(default="forecast", init=False)

    def __post_init__(self) -> None:
        _validate_job(self.name, self.training_seed, self.dependencies)
        next_latent_model = isinstance(self.model, NextLatentLSTMEncoderSpec)
        next_latent_objective = isinstance(self.objective, NextLatentObjectiveSpec)
        next_latent_evaluation = isinstance(self.evaluation, LatentEvaluationSpec)
        if len({next_latent_model, next_latent_objective, next_latent_evaluation}) != 1:
            raise ValueError(
                "next-latent model, objective, and evaluation must be configured together"
            )
        if next_latent_model:
            if self.observation.future_known:
                raise ValueError("next-latent prediction must not use future-known features")
            return

        auxiliary_next_latent_model = isinstance(
            self.model, AuxiliaryNextLatentLSTMEncoderSpec
        )
        auxiliary_next_latent_objective = isinstance(
            self.objective, AuxiliaryNextLatentObjectiveSpec
        )
        if auxiliary_next_latent_model != auxiliary_next_latent_objective:
            raise ValueError(
                "auxiliary next-latent model and objective must be configured together"
            )
        if auxiliary_next_latent_model:
            if not isinstance(self.evaluation, ForecastEvaluationSpec):
                raise ValueError(
                    "auxiliary next-latent forecasting requires forecast evaluation"
                )
            if not self.observation.future_known:
                raise ValueError(
                    "auxiliary next-latent forecasting requires future-known action features"
                )

        names = {head.name for head in self.model.heads}
        if isinstance(self.objective, OutcomeObjectiveSpec):
            references = {name for name, _ in self.objective.head_weights}
        elif isinstance(self.objective, DistillationObjectiveSpec):
            references = {self.objective.student_head}
        else:
            references = {"chlorine"}
        missing = references - names
        if missing:
            raise ValueError(f"objective references unknown model heads: {sorted(missing)}")
        missing = set(self.evaluation.targets) - names
        if missing:
            raise ValueError(f"evaluation references unknown model heads: {sorted(missing)}")
        if self.evaluation.effective_plots is not None:
            missing_breakdowns = (
                set(getattr(self.evaluation.effective_plots, "metric_breakdown_plots", ()))
                - set(self.evaluation.breakdowns)
            )
            if missing_breakdowns:
                raise ValueError(
                    "metric plots require matching evaluation breakdowns: "
                    f"{sorted(missing_breakdowns)}"
                )
        latent_model = isinstance(self.model, LatentDynamicsForecastSpec)
        latent_objective = isinstance(self.objective, LatentDynamicsObjectiveSpec)
        if latent_model != latent_objective:
            raise ValueError("latent-dynamics model and objective must be configured together")


@dataclass(frozen=True)
class ControlTrainingSpec:
    name: str
    dataset: DatasetSpec
    observation: ControlPolicyObservationSpec
    algorithm: RecurrentPPOSpec
    reward: RewardSpec
    training_seed: int
    total_timesteps: int
    max_injection: float = MAX_CHLORINE_INJECTION
    max_delta: float = 10_000.0
    initial_injection: float = 0.0
    output: OutputSpec = field(default_factory=OutputSpec)
    dependencies: tuple[str, ...] = ()
    hooks: ControlHooksSpec = field(default_factory=ControlHooksSpec)
    evaluate_test: bool = True
    kind: Literal["control"] = field(default="control", init=False)

    def __post_init__(self) -> None:
        _validate_job(self.name, self.training_seed, self.dependencies)
        _positive(self.total_timesteps, "total timesteps")
        _positive(self.max_injection, "maximum injection")
        _positive(self.max_delta, "maximum injection delta")
        if not 0 <= self.initial_injection <= self.max_injection:
            raise ValueError("initial injection must be between zero and maximum injection")
        if not isinstance(self.evaluate_test, bool):
            raise ValueError("evaluate_test must be a boolean")


@dataclass(frozen=True)
class ControlTrainingBaseSpec:
    """Partial, non-runnable configuration used to build control jobs."""

    dataset: DatasetSpec | None = None
    observation: ControlPolicyObservationSpec | None = None
    algorithm: RecurrentPPOSpec | None = None
    reward: RewardSpec | None = None
    training_seed: int | None = None
    total_timesteps: int | None = None
    max_injection: float | None = None
    max_delta: float | None = None
    initial_injection: float | None = None
    output: OutputSpec | None = None
    dependencies: tuple[str, ...] | None = None
    hooks: ControlHooksSpec | None = None
    evaluate_test: bool | None = None

    def build(self, *, name: str, **overrides: Any) -> ControlTrainingSpec:
        fields = tuple(self.__dataclass_fields__)
        unknown = sorted(set(overrides) - set(fields))
        if unknown:
            raise TypeError(f"unknown control training properties: {unknown}")

        values = {
            field_name: value
            for field_name in fields
            if (value := getattr(self, field_name)) is not None
        }
        values.update({field_name: value for field_name, value in overrides.items() if value is not None})
        required = ("dataset", "observation", "algorithm", "reward", "training_seed", "total_timesteps")
        missing = [field_name for field_name in required if values.get(field_name) is None]
        if missing:
            raise ValueError(f"missing required control training properties: {missing}")
        return ControlTrainingSpec(name=name, **values)


TrainingSpec: TypeAlias = ForecastTrainingSpec | ControlTrainingSpec


def _validate_job(name: str, training_seed: int, dependencies: tuple[str, ...]) -> None:
    _identifier(name, "job name")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError("training_seed must be an integer")
    if name in dependencies:
        raise ValueError(f"job {name!r} cannot depend on itself")
    if len(set(dependencies)) != len(dependencies):
        raise ValueError(f"job {name!r} has duplicate dependencies")
