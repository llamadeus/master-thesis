#!/usr/bin/env python3
"""Train an encoder-decoder LSTM to forecast node chlorine levels."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer import (  # noqa: E402
    EncoderDecoderLSTMSpec,
    Experiment,
    ForecastDiagnosticPlotsSpec,
    ForecastEvaluationSpec,
    ForecastFeatureSpec,
    ForecastHeadSpec,
    ForecastInputSpec,
    ForecastLoopSpec,
    ForecastTrainingSpec,
    ForecastTrajectoryDataSpec,
    OptimizerSpec,
    OutcomeObjectiveSpec,
    OutputSpec,
    main,
)
from training.datasets.hanoi import forecasting_full  # noqa: E402


HISTORY_STEPS = 48  # 24 hours at the Hanoi 30-minute sampling interval.
HORIZON_STEPS = 16  # 8 hours.

CYCLIC_TIME = ForecastFeatureSpec(
    name="cyclic-time",
    scope="global",
    cycles=("day", "week", "year"),
)
APPLIED_INJECTION = ForecastFeatureSpec(
    name="chlorine-injection",
    scope="source",
)

INPUT = ForecastInputSpec(
    history=(
        ForecastFeatureSpec(name="chlorine", scope="node"),
        APPLIED_INJECTION,
        CYCLIC_TIME,
    ),
    future_known=(APPLIED_INJECTION, CYCLIC_TIME),
)

CHLORINE_HEAD = ForecastHeadSpec(
    name="chlorine",
    target="chlorine",
    prediction="level",
    transform="identity",
)

MODEL = EncoderDecoderLSTMSpec(
    history_steps=HISTORY_STEPS,
    horizon_steps=HORIZON_STEPS,
    hidden_size=192,
    num_layers=1,
    dropout=0.0,
    decoder_mode="shared",
    heads=(CHLORINE_HEAD,),
)

EVALUATION = ForecastEvaluationSpec(
    targets=("chlorine",),
    metrics=("mae", "rmse"),
    breakdowns=("overall", "node", "horizon"),
    report_in_physical_units=True,
    scenario_weighting="equal",
    checkpoint_metric="chlorine.rmse",
    plots=ForecastDiagnosticPlotsSpec(
        splits=("validation", "test"),
        one_plot_per_scenario=True,
        scenario_trajectory_selection="first-fixed-schedule-pair",
        separate_target_files=True,
        node_selection="largest-actual-variation",
        rolling_prediction_steps=1,
        actual_line_style="solid",
        rolling_prediction_line_style="dashed",
        include_random_horizon_plot=True,
        random_horizon_include_past_predictions=True,
        random_horizon_line_style="dotted",
        random_seed=24_816,
    ),
)

TRAJECTORY_DATA = ForecastTrajectoryDataSpec(
    backend="epanet",
    cache_root="data/runs",
    materialize_missing=True,
    schedule_pairing="cartesian",
)


def build_experiment() -> Experiment:
    return Experiment(
        name="forecasting-encoder-decoder-lstm",
        description=(
            "Train an encoder-decoder LSTM to forecast eight hours of "
            "node-level chlorine concentrations from 24 hours of history."
        ),
        jobs=(
            ForecastTrainingSpec(
                name="encoder-decoder-lstm",
                dataset=forecasting_full,
                observation=INPUT,
                model=MODEL,
                objective=OutcomeObjectiveSpec(
                    loss="mse",
                    head_weights=(("chlorine", 1.0),),
                    normalize_targets_independently=True,
                ),
                optimizer=OptimizerSpec(
                    learning_rate=7.5e-4,
                    weight_decay=1e-5,
                ),
                loop=ForecastLoopSpec(epochs=100, batch_size=256),
                training_seed=102,
                # Fit feature-wise statistics on training windows and persist
                # them in the deployable forecast bundle.
                trajectory_data=TRAJECTORY_DATA,
                evaluation=EVALUATION,
                output=OutputSpec(group="forecasting/encoder-decoder-lstm"),
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(build_experiment()))
