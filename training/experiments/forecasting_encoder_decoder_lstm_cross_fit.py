#!/usr/bin/env python3
"""Train five scenario-wise cross-fitted chlorine forecasters."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.trainer import Experiment, OutputSpec, cross_fit_folds
from packages.trainer import main as trainer_main  # noqa: E402
from training.datasets.hanoi import forecasting_full  # noqa: E402
from training.experiments.forecasting_encoder_decoder_lstm import \
    build_experiment as build_reference_experiment  # noqa: E402

FOLD_COUNT = 5
FOLDS = cross_fit_folds(forecasting_full, FOLD_COUNT)
OUTPUT_GROUP = "forecasting/encoder-decoder-lstm-cross-fit"


def build_experiment(*, training_seed: int | None = None) -> Experiment:
    reference = build_reference_experiment().jobs[0]
    if training_seed is not None:
        reference = replace(reference, training_seed=training_seed)
    fold_jobs = tuple(
        replace(
            reference,
            name=f"encoder-decoder-lstm-fold-{fold.index}",
            dataset=fold.forecasting,
            output=OutputSpec(group=OUTPUT_GROUP),
        )
        for fold in FOLDS
    )
    return Experiment(
        name="forecasting-encoder-decoder-lstm-cross-fit",
        description=(
            "Train five scenario-wise out-of-fold chlorine forecasters for CEM "
            "teacher labeling and supporting forecast evaluation."
        ),
        jobs=fold_jobs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--seed", "--training-seed", type=int, dest="training_seed",
        help="override the training seed for all five fold models",
    )
    args, trainer_args = parser.parse_known_args(argv)
    if not trainer_args or any(arg in {"-h", "--help", "help"} for arg in trainer_args):
        parser.print_help()
        print()
    return trainer_main(
        build_experiment(training_seed=args.training_seed), argv=trainer_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
