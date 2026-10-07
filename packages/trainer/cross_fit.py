"""Scenario-wise cross-fitting utilities for dataset specifications."""

from __future__ import annotations

from dataclasses import dataclass

from .specs import DatasetSpec, DatasetSplitSpec


@dataclass(frozen=True)
class CrossFitFold:
    """One forecast-training dataset and its scenario-wise held-out split."""

    index: int
    forecasting: DatasetSpec
    held_out: DatasetSplitSpec


def cross_fit_folds(
    dataset: DatasetSpec,
    fold_count: int,
) -> tuple[CrossFitFold, ...]:
    """Partition ``dataset.train`` by scenario in stable round-robin order."""
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise ValueError("cross-fit fold_count must be an integer")
    scenarios = dataset.train.scenarios
    if fold_count < 2 or fold_count > len(scenarios):
        raise ValueError(
            "cross-fit fold_count must lie between two and the number of "
            "training scenarios"
        )

    folds: list[CrossFitFold] = []
    for index in range(fold_count):
        held_out_scenarios = scenarios[index::fold_count]
        held_out_set = frozenset(held_out_scenarios)
        training_scenarios = tuple(
            scenario for scenario in scenarios if scenario not in held_out_set
        )
        held_out = DatasetSplitSpec(
            scenarios=held_out_scenarios,
            arsenic=dataset.train.arsenic,
            chlorine=dataset.train.chlorine,
        )
        forecasting = DatasetSpec(
            network=dataset.network,
            scenarios_dir=dataset.scenarios_dir,
            uncertainty_seed=dataset.uncertainty_seed,
            train=DatasetSplitSpec(
                scenarios=training_scenarios,
                arsenic=dataset.train.arsenic,
                chlorine=dataset.train.chlorine,
            ),
            validation=dataset.validation,
            test=dataset.test,
        )
        folds.append(CrossFitFold(index, forecasting, held_out))

    held_out_sets = [set(fold.held_out.scenarios) for fold in folds]
    if any(
        left & right
        for position, left in enumerate(held_out_sets)
        for right in held_out_sets[position + 1 :]
    ):
        raise RuntimeError("cross-fit held-out folds overlap")
    if set().union(*held_out_sets) != set(scenarios):
        raise RuntimeError("cross-fit held-out folds do not cover training scenarios")
    for fold in folds:
        training = set(fold.forecasting.train.scenarios)
        held_out = set(fold.held_out.scenarios)
        if training & held_out or training | held_out != set(scenarios):
            raise RuntimeError("cross-fit training and held-out scenarios are not complementary")
    return tuple(folds)


__all__ = ["CrossFitFold", "cross_fit_folds"]
