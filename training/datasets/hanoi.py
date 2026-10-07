"""Readable, named Hanoi episode datasets."""

from packages.trainer import (V70_MAX_CHLORINE_INJECTION,
                              ArsenicScheduleStreamSpec,
                              ChlorineInjectionScheduleGeneratorSpec,
                              DatasetSpec, DatasetSplitSpec,
                              FixedArsenicScheduleSetSpec,
                              FixedChlorineInjectionScheduleSetSpec,
                              HydraulicPeakArsenicScheduleGeneratorSpec)

TRAIN_SCENARIOS_FULL = tuple(f"scenario-{index}" for index in range(80))
TRAIN_SCENARIOS_MINIMAL = tuple(f"scenario-{index}" for index in range(16))

VALIDATION_SCENARIOS_FULL = tuple(f"scenario-{index}" for index in range(80, 90))
VALIDATION_SCENARIOS_MINIMAL = tuple(f"scenario-{index}" for index in range(80, 82))

TEST_SCENARIOS_FULL = tuple(f"scenario-{index}" for index in range(90, 100))
TEST_SCENARIOS_MINIMAL = tuple(f"scenario-{index}" for index in range(90, 92))

CHLORINE_SCHEDULE_GENERATOR = ChlorineInjectionScheduleGeneratorSpec(
    families=("constant", "periodic", "piecewise-constant", "stairs", "random-walk"),
    mix_families_within_schedule=True,
    set_coverage="balanced",
    transition_style="mostly-smooth-with-occasional-ramps-and-abrupt-changes",
    minimum_injection=0.0,
    maximum_injection=V70_MAX_CHLORINE_INJECTION,
    sampling_bands=(
        (0.0, 200_000.0, 0.70),
        (200_000.0, 350_000.0, 0.20),
        (350_000.0, V70_MAX_CHLORINE_INJECTION, 0.10),
    ),
)

ARSENIC_SCHEDULE_GENERATOR = HydraulicPeakArsenicScheduleGeneratorSpec(
    concentration_distribution="log-uniform"
)


forecasting_full = DatasetSpec(
    network="hanoi",
    scenarios_dir="hanoi",
    uncertainty_seed=42,
    train=DatasetSplitSpec(
        scenarios=TRAIN_SCENARIOS_FULL,
        arsenic=FixedArsenicScheduleSetSpec(seed=6_103, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
        chlorine=FixedChlorineInjectionScheduleSetSpec(
            seed=7_103,
            schedule_count=5,
            generator=CHLORINE_SCHEDULE_GENERATOR,
        ),
    ),
    validation=DatasetSplitSpec(
        scenarios=VALIDATION_SCENARIOS_FULL,
        arsenic=FixedArsenicScheduleSetSpec(seed=6_104, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
        chlorine=FixedChlorineInjectionScheduleSetSpec(
            seed=7_104,
            schedule_count=5,
            generator=CHLORINE_SCHEDULE_GENERATOR,
        ),
    ),
    test=DatasetSplitSpec(
        scenarios=TEST_SCENARIOS_FULL,
        arsenic=FixedArsenicScheduleSetSpec(seed=6_105, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
        chlorine=FixedChlorineInjectionScheduleSetSpec(
            seed=7_105,
            schedule_count=5,
            generator=CHLORINE_SCHEDULE_GENERATOR,
        ),
    ),
)


control_full = DatasetSpec(
    network="hanoi",
    scenarios_dir="hanoi",
    uncertainty_seed=42,
    train=DatasetSplitSpec(
        scenarios=TRAIN_SCENARIOS_FULL,
        arsenic=ArsenicScheduleStreamSpec(seed=8_128, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
    validation=DatasetSplitSpec(
        scenarios=VALIDATION_SCENARIOS_FULL,
        arsenic=FixedArsenicScheduleSetSpec(seed=8_129, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
    test=DatasetSplitSpec(
        scenarios=TEST_SCENARIOS_FULL,
        arsenic=FixedArsenicScheduleSetSpec(seed=8_130, schedule_count=10, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
)


control_minimal = DatasetSpec(
    network="hanoi",
    scenarios_dir="hanoi",
    uncertainty_seed=42,
    train=DatasetSplitSpec(
        scenarios=TRAIN_SCENARIOS_MINIMAL,
        arsenic=ArsenicScheduleStreamSpec(seed=8_128, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
    validation=DatasetSplitSpec(
        scenarios=VALIDATION_SCENARIOS_MINIMAL,
        arsenic=FixedArsenicScheduleSetSpec(seed=8_129, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
    test=DatasetSplitSpec(
        scenarios=TEST_SCENARIOS_MINIMAL,
        arsenic=FixedArsenicScheduleSetSpec(seed=8_130, schedule_count=5, generator=ARSENIC_SCHEDULE_GENERATOR),
    ),
)


normal_operation_full = DatasetSpec(
    network="hanoi",
    scenarios_dir="hanoi",
    uncertainty_seed=42,
    train=DatasetSplitSpec(scenarios=TRAIN_SCENARIOS_FULL),
    validation=DatasetSplitSpec(scenarios=VALIDATION_SCENARIOS_FULL),
    test=DatasetSplitSpec(scenarios=TEST_SCENARIOS_FULL),
)

normal_operation_minimal = DatasetSpec(
    network="hanoi",
    scenarios_dir="hanoi",
    uncertainty_seed=42,
    train=DatasetSplitSpec(scenarios=TRAIN_SCENARIOS_MINIMAL),
    validation=DatasetSplitSpec(scenarios=VALIDATION_SCENARIOS_MINIMAL),
    test=DatasetSplitSpec(scenarios=TEST_SCENARIOS_MINIMAL),
)
