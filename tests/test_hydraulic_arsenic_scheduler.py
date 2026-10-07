from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from packages.trainer.data import ResolvedDataset, ScenarioEntry
from packages.trainer.schedules.arsenic import (
    ArsenicEvent,
    ArsenicScheduleStream,
    _EventPlacement,
    _generate_schedule,
    _hydraulically_feasible_start_half_hours,
    _peak_concentration_from_quantile,
    _valid_event_placements,
    fixed_schedule,
)
from packages.trainer.serialization import canonical_data
from packages.trainer.simulation import (
    HydraulicOutflowProfile,
    prepare_hydraulic_outflow_cache,
)
from packages.trainer.specs import (
    ArsenicScheduleGeneratorSpec,
    ArsenicScheduleStreamSpec,
    FixedArsenicScheduleSetSpec,
    HydraulicPeakArsenicScheduleGeneratorSpec,
)
from training.datasets.hanoi import ARSENIC_SCHEDULE_GENERATOR


def _scenario(
    root: Path, split: str = "train", scenario_id: str = "scenario"
) -> ScenarioEntry:
    path = root / scenario_id
    path.mkdir(parents=True)
    (path / "scenario.inp").write_text("[JUNCTIONS]\nn1 0 0\nn2 0 0\n")
    (path / "scenario.epytflow_scenario_config").write_text("fake config\n")
    (path / "scenario.msx").write_text("fake msx\n")
    (path / "manifest.json").write_text("{}\n")
    return ScenarioEntry(
        scenario_id, split, path, 0, (), ("n1", "n2"), ("n1", "n2"), ()
    )


class HydraulicPeakArsenicSchedulerTests(unittest.TestCase):
    def test_week_slot_candidates_include_every_eligible_node_and_week(self):
        duration_seconds = 4 * 1_800
        events = [ArsenicEvent("n1", 10 * 1_800, 14 * 1_800, 1_000.0)]

        placements, rejected = _valid_event_placements(
            events,
            ("n2", "n1"),
            0,
            2 * 7 * 48,
            duration_seconds,
            10,
            ArsenicScheduleGeneratorSpec(),
        )

        self.assertEqual(rejected, [])
        self.assertEqual(
            [(item.node_id, item.start_half_hour) for item in placements],
            [("n1", 346), ("n2", 10), ("n2", 346)],
        )

    def test_week_slot_checks_all_nodes_before_retrying(self):
        timestep = 1_800
        duration = 2 * 7 * 86_400
        times = np.arange(0, duration + timestep, timestep, dtype=np.int64)
        n1 = np.where((times // timestep) % 2 == 0, 1.0, 2_000.0)
        n2 = np.full(len(times), 10.0)
        profile = HydraulicOutflowProfile(
            "profile-a", "scenario", times, ("n1", "n2"),
            np.column_stack((n1, n2)),
        )

        placements, rejected = _valid_event_placements(
            [],
            ("n1", "n2"),
            0,
            2 * 7 * 48,
            4 * timestep,
            0,
            HydraulicPeakArsenicScheduleGeneratorSpec(),
            profile,
        )

        self.assertTrue(rejected)
        self.assertEqual({item.node_id for item in placements}, {"n2"})
        self.assertEqual(
            [item.start_half_hour for item in placements],
            [0, 7 * 48],
        )

    def test_empty_week_slot_retries_without_replacement(self):
        class _ScriptedRandom:
            operations = []
            integers = iter((1, 4, 5, 5, 0))

            def __init__(self, _seed):
                pass

            def integer(self, low, high):
                value = next(self.integers)
                self.operations.append(("integer", low, high, value))
                return value

            def uniform(self, low, high):
                self.operations.append(("uniform", low, high))
                return 0.5

        placement = _EventPlacement("n1", 6)
        with (
            patch(
                "packages.trainer.schedules.arsenic._DeterministicRandom",
                _ScriptedRandom,
            ),
            patch(
                "packages.trainer.schedules.arsenic._valid_event_placements",
                side_effect=[([], []), ([placement], [])],
            ) as candidates,
        ):
            schedule = _generate_schedule(
                ("n1",),
                identity=("test",),
                generator=ArsenicScheduleGeneratorSpec("uniform", 1_000, 2_000),
                warmup_days=0,
                main_days=7,
                minimum_event_count=1,
                maximum_event_count=1,
                minimum_event_duration_steps=4,
                maximum_event_duration_steps=4,
            )

        self.assertEqual(
            [call.args[5] for call in candidates.call_args_list],
            [5, 6],
        )
        self.assertEqual(schedule.events[0].start_time_seconds, 6 * 1_800)
        self.assertEqual(_ScriptedRandom.operations[-1][0], "uniform")

    def test_strength_is_sampled_after_global_placement_selection(self):
        class _RecordingRandom:
            operations = []

            def __init__(self, _seed):
                pass

            def integer(self, low, high):
                self.operations.append(("integer", low, high))
                return low

            def uniform(self, low, high):
                self.operations.append(("uniform", low, high))
                return 0.5

        with patch(
            "packages.trainer.schedules.arsenic._DeterministicRandom",
            _RecordingRandom,
        ):
            schedule = _generate_schedule(
                ("n1",),
                identity=("test",),
                generator=ArsenicScheduleGeneratorSpec("uniform", 1_000, 2_000),
                warmup_days=0,
                main_days=7,
                minimum_event_count=1,
                maximum_event_count=1,
                minimum_event_duration_steps=4,
                maximum_event_duration_steps=4,
            )

        self.assertEqual(
            [item[0] for item in _RecordingRandom.operations],
            ["integer", "integer", "integer", "integer", "uniform"],
        )
        self.assertEqual(schedule.events[0].strength, 1_500.0)

    def test_global_node_week_candidates_use_one_uniform_selection(self):
        class _LastPlacementRandom:
            integers = iter((1, 4, 10, 3))

            def __init__(self, _seed):
                pass

            def integer(self, _low, _high):
                return next(self.integers)

            def uniform(self, _low, _high):
                return 0.5

        with patch(
            "packages.trainer.schedules.arsenic._DeterministicRandom",
            _LastPlacementRandom,
        ):
            schedule = _generate_schedule(
                ("n2", "n1"),
                identity=("test",),
                generator=ArsenicScheduleGeneratorSpec("uniform", 1_000, 2_000),
                warmup_days=0,
                main_days=14,
                minimum_event_count=1,
                maximum_event_count=1,
                minimum_event_duration_steps=4,
                maximum_event_duration_steps=4,
            )

        event = schedule.events[0]
        self.assertEqual(event.node_id, "n2")
        self.assertEqual(event.start_time_seconds, 346 * 1_800)

    def test_placement_fails_only_after_every_week_slot_is_exhausted(self):
        class _LowestRandom:
            def __init__(self, _seed):
                pass

            def integer(self, low, _high):
                return low

        with (
            patch(
                "packages.trainer.schedules.arsenic._DeterministicRandom",
                _LowestRandom,
            ),
            patch(
                "packages.trainer.schedules.arsenic._valid_event_placements",
                return_value=([], []),
            ) as candidates,
            self.assertRaisesRegex(RuntimeError, "without same-node overlap"),
        ):
            _generate_schedule(
                ("n1",),
                identity=("test",),
                generator=ArsenicScheduleGeneratorSpec(),
                warmup_days=0,
                main_days=7,
                minimum_event_count=1,
                maximum_event_count=1,
                minimum_event_duration_steps=4,
                maximum_event_duration_steps=4,
            )

        self.assertEqual(candidates.call_count, 7 * 48)

    def test_event_durations_use_the_inclusive_half_hour_grid(self):
        duration_seconds = []
        start_seconds = []
        for schedule_index in range(200):
            schedule = fixed_schedule(
                FixedArsenicScheduleSetSpec(77, 200),
                "scenario",
                schedule_index,
                ("n1", "n2"),
            )
            duration_seconds.extend(
                event.end_time_seconds - event.start_time_seconds
                for event in schedule.events
            )
            start_seconds.extend(event.start_time_seconds for event in schedule.events)

        self.assertTrue(duration_seconds)
        self.assertEqual(min(duration_seconds), 2 * 3600)
        self.assertEqual(max(duration_seconds), 72 * 3600)
        self.assertTrue(all(value % 1800 == 0 for value in duration_seconds))
        self.assertTrue(any(value % 3600 == 1800 for value in duration_seconds))
        self.assertTrue(all(value % 1800 == 0 for value in start_seconds))
        self.assertTrue(any(value % 3600 == 1800 for value in start_seconds))

    def test_hanoi_uses_log_uniform_peak_targets(self):
        self.assertEqual(
            ARSENIC_SCHEDULE_GENERATOR.concentration_distribution,
            "log-uniform",
        )

    def test_log_gaussian_configuration_and_quantiles(self):
        generator = HydraulicPeakArsenicScheduleGeneratorSpec()
        lower = math.log(0.01)
        upper = math.log(10.0)
        self.assertEqual(generator.mode, "hydraulic-local-peak")
        self.assertEqual(
            generator.minimum_local_concentration_mg_per_liter, 0.01
        )
        self.assertEqual(
            generator.maximum_local_concentration_mg_per_liter, 10.0
        )
        self.assertEqual(generator.maximum_mass_rate_mg_per_minute, 350_000.0)
        configured = canonical_data(generator)
        self.assertEqual(
            configured["minimum_local_concentration_mg_per_liter"], 0.01
        )
        self.assertEqual(
            configured["maximum_local_concentration_mg_per_liter"], 10.0
        )
        self.assertEqual(
            configured["maximum_mass_rate_mg_per_minute"], 350_000.0
        )
        self.assertNotIn("minimum_peak_concentration_mg_per_liter", configured)
        self.assertNotIn("maximum_peak_concentration_mg_per_liter", configured)
        self.assertAlmostEqual(generator.log_peak_mean, (lower + upper) / 2.0)
        self.assertAlmostEqual(
            generator.log_peak_variance, ((upper - lower) / 4.0) ** 2
        )
        self.assertAlmostEqual(
            generator.median_peak_concentration_mg_per_liter,
            math.sqrt(0.01 * 10.0),
        )
        self.assertEqual(_peak_concentration_from_quantile(0.0, generator), 0.01)
        self.assertEqual(_peak_concentration_from_quantile(1.0, generator), 10.0)

        quantiles = np.linspace(0.0001, 0.9999, 20_001)
        samples = np.asarray([
            _peak_concentration_from_quantile(float(value), generator)
            for value in quantiles
        ])
        self.assertTrue(np.all((0.01 <= samples) & (samples <= 10.0)))
        self.assertAlmostEqual(float(np.mean(np.log(samples))), generator.log_peak_mean,
                               delta=1e-3)
        self.assertAlmostEqual(float(np.median(samples)),
                               generator.median_peak_concentration_mg_per_liter,
                               delta=1e-3)
        shifted = HydraulicPeakArsenicScheduleGeneratorSpec(
            log_gaussian_midpoint_mg_per_liter=5.0
        )
        self.assertAlmostEqual(shifted.log_peak_mean, math.log(5.0))
        self.assertAlmostEqual(
            shifted.median_peak_concentration_mg_per_liter, 5.0
        )
        self.assertNotIn("log_gaussian_midpoint_mg_per_liter", canonical_data(generator))
        self.assertEqual(
            canonical_data(shifted)["log_gaussian_midpoint_mg_per_liter"], 5.0
        )
        self.assertNotEqual(
            _peak_concentration_from_quantile(0.5, shifted),
            _peak_concentration_from_quantile(0.5, generator),
        )
        log_uniform = HydraulicPeakArsenicScheduleGeneratorSpec(
            concentration_distribution="log-uniform"
        )
        self.assertEqual(
            _peak_concentration_from_quantile(0.0, log_uniform), 0.01
        )
        self.assertAlmostEqual(
            _peak_concentration_from_quantile(0.5, log_uniform),
            math.sqrt(0.01 * 10.0),
        )
        self.assertEqual(
            _peak_concentration_from_quantile(1.0, log_uniform), 10.0
        )
        self.assertEqual(
            _peak_concentration_from_quantile(
                0.0, log_uniform, lower_bound_mg_per_liter=0.04
            ),
            0.04,
        )
        self.assertAlmostEqual(
            _peak_concentration_from_quantile(
                0.5,
                log_uniform,
                lower_bound_mg_per_liter=0.04,
                upper_bound_mg_per_liter=2.0,
            ),
            math.sqrt(0.04 * 2.0),
        )
        log_uniform_samples = np.asarray([
            _peak_concentration_from_quantile(float(value), log_uniform)
            for value in quantiles
        ])
        self.assertAlmostEqual(
            float(np.mean(np.log(log_uniform_samples))),
            (lower + upper) / 2.0,
            delta=1e-3,
        )
        for values in (
            {"concentration_distribution": "uniform"},
            {"minimum_local_concentration_mg_per_liter": 0.0},
            {"maximum_local_concentration_mg_per_liter": math.inf},
            {
                "minimum_local_concentration_mg_per_liter": 1.0,
                "maximum_local_concentration_mg_per_liter": 1.0,
            },
            {"maximum_mass_rate_mg_per_minute": 0.0},
            {"maximum_mass_rate_mg_per_minute": math.inf},
            {"log_gaussian_midpoint_mg_per_liter": 10.0},
            {
                "concentration_distribution": "log-uniform",
                "log_gaussian_midpoint_mg_per_liter": 5.0,
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HydraulicPeakArsenicScheduleGeneratorSpec(**values)

    def test_mass_rate_enforces_event_window_local_concentration_bounds(self):
        timestep = 1_800
        duration = 25 * 86_400
        times = np.arange(0, duration + timestep, timestep, dtype=np.int64)
        n1 = np.where((times // timestep) % 2 == 0, 10.0, 40.0)
        n2 = np.full(len(times), 20.0)
        profile = HydraulicOutflowProfile(
            "profile-a", "scenario", times, ("n1", "n2"),
            np.column_stack((n1, n2)),
        )
        hydraulic = FixedArsenicScheduleSetSpec(
            77, 1, HydraulicPeakArsenicScheduleGeneratorSpec()
        )
        legacy = FixedArsenicScheduleSetSpec(
            77, 1, ArsenicScheduleGeneratorSpec()
        )
        schedule = fixed_schedule(
            hydraulic, "scenario", 0, ("n1", "n2"), hydraulic_profile=profile
        )
        legacy_schedule = fixed_schedule(legacy, "scenario", 0, ("n1", "n2"))

        self.assertEqual(
            [(event.node_id, event.start_time_seconds, event.end_time_seconds)
             for event in schedule.events],
            [(event.node_id, event.start_time_seconds, event.end_time_seconds)
             for event in legacy_schedule.events],
        )
        self.assertEqual(schedule.hydraulic_profile_identity, "profile-a")
        for event in schedule.events:
            target = event.target_peak_concentration_mg_per_liter
            self.assertIsNotNone(target)
            assert target is not None
            minimum, maximum = profile.outflow_range_liters_per_minute(
                event.node_id, event.start_time_seconds, event.end_time_seconds
            )
            self.assertAlmostEqual(event.strength, target * minimum)
            self.assertAlmostEqual(event.strength / minimum, target)
            self.assertGreaterEqual(target, 0.01 * maximum / minimum)
            self.assertLessEqual(target, 10.0)
            column = profile.node_ids.index(event.node_id)
            active = (
                (times >= event.start_time_seconds)
                & (times < event.end_time_seconds)
            )
            local_concentrations = (
                event.strength
                / profile.outflow_liters_per_minute[active, column]
            )
            self.assertAlmostEqual(float(np.max(local_concentrations)), target)
            self.assertGreaterEqual(float(np.min(local_concentrations)), 0.01)
            self.assertLessEqual(float(np.max(local_concentrations)), 10.0)
            self.assertLessEqual(event.strength, 350_000.0)

        other_profile = HydraulicOutflowProfile(
            "profile-b", "scenario", times, profile.node_ids,
            profile.outflow_liters_per_minute,
        )
        other = fixed_schedule(
            hydraulic, "scenario", 0, ("n1", "n2"),
            hydraulic_profile=other_profile,
        )
        self.assertNotEqual(schedule.hash, other.hash)

    def test_hydraulically_infeasible_windows_are_filtered(self):
        timestep = 1_800
        times = np.arange(0, 4 * 3600, timestep, dtype=np.int64)
        outflow = np.asarray([1.0, 2_000.0, 1.0, 2_000.0, 10.0, 10.0, 10.0, 10.0])
        profile = HydraulicOutflowProfile(
            "profile-a", "scenario", times, ("n1",), outflow[:, None]
        )

        feasible, windows, rejected = _hydraulically_feasible_start_half_hours(
            profile,
            "n1",
            [0, 4],
            2 * 3600,
            HydraulicPeakArsenicScheduleGeneratorSpec(),
        )

        self.assertEqual(feasible, [4])
        self.assertEqual(set(windows), {4})
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0][:2], ("n1", 0))

    def test_mass_rate_cap_reduces_the_event_peak_upper_bound(self):
        timestep = 1_800
        duration = 25 * 86_400
        times = np.arange(0, duration + timestep, timestep, dtype=np.int64)
        profile = HydraulicOutflowProfile(
            "profile-a",
            "scenario",
            times,
            ("n1",),
            np.full((len(times), 1), 100_000.0),
        )
        spec = FixedArsenicScheduleSetSpec(
            77,
            1,
            HydraulicPeakArsenicScheduleGeneratorSpec(
                concentration_distribution="log-uniform"
            ),
        )

        schedule = fixed_schedule(
            spec, "scenario", 0, ("n1",), hydraulic_profile=profile
        )

        for event in schedule.events:
            self.assertLessEqual(
                event.target_peak_concentration_mg_per_liter, 3.5
            )
            self.assertLessEqual(event.strength, 350_000.0)

    def test_fully_infeasible_local_bounds_fail_with_diagnostics(self):
        timestep = 1_800
        duration = 25 * 86_400
        times = np.arange(0, duration + timestep, timestep, dtype=np.int64)
        outflow = np.where((times // timestep) % 2 == 0, 1.0, 2_000.0)
        profile = HydraulicOutflowProfile(
            "profile-a", "scenario", times, ("n1",), outflow[:, None]
        )
        spec = FixedArsenicScheduleSetSpec(
            77, 1, HydraulicPeakArsenicScheduleGeneratorSpec()
        )

        with self.assertRaisesRegex(
            ValueError,
            "could not place a hydraulically feasible arsenic event.*scenario=scenario",
        ):
            fixed_schedule(
                spec, "scenario", 0, ("n1",), hydraulic_profile=profile
            )

    def test_stream_resume_rejects_a_changed_hydraulic_catalog(self):
        spec = ArsenicScheduleStreamSpec(
            7, generator=HydraulicPeakArsenicScheduleGeneratorSpec()
        )
        state = ArsenicScheduleStream(spec, "catalog-a").state_dict()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ArsenicScheduleStream(spec, "catalog-b").load_state_dict(state)

    def test_legacy_dataset_does_no_hydraulic_work(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scenario = _scenario(root)
            schedule = FixedArsenicScheduleSetSpec(
                7, 1, ArsenicScheduleGeneratorSpec()
            )
            dataset = ResolvedDataset(
                "net", 60, 180, (scenario,), schedule, None, None
            )
            backend = Mock()
            catalog = prepare_hydraulic_outflow_cache(
                dataset, backend, cache_root=root / "hydraulics"
            )
            self.assertIsNone(catalog)
            backend.prepare_hydraulics.assert_not_called()
            backend.create_session.assert_not_called()
            self.assertFalse((root / "hydraulics").exists())



if __name__ == "__main__":
    unittest.main()
