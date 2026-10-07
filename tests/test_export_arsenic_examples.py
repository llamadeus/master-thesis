from types import SimpleNamespace
import unittest

import numpy as np

from tools.analysis.dataset.export_arsenic_examples import (
    ZeroTrajectoryResolver, candidates, clean_window, select_example, validate_zero,
)


def trajectory():
    times = np.arange(0, 96 * 3600 + 1, 1800)
    return SimpleNamespace(time_seconds=times, node_ids=("13", "3", "4", "5"),
                           arsenic=np.zeros((len(times), 4)),
                           chlorine=np.zeros((len(times), 4)),
                           injection=np.zeros(len(times)), scenario_id="fixture")


def event(node, start=12, end=24):
    return dict(node_id=node, start_time_seconds=start * 3600,
                end_time_seconds=end * 3600, strength=100)


class ArsenicExampleSelectionTests(unittest.TestCase):
    def test_analysis_requires_endpoint_even_when_forecast_grid_is_complete(self):
        run = trajectory()
        run.arsenic_schedule_hash = "arsenic"
        run.chlorine_profile_hash = "chlorine"
        reference = SimpleNamespace(scenario_id="fixture", arsenic_schedule_hash="arsenic",
                                    chlorine_schedule_hash="chlorine")
        dataset = SimpleNamespace(duration_seconds=96 * 3600, hydraulic_timestep_seconds=1800)
        resolver = ZeroTrajectoryResolver(dataset, None, None)
        resolver._validate_trajectory(run, reference)
        run.time_seconds = run.time_seconds[:-1]
        with self.assertRaisesRegex(ValueError, "endpoint"):
            resolver._validate_trajectory(run, reference)

    def test_zero_policy_does_not_imply_chlorfree_state(self):
        run = trajectory()
        run.chlorine[0, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "zero chlorine"):
            validate_zero(run)

    def test_terminal_source_is_not_an_example(self):
        run = trajectory()
        run.arsenic[24:48, 0] = 1
        record = dict(index=0, events=[event("13")])
        self.assertFalse(candidates(record, run)["mild"])
        self.assertFalse(candidates(record, run)["strong"])

    def test_strength_groups_take_precedence_over_visual_score(self):
        pool = [dict(run_index=i, score=10-i, nodes=["3", "4"],
                     diagnostics={"source_peak_mg_l": peak})
                for i, peak in enumerate((0.2, 0.4, 1.0, 3.0, 7.0))]
        self.assertEqual(select_example(pool, "mild")["run_index"], 0)
        self.assertEqual(select_example(pool, "strong")["run_index"], 3)
        self.assertIsNone(select_example(pool, "strong", requested=0))

    def test_strong_prefers_broad_transport_within_upper_quartile(self):
        pool = [dict(run_index=i, score=10-i, nodes=["3", "4"],
                     diagnostics={"source_peak_mg_l": peak})
                for i, peak in enumerate((0.2, 0.4, 1.0, 3.0, 7.0))]
        pool[0]["nodes"] = ["3", "4", "5", "6", "7"]
        pool[4]["nodes"] = ["3", "4", "5", "6"]
        self.assertEqual(select_example(pool, "strong")["run_index"], 4)

    def test_strong_includes_earlier_arrival_without_changing_mild(self):
        run = trajectory()
        run.arsenic[24:48, 1] = 1
        run.arsenic[26:50, 0] = 0.8
        run.arsenic[28:52, 2] = 0.6
        run.arsenic[31:55, 3] = 0.4
        found = candidates(dict(index=2, events=[event("3")]), run)
        self.assertEqual(found["strong"][0]["nodes"], ["3", "13", "4", "5"])
        self.assertEqual(found["mild"][0]["nodes"], ["3", "4", "5"])

    def test_transport_requires_delay_and_visible_attenuated_wave(self):
        run = trajectory()
        run.arsenic[24:48, 1] = 1
        run.arsenic[32:56, 2] = 0.4
        record = dict(index=2, events=[event("3")])
        found = candidates(record, run)["strong"][0]
        self.assertEqual(found["nodes"], ["3", "4"])
        self.assertEqual(found["diagnostics"]["downstream"][0]["delay_hours"], 4)
        run.arsenic[:, 2] *= 0.01
        self.assertFalse(candidates(record, run)["strong"])

    def test_prior_contamination_rejects_attribution(self):
        run = trajectory()
        run.arsenic[15, 2] = 0.1
        self.assertIsNone(clean_window(run, [event("3")], (0,)))

    def test_overlap_is_source_overlap_not_only_concentration_tail(self):
        run = trajectory()
        run.arsenic[24:48, 1] = 1
        run.arsenic[36:60, 2] = 0.7
        record = dict(index=4, events=[event("3"), event("4", 18, 30)])
        self.assertEqual(candidates(record, run)["overlap"][0]["diagnostics"]["overlap_hours"], 6)
        record["events"][1] = event("4", 24, 30)
        self.assertFalse(candidates(record, run)["overlap"])

    def test_next_event_must_not_truncate_visible_transport_tail(self):
        run = trajectory()
        run.arsenic[24:48, 1] = 1
        run.arsenic[32:120, 2] = 0.4
        events = [event("3"), event("5", 40, 50)]
        self.assertIsNone(clean_window(run, events, (0,)))


if __name__ == "__main__":
    unittest.main()
