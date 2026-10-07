import unittest
import numpy as np

from tools.analysis.export_principal_v89_post_cluster import (
    aggregate, post_cluster_fractions, excess_partition, summarize_excess_partitions,
)
from tools.analysis.paper_arsenic_clearance import physical_projection


class PostClusterTests(unittest.TestCase):
    def arrays(self):
        time = np.arange(32) * 1800.
        strengths = np.zeros((32, 2))
        strengths[2:4, 0] = 1
        strengths[8:10, 1] = 1
        reference = np.zeros((32, 2))
        reference[3:7, 0] = .02
        reference[9:15, 1] = .02
        policy = np.zeros((32, 2))
        policy[3:5, 0] = .02
        policy[9:11, 1] = .02
        chlorine = np.full((31, 2), .4)
        chlorine[10, 0] = .41
        return dict(reference_time_seconds=time, time_seconds=time[1:],
                    reference_event_strength_by_node=strengths,
                    reference_transported_arsenic=reference,
                    transported_arsenic=policy[1:], chlorine=chlorine)

    def test_every_policy_cluster_has_its_own_clearance_and_followup(self):
        result = post_cluster_fractions(self.arrays())
        self.assertEqual([r["clearance_time_seconds"] for r in result], [5 * 1800, 11 * 1800])
        self.assertEqual(result[0]["fractions"][:3], [0, 0, 0])
        # The next source starts at step 8; its own first exceedance is at 9.
        self.assertTrue(all(v is None for v in result[0]["fractions"][3:]))
        self.assertEqual(result[1]["fractions"][0], .5)
        self.assertEqual(result[1]["fractions"][1], 0)
        self.assertEqual(result[1]["fractions"][20], 0)
        self.assertIsNone(result[1]["fractions"][21])
        self.assertTrue(all(len(r["fractions"]) == 25 for r in result))

    def test_absent_reference_or_policy_cluster_is_excluded(self):
        for key in ("reference_transported_arsenic", "transported_arsenic"):
            with self.subTest(key=key):
                arrays = self.arrays()
                arrays[key][:] = 0
                self.assertEqual(post_cluster_fractions(arrays), [])

    def test_twelve_hour_endpoint_is_included(self):
        arrays = self.arrays()
        arrays["reference_event_strength_by_node"][8:10] = 0
        arrays["transported_arsenic"][8:11] = 0
        arrays["chlorine"][28] = .5
        result = post_cluster_fractions(arrays)
        self.assertEqual(result[0]["clearance_time_seconds"], 5 * 1800)
        self.assertEqual(result[0]["fractions"][24], 1.)

    def test_clearance_at_episode_end_has_one_observation(self):
        arrays = self.arrays()
        arrays["transported_arsenic"][8:30, 1] = .02
        result = post_cluster_fractions(arrays)
        self.assertEqual(result[-1]["clearance_time_seconds"], 31 * 1800)
        self.assertEqual(result[-1]["fractions"][0], 0)
        self.assertTrue(all(v is None for v in result[-1]["fractions"][1:]))

    def test_hierarchy_and_support_do_not_pool_or_zero_fill(self):
        records = [dict(approach="baseline", seed=seed, scenario_id=scenario, schedule_hash=schedule,
                        fractions=[value] + [None] * 24)
                   for seed, scenario, schedule, value in ((17, "a", "one", 1.), (17, "a", "one", 1.),
                                                          (17, "a", "two", 0.), (17, "b", "one", 0.),
                                                          (42, "a", "one", 0.))]
        rows = aggregate(records)
        self.assertEqual(rows[0]["mean"], .125)
        self.assertEqual(rows[0]["observation_count"], 5)
        self.assertEqual(rows[0]["rollout_count"], 4)
        self.assertEqual(rows[0]["seed_count"], 2)
        self.assertIsNone(rows[1]["mean"])
        self.assertEqual(rows[1]["observation_count"], 0)

    def test_support_loses_interrupted_cluster_but_keeps_zero_fraction(self):
        records = [dict(approach="baseline", seed=17, scenario_id="a", schedule_hash="one", **r)
                   for r in post_cluster_fractions(self.arrays())]
        rows = aggregate(records)
        self.assertEqual(rows[0]["observation_count"], 2)
        self.assertEqual(rows[0]["mean"], .25)
        self.assertEqual(rows[2]["observation_count"], 2)
        self.assertEqual(rows[3]["observation_count"], 1)
        self.assertEqual(rows[3]["mean"], 0)
        self.assertEqual(rows[21]["observation_count"], 0)
        self.assertIsNone(rows[21]["mean"])

    def test_excess_partition_uses_reference_clearance_and_stops_at_next_source(self):
        times = np.arange(1201) * 1800.
        chlorine = np.full((1200, 2), .4)
        # Warm-up/boundary, interval start/end, interruption, and episode end.
        for step in (95, 96, 100, 103, 105, 124, 125, 126, 130, 152, 1200):
            chlorine[step - 1, 0] = .41
        source = np.zeros((1201, 2))
        source[99:101, 0] = 1
        source[125:127, 1] = 1
        reference = np.zeros((1201, 2))
        reference[100:104, 0] = .02
        reference[126:128, 1] = .02
        arrays = dict(time_seconds=times[1:], chlorine=chlorine, reference_time_seconds=times,
                      reference_event_strength_by_node=source, reference_transported_arsenic=reference)
        result = excess_partition(arrays)
        # Clearance is at 104/128. The next injection at 125 stops follow-up
        # even though its transported threshold exceedance begins only at 126.
        # The second 12-hour interval excludes step 152, its right endpoint.
        self.assertEqual(result, dict(inside_count=3, outside_count=2, total_count=5,
                                     normal_observation_count=(1008 - 6) * 2))
        arrays.update(transported_arsenic=np.zeros_like(chlorine), injection=np.zeros(1200))
        self.assertEqual(excess_partition(arrays), result)
        arrays["transported_arsenic"][:] = .02
        self.assertEqual(excess_partition(arrays), result)
        score, _ = physical_projection(arrays)
        counts = score["counts"]["normal_chlorine_excess_fraction"]
        self.assertEqual(result["total_count"], counts["numerator"])
        self.assertEqual(result["normal_observation_count"], counts["denominator"])

    def test_partition_includes_clearance_but_excludes_twelve_hour_endpoint(self):
        times = np.arange(1201) * 1800.
        source = np.zeros((1201, 1))
        source[99:101] = 1
        reference = np.zeros((1201, 1))
        reference[100:104] = .02
        chlorine = np.full((1200, 1), .4)
        chlorine[np.array([104, 127, 128]) - 1] = .41
        arrays = dict(time_seconds=times[1:], chlorine=chlorine, reference_time_seconds=times,
                      reference_event_strength_by_node=source, reference_transported_arsenic=reference)
        result = excess_partition(arrays)
        self.assertEqual((result["inside_count"], result["outside_count"]), (2, 1))
        arrays["reference_transported_arsenic"][:] = 0
        result = excess_partition(arrays)
        self.assertEqual((result["inside_count"], result["outside_count"]), (0, 3))

    def test_partition_truncates_followup_at_main_boundary_and_excludes_tail(self):
        times = np.arange(1201) * 1800.
        source = np.zeros((1201, 1))
        source[1097:1099] = 1
        reference = np.zeros((1201, 1))
        reference[1098:1104] = .02
        chlorine = np.full((1200, 1), .4)
        chlorine[np.array([1104, 1105, 1127, 1200]) - 1] = .41
        arrays = dict(time_seconds=times[1:], chlorine=chlorine, reference_time_seconds=times,
                      reference_event_strength_by_node=source, reference_transported_arsenic=reference)
        result = excess_partition(arrays)
        self.assertEqual((result["inside_count"], result["outside_count"]), (1, 0))
        arrays["reference_transported_arsenic"][1104] = .02
        result = excess_partition(arrays)
        self.assertEqual(result["total_count"], 0)

    def test_excess_shares_decompose_table_metric_with_different_denominators(self):
        records = [dict(approach="baseline", seed=17, scenario_id="a", inside_count=1,
                        outside_count=0, normal_observation_count=10),
                   dict(approach="baseline", seed=42, scenario_id="a", inside_count=9,
                        outside_count=90, normal_observation_count=100)]
        result = summarize_excess_partitions(records)[0]
        self.assertAlmostEqual(result["inside_metric_fraction"], .095)
        self.assertAlmostEqual(result["outside_metric_fraction"], .45)
        self.assertAlmostEqual(result["total_metric_fraction"], .545)
        self.assertAlmostEqual(result["inside_fraction"], .095 / .545)
        self.assertAlmostEqual(result["outside_fraction"], .45 / .545)
        self.assertEqual(result["total_count"], 100)

    def test_partition_components_use_schedule_scenario_and_seed_hierarchy(self):
        records = [dict(approach="baseline", seed=seed, scenario_id=scenario,
                        inside_count=inside, outside_count=outside, normal_observation_count=10)
                   for seed, scenario, inside, outside in (
                       (17, "a", 1, 0), (17, "a", 0, 0), (17, "b", 0, 2), (42, "a", 0, 4))]
        result = summarize_excess_partitions(records)[0]
        self.assertAlmostEqual(result["inside_metric_fraction"], .0125)
        self.assertAlmostEqual(result["outside_metric_fraction"], .25)
        self.assertAlmostEqual(result["total_metric_fraction"], .2625)


if __name__ == "__main__":
    unittest.main()
