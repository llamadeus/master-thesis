import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.analysis.dataset.export_arsenic_statistics import contamination_clusters, validation_catalog


def event(start, end):
    return dict(start_time_seconds=start, end_time_seconds=end)


class ContaminationClusterTests(unittest.TestCase):
    def test_delayed_tail_counts_after_injection_ends(self):
        values = np.zeros((21, 3))
        values[7:10, 1] = 0.02
        cluster, = contamination_clusters([event(2, 4)], np.arange(21), values)
        self.assertEqual(cluster['end_time_seconds'], 10)
        self.assertEqual(sum(p > 0.01 for p in cluster['node_peak_mg_l']), 1)

    def test_node_threshold_is_strict_and_not_aggregate(self):
        values = np.zeros((21, 3))
        values[3:6] = [0.01, 0.009, 0.008]
        cluster, = contamination_clusters([event(2, 4)], np.arange(21), values)
        self.assertEqual(sum(p > 0.01 for p in cluster['node_peak_mg_l']), 0)
        self.assertEqual(cluster['end_time_seconds'], 6)

    def test_overlapping_sources_form_one_cluster(self):
        clusters = contamination_clusters([event(2, 8), event(6, 10)], np.arange(21), np.zeros((21, 2)))
        self.assertEqual(clusters[0]['event_indices'], [0, 1])
        self.assertEqual(len(clusters), 1)

    def test_persistent_tail_merges_separated_source_intervals(self):
        values = np.zeros((21, 2))
        values[4:16, 0] = 0.1
        clusters = contamination_clusters([event(2, 4), event(10, 12)], np.arange(21), values)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]['end_time_seconds'], 16)

    def test_clearance_separates_clusters_but_dips_do_not_end_tails(self):
        values = np.zeros((21, 2))
        values[3:5, 0] = 0.1
        values[7, 0] = 0.1
        clusters = contamination_clusters([event(2, 4), event(12, 14)], np.arange(21), values)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]['end_time_seconds'], 8)
        self.assertEqual(clusters[1]['node_peak_mg_l'], [0, 0])

    def test_uncleared_final_tail_is_flagged(self):
        values = np.zeros((21, 2))
        values[3:, 0] = 0.1
        cluster, = contamination_clusters([event(2, 4)], np.arange(21), values)
        self.assertTrue(cluster['right_censored'])


class ValidationCatalogTests(unittest.TestCase):
    def test_schedule_statistics_and_reach_share_the_materialized_cases(self):
        runs = []
        for index in range(2):
            record = dict(index=index, scenario_id=f"scenario-{80 + index}",
                          schedule_index=3, schedule_hash=f"schedule-{index}",
                          cache_dir=f"cache-{index}", events=[event(2, 4)])
            trajectory = SimpleNamespace(
                time_seconds=np.arange(21), calendar_start_day=index,
                node_ids=("2", "3"), flow_units="LPS",
                arsenic=np.zeros((21, 2)), event_strength_by_node=np.zeros((21, 2)),
                hydraulic_outflow=np.ones((21, 2)))
            runs.append((record, trajectory))
        with patch("tools.analysis.dataset.export_arsenic_statistics.materialize", return_value=runs):
            catalog = validation_catalog(workers=1)
        self.assertEqual((catalog['dataset'], catalog['split'], catalog['seed']),
                         ('control_full', 'validation', 8129))
        self.assertEqual(catalog['duration_seconds'], 20)
        self.assertEqual(len(catalog['runs']), 2)
        for (source, trajectory), exported in zip(runs, catalog['runs']):
            self.assertEqual(exported['events'], source['events'])
            self.assertEqual(exported['schedule_hash'], source['schedule_hash'])
            self.assertEqual(exported['calendar_start_day'], trajectory.calendar_start_day)
            self.assertEqual(exported['clusters'][0]['event_indices'], [0])
            self.assertEqual(exported['clusters'][0]['node_peak_mg_l'], [0, 0])


if __name__ == '__main__':
    unittest.main()
